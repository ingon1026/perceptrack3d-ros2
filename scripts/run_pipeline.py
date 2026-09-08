"""전체 파이프라인 실행: 2D 검출(JSON) + LiDAR → 3D 객체 (→ 추적) 를 전 프레임에 돌리고 결과·런타임을 저장한다.

실행
    .venv/bin/python scripts/run_pipeline.py --variant raw        [--frames 0 297]
    .venv/bin/python scripts/run_pipeline.py --variant clustered  [--frames 0 297]
    .venv/bin/python scripts/run_pipeline.py --variant tracked    [--frames 0 297]
산출 (docs/architecture.md 4절)
    raw       → outputs/phase5/objects_raw.json, runtime.json, frame{0,50,150}_fusion.png / _bev.png
    clustered → outputs/phase6/objects_clustered.json, runtime.json, frame{0,50,150}_fusion.png / _bev.png
    tracked   → outputs/phase7/tracks.json, runtime.json, bev_trajectories.png, frame{0,50,150}_tracks.png
                (clustered 융합을 다시 계산한 뒤 KalmanTracker 를 프레임 순서대로 적용)
runtime.json 형식
    {"meta": {hardware, sequence, n_frames, method}, "stages": [...],
     "frames": {"0": {"load": ms, "projection": ms, "filters": ms, "cluster": ms, "tracking": ms, "total": ms}, ...}}
    각 값은 프레임 1장의 단계별 경과 ms (time.perf_counter, 단일 프로세스, 시각화 저장 시간은 제외).
2D 검출은 outputs/phase4/detections.json 을 읽는다 (Phase 4 에서 생성, YOLO 를 다시 돌리지 않는다).
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import open3d as o3d
from tqdm import tqdm

from perceptrack3d.config import DEFAULT_CONFIG, load_config
from perceptrack3d.data.kitti_loader import KittiDataset
from perceptrack3d.detection.detector import load_detections_json
from perceptrack3d.evaluation.tracklets import gt_boxes_by_frame, gt_in_front_fov
from perceptrack3d.fusion.frustum_fusion import fuse_frame_clustered, fuse_frame_raw, save_objects_json
from perceptrack3d.geometry.calibration import KittiCalibration
from perceptrack3d.geometry.projection import project_velo_to_image
from perceptrack3d.tracking.kalman_tracker import KalmanTracker
from perceptrack3d.visualization.fusion_viz import draw_fusion, plot_bev_objects
from perceptrack3d.visualization.track_frames import plot_frame_tracks
from perceptrack3d.visualization.tracks import plot_bev_tracks

REP_FRAMES = [0, 50, 150]                     # 대표 프레임 (시각화 저장)
PHASE_DIR = {"raw": "phase5", "clustered": "phase6", "tracked": "phase7"}
STAGES = ["load", "projection", "filters", "cluster", "fusion", "tracking", "total"]


def cpu_model_name() -> str:
    try:
        out = subprocess.run(["lscpu"], capture_output=True, text=True, check=True).stdout
        for line in out.splitlines():
            if line.startswith("Model name"):
                return line.split(":", 1)[1].strip()
    except Exception:
        pass
    return platform.processor() or "unknown"


def runtime_meta(cfg: dict, variant: str, frame_ids: list[int]) -> dict:
    ds = cfg["dataset"]
    return {
        "variant": variant,
        "cpu": cpu_model_name(),
        "logical_cores": os.cpu_count(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "open3d": o3d.__version__,
        "sequence": f"{ds['date']}_drive_{ds['drive']}_sync",
        "n_frames": len(frame_ids),
        "frame_range": [frame_ids[0], frame_ids[-1]] if frame_ids else None,
        "unit": "ms per frame",
        "method": "time.perf_counter() per stage, single process, warm cache, visualization excluded",
        "load_avg_1_5_15": list(os.getloadavg()),   # 측정 시작 시 부하 (팀 G: 공용 PC 라 간섭 여부를 기록)
    }


def summarize(rows: dict[int, dict]) -> dict[str, dict]:
    """단계별 mean / median / p95 (ms)."""
    out = {}
    for st in STAGES:
        vals = np.array([r[st] for r in rows.values() if st in r])
        if len(vals):
            out[st] = {"mean": float(vals.mean()), "median": float(np.median(vals)), "p95": float(np.percentile(vals, 95))}
    return out


def gt_trajectories(gts: dict, frame_ids: list[int], x_max: float = 70.0) -> dict[int, np.ndarray]:
    """카메라 FOV 안·x_max 이내의 GT 를 tracklet_id 별 (K, 3) 궤적으로 묶는다 (BEV 궤적 그림용)."""
    traj: dict[int, list] = collections.defaultdict(list)
    for fid in frame_ids:
        for b in gt_in_front_fov(gts.get(fid, [])):
            if b.center[0] <= x_max:
                traj[b.tracklet_id].append(b.center)
    return {k: np.asarray(v) for k, v in traj.items() if len(v) >= 2}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--variant", choices=list(PHASE_DIR), required=True)
    ap.add_argument("--frames", nargs=2, type=int, metavar=("START", "END"), help="[START, END) 구간만 실행")
    ap.add_argument("--config", default=str(DEFAULT_CONFIG))
    ap.add_argument("--no-viz", action="store_true", help="대표 프레임 이미지를 저장하지 않는다")
    args = ap.parse_args()

    cfg = load_config(args.config)
    ds = KittiDataset(cfg)
    calib = KittiCalibration.from_dir(cfg["dataset"]["calib_dir"])
    frame_ids = ds.frame_ids()
    if args.frames:
        frame_ids = [f for f in frame_ids if args.frames[0] <= f < args.frames[1]]
    out_root = Path(cfg["outputs"]["dir"])
    det_path = out_root / "phase4" / "detections.json"
    if not det_path.is_file():
        sys.exit(f"2D 검출 결과가 없습니다: {det_path}  (먼저 scripts/run_detection.py 를 실행하세요)")
    dets = load_detections_json(det_path)
    gts = gt_boxes_by_frame(ds.tracklets_path)
    out_dir = out_root / PHASE_DIR[args.variant]
    out_dir.mkdir(parents=True, exist_ok=True)

    fuse = fuse_frame_raw if args.variant == "raw" else fuse_frame_clustered
    method = "raw" if args.variant == "raw" else "clustered"
    tracker = KalmanTracker(cfg) if args.variant == "tracked" else None

    objects_by_frame: dict[int, list] = {}
    tracks_by_frame: dict[int, list] = {}
    history: dict[int, list] = collections.defaultdict(list)      # track_id → [(x, y, z)] 관측으로 갱신된 프레임만 (coasting 제외)
    runtime: dict[int, dict] = {}
    t_start = time.perf_counter()

    for fid in tqdm(frame_ids, desc=args.variant):
        t0 = time.perf_counter()
        fr = ds.load_frame(fid)
        t1 = time.perf_counter()
        pts, img = fr["points"], fr["image"]
        timings: dict = {}
        objs = fuse(pts, dets.get(fid, []), calib, img.shape[:2], cfg, timings=timings)
        t2 = time.perf_counter()
        row = {"load": (t1 - t0) * 1e3, **timings}
        if tracker is not None:
            tracks = tracker.step(objs, fid)
            t3 = time.perf_counter()
            row["tracking"] = (t3 - t2) * 1e3
            # Track 객체는 다음 프레임에 제자리 갱신되므로, 이 프레임의 상태를 지금 dict 로 고정해 둔다
            tracks_by_frame[fid] = [tr.as_dict(fid) for tr in tracks]
            for tr in tracks:
                if tr.misses == 0:                 # coasting 위치는 예측값이라 궤적 그림에서 뺀다 (긴 직선 꼬리 방지)
                    history[tr.track_id].append(tr.position.copy())
        row["total"] = (time.perf_counter() - t0) * 1e3
        runtime[fid] = row
        objects_by_frame[fid] = objs

        if fid in REP_FRAMES and not args.no_viz:
            gt_f = gt_in_front_fov(gts.get(fid, []))
            if tracker is None:
                uv, depth, mask = project_velo_to_image(pts, calib, img.shape[:2])
                import cv2
                cv2.imwrite(str(out_dir / f"frame{fid}_fusion.png"), draw_fusion(img, uv, depth, objs, mask=mask))
                plot_bev_objects(pts, objs, out_dir / f"frame{fid}_bev.png", gt=gt_f,
                                 title=f"frame {fid} [{method}]: pred (red) vs GT (green)")
            else:
                plot_frame_tracks(img, tracks, pts, calib, out_dir / f"frame{fid}_tracks.png",
                                  gt=gt_f, history={k: np.asarray(v) for k, v in history.items()}, frame_id=fid)

    wall_s = time.perf_counter() - t_start

    # ---- 저장
    if args.variant == "raw":
        save_objects_json(objects_by_frame, out_dir / "objects_raw.json")
    elif args.variant == "clustered":
        save_objects_json(objects_by_frame, out_dir / "objects_clustered.json")
    else:
        payload = {str(f): tracks_by_frame[f] for f in frame_ids}
        with open(out_dir / "tracks.json", "w", encoding="utf-8") as fp:
            json.dump(payload, fp, indent=1)
        if not args.no_viz:
            plot_bev_tracks({k: np.asarray(v) for k, v in history.items()}, out_dir / "bev_trajectories.png",
                            gt=gt_trajectories(gts, frame_ids),
                            title=f"BEV trajectories: {len(history)} confirmed tracks vs GT (dashed), frames {frame_ids[0]}-{frame_ids[-1]}")

    meta = runtime_meta(cfg, args.variant, frame_ids)
    meta["wall_time_s"] = wall_s
    with open(out_dir / "runtime.json", "w", encoding="utf-8") as fp:
        json.dump({"meta": meta, "stages": STAGES, "summary": summarize(runtime),
                   "frames": {str(f): runtime[f] for f in frame_ids}}, fp, indent=1)

    # ---- 콘솔 요약
    status = collections.Counter(o.status for objs in objects_by_frame.values() for o in objs)
    print(f"\n[{args.variant}] {len(frame_ids)} frames, wall {wall_s:.1f} s ({wall_s / max(len(frame_ids), 1) * 1e3:.0f} ms/frame incl. viz)")
    print("status:", dict(status))
    for st, s in summarize(runtime).items():
        print(f"  {st:<11} mean {s['mean']:7.1f}  median {s['median']:7.1f}  p95 {s['p95']:7.1f} ms")
    if tracker is not None:
        n_ids = len(history)
        per_frame = [len(v) for v in tracks_by_frame.values()]
        print(f"tracks: {n_ids} ids, per-frame confirmed mean {np.mean(per_frame):.2f} max {max(per_frame)}")
    print(f"saved to {out_dir}")


if __name__ == "__main__":
    main()
