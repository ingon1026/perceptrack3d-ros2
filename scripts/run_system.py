"""PercepTrack3D 통합 실행 — 한 파일로 읽는 전체 시스템.

KITTI 프레임 하나가 아래 여섯 단계를 순서대로 지난다. 각 단계는 src/perceptrack3d/ 의 모듈 하나를 부른다.

    [1] 로드      KittiDataset.load_frame      → image (H,W,3) uint8 BGR,  points (N,4) float32 [x,y,z,r] (Velodyne)
    [2] 2D 검출   YoloDetector.detect_timed    → list[Detection2D]        (픽셀 xyxy, class, conf)
    [3] 융합      fuse_frame_clustered          → list[Object3D]           (center (3,) Velodyne m, size, status)
                  (내부에서 [3a] 투영 → [3b] ROI·지면 제거 → [3c] 박스 안 점 → [3d] DBSCAN → [3e] 표면→중심 오프셋)
    [4] 추적      KalmanTracker.step            → list[Track]              (track_id, state [x,y,z,vx,vy,vz])
    [5] 시각화    이미지(2D 박스 + 트랙 ID) + BEV(점 + 트랙 + GT)          → PNG / mp4 / 화면
                  --view3d: LiDAR 3D 뷰(자차 뒤 위 카메라, 높이 색 점군 + 트랙·GT 3D 박스)  → lidar3d.mp4 / 화면
    [6] 평가      evaluate_variant (tracklet GT 와 BEV 2 m 매칭)            → 재현율·정밀도·중심 오차·IDSW, 런타임

실행 예
    .venv/bin/python scripts/run_system.py                      # 297 프레임, mp4 + 요약
    .venv/bin/python scripts/run_system.py --frames 0 60 --show # 60 프레임, 창으로 보면서
    .venv/bin/python scripts/run_system.py --fusion raw --no-video
    .venv/bin/python scripts/run_system.py --view3d --show        # LiDAR 3D 창까지 함께

출력  outputs/system/perceptrack3d.mp4, frames/frame{N}.png, summary.md, (--view3d) lidar3d.mp4, lidar3d/frame{N}.png
"""
from __future__ import annotations

import argparse
import collections
import time
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from perceptrack3d.config import load_config
from perceptrack3d.data.kitti_loader import KittiDataset
from perceptrack3d.detection.detector import YoloDetector
from perceptrack3d.evaluation.evaluate import evaluate_variant
from perceptrack3d.evaluation.tracklets import gt_boxes_by_frame, gt_in_front_fov
from perceptrack3d.fusion.frustum_fusion import fuse_frame_clustered, fuse_frame_raw
from perceptrack3d.geometry.calibration import KittiCalibration
from perceptrack3d.tracking.kalman_tracker import KalmanTracker
from perceptrack3d.visualization.boxes2d import draw_detections
from perceptrack3d.visualization.lidar3d import LidarView3D, track_color
from perceptrack3d.visualization.track_frames import plot_frame_tracks


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default="configs/kitti.yaml")
    p.add_argument("--frames", nargs=2, type=int, default=[0, 297], metavar=("START", "END"), help="[START, END) 프레임")
    p.add_argument("--fusion", choices=["clustered", "raw"], default="clustered", help="Phase 6 개선(기본) 또는 Phase 5 기준선")
    p.add_argument("--no-video", action="store_true", help="프레임 PNG·mp4 를 만들지 않는다 (평가·런타임만)")
    p.add_argument("--show", action="store_true", help="프레임마다 창에 띄운다 (q 로 중단)")
    p.add_argument("--no-eval", action="store_true")
    p.add_argument("--view3d", action="store_true", help="LiDAR 3D 뷰도 만든다 (lidar3d.mp4, --show 면 창)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    out_dir = Path(cfg["outputs"]["dir"]) / "system"
    (out_dir / "frames").mkdir(parents=True, exist_ok=True)
    (out_dir / "lidar3d").mkdir(exist_ok=True)

    # ---- 준비: 데이터·캘리브레이션·검출기·트래커·GT --------------------------------------------
    ds = KittiDataset(cfg)
    calib = KittiCalibration.from_dir(cfg["dataset"]["calib_dir"])     # P_velo_to_img = P_rect_02 @ R_rect_00 @ T_velo_to_cam
    detector = YoloDetector(cfg)                                        # 생성 시 yolov8n 로드 + warm-up
    tracker = KalmanTracker(cfg)                                        # 등속 칼만 + 마할라노비스 게이트 + 헝가리안
    fuse = fuse_frame_clustered if args.fusion == "clustered" else fuse_frame_raw
    gts = gt_boxes_by_frame(cfg["dataset"]["tracklets"])                # {frame_id: [GtBox3D]} — 시각화·평가용
    frame_ids = [f for f in ds.frame_ids() if args.frames[0] <= f < args.frames[1]]

    objects_by_frame: dict[int, list] = {}
    tracks_by_frame: dict[int, list] = {}
    history: dict[int, list] = collections.defaultdict(list)            # track_id → 관측된 위치들 (BEV 궤적)
    runtime: list[dict] = []
    writer: cv2.VideoWriter | None = None
    view3d = LidarView3D() if args.view3d else None                      # EGL 오프스크린 렌더러 (초기화 ~0.6 s)
    writer3d: cv2.VideoWriter | None = None

    # ---- 프레임 루프: 한 프레임이 시스템을 통과하는 경로 --------------------------------------------
    for fid in tqdm(frame_ids, desc=f"perceptrack3d ({args.fusion})"):
        t0 = time.perf_counter()
        fr = ds.load_frame(fid)                                         # [1] 로드
        img, pts = fr["image"], fr["points"]
        t1 = time.perf_counter()

        dets, det_ms = detector.detect_timed(img, fid)                  # [2] 2D 검출 (도로 객체 클래스만)

        timings: dict = {}
        objs = fuse(pts, dets, calib, img.shape[:2], cfg, timings=timings)   # [3] 융합 → 3D 중심 (Velodyne)
        t2 = time.perf_counter()

        tracks = tracker.step(objs, fid)                                # [4] 추적 → confirmed 트랙 (coasting 포함)
        t3 = time.perf_counter()

        objects_by_frame[fid] = objs
        tracks_by_frame[fid] = [tr.as_dict(fid) for tr in tracks]       # Track 은 다음 프레임에 제자리 갱신되므로 지금 고정
        for tr in tracks:
            if tr.misses == 0:                                          # 예측만 한 coasting 위치는 궤적에서 뺀다
                history[tr.track_id].append(tr.position.copy())
        runtime.append({"load": (t1 - t0) * 1e3, "detection": det_ms, **timings,
                        "tracking": (t3 - t2) * 1e3, "total": (t3 - t0) * 1e3})

        if args.no_video and not args.show:
            continue
        # [5] 시각화: 위 = 2D 박스 + 트랙 ID 를 투영한 이미지, 아래 = BEV (점 + 트랙 + 궤적 + GT 초록 박스)
        gt_f = gt_in_front_fov(gts.get(fid, []))
        png = plot_frame_tracks(draw_detections(img, dets), tracks, pts, calib, out_dir / "frames" / f"frame{fid:04d}.png",
                                gt=gt_f, frame_id=fid,
                                history={k: np.asarray(v) for k, v in history.items()})
        canvas = cv2.imread(str(png))
        if not args.no_video:
            if writer is None:
                h, w = canvas.shape[:2]
                writer = cv2.VideoWriter(str(out_dir / "perceptrack3d.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (w, h))
            writer.write(canvas)
        if view3d is not None:
            # [5b] LiDAR 3D 뷰: 트랙 박스(ID 색, 크기 없으면 승용차 기본값) + GT 박스(초록). 점 색 = 높이
            boxes = [(tr.position, tr.size if tr.size is not None else (4.0, 1.8, 1.6), 0.0, track_color(tr.track_id))
                     for tr in tracks]
            boxes += [(g.center, g.size, g.yaw, (0.2, 1.0, 0.3)) for g in gt_f]
            img3d = view3d.render(pts, boxes, label=f"frame {fid}   tracks {len(tracks)}   GT {len(gt_f)}")
            cv2.imwrite(str(out_dir / "lidar3d" / f"frame{fid:04d}.png"), img3d)
            if not args.no_video:
                if writer3d is None:
                    h, w = img3d.shape[:2]
                    writer3d = cv2.VideoWriter(str(out_dir / "lidar3d.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (w, h))
                writer3d.write(img3d)
            if args.show:
                cv2.imshow("PercepTrack3D LiDAR 3D", img3d)
        if args.show:
            cv2.imshow("PercepTrack3D  (q: quit)", canvas)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    for wr in (writer, writer3d):
        if wr is not None:
            wr.release()
    if args.show:
        cv2.destroyAllWindows()
    frame_ids = list(tracks_by_frame)                                   # q 로 중단했으면 처리한 프레임까지만 평가

    # ---- [6] 평가·런타임 요약 -------------------------------------------------------------------
    lines = [f"# PercepTrack3D 통합 실행 요약 (fusion={args.fusion}, frames {frame_ids[0]}..{frame_ids[-1]}, n={len(frame_ids)})", ""]
    med = {k: float(np.median([r[k] for r in runtime if k in r])) for k in runtime[0]}
    lines += ["## 런타임 (프레임당 중앙값 ms, 시각화 제외)", "",
              "| " + " | ".join(med) + " |", "|" + "---|" * len(med),
              "| " + " | ".join(f"{v:.1f}" for v in med.values()) + " |",
              f"", f"end-to-end {med['total']:.1f} ms → {1000 / med['total']:.0f} FPS", ""]
    if not args.no_eval:
        for name, preds in (("B_objects", objects_by_frame), ("C_tracks", tracks_by_frame)):
            r = evaluate_variant(preds, gts, cfg, name, frames=frame_ids)
            o, t = r["overall"], r["tracking"]
            lines += [f"## {name}  (GT {o['n_gt']}, 예측 {o['n_pred']}, 매칭 {o['n_matched']}, BEV 2 m)", "",
                      f"- 재현율 {o['recall']:.2f}, 정밀도 {o['precision']:.2f}, F1 {o['f1']:.2f}, MOTA {o['mota']:.2f}",
                      f"- BEV 중심 오차 mean {o['bev_mean']:.2f} / median {o['bev_median']:.2f} m, 깊이 bias {o['depth_bias']:+.2f} m"]
            if t:
                lines.append(f"- 트랙 {t['n_track_ids']} 개 vs GT tracklet {t['n_gt_tracklets']} 개, IDSW {t['idsw']}, 단절 {t['frag']}, "
                             f"MT/PT/ML {t['mt']}/{t['pt']}/{t['ml']}")
            lines.append("")
    summary = "\n".join(lines)
    (out_dir / "summary.md").write_text(summary, encoding="utf-8")
    print("\n" + summary)
    print(f"저장: {out_dir}")


if __name__ == "__main__":
    main()
