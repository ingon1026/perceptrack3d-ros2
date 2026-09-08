"""Phase 5(raw) 대 Phase 6(clustered) 융합 결과를 GT 와 수치로 비교해 outputs/phase6/comparison.md 를 쓴다.

실행: .venv/bin/python scripts/compare_fusion.py   (먼저 run_pipeline.py --variant raw / clustered 를 실행해 둘 것)
매칭: 프레임마다 status == "ok" 인 예측 중심 ↔ 카메라 FOV 안·ROI x_max 이내 GT 중심을 BEV 거리로 1:1 헝가리안 매칭
      (evaluation.metrics.match_by_center_distance). 임계값은 evaluation.match_distance_m (2 m) 와 넉넉한 4 m 두 가지로 보고한다.
지표: 매칭 수, 중심 오차(3D/BEV mean·median), 깊이(x) 오차 mean|·| 와 bias(예측 − GT, 음수 = 가깝게 추정), 매칭률.
주의: GT 에는 tracklet 이 없는 주차 차량이 있고(팀 B 관찰), YOLO 오검출도 있으므로 "미매칭 예측" 이 전부 오류는 아니다.
"""
from __future__ import annotations

import collections
from pathlib import Path

import numpy as np

from perceptrack3d.config import load_config
from perceptrack3d.evaluation.metrics import center_errors, depth_errors, match_by_center_distance
from perceptrack3d.evaluation.tracklets import gt_boxes_by_frame, gt_in_front_fov
from perceptrack3d.fusion.frustum_fusion import load_objects_json
from perceptrack3d.types import GtBox3D, Object3D

REP_FRAMES = [0, 50, 150]


def gt_for_eval(gts: dict[int, list[GtBox3D]], fid: int, x_max: float, classes: list[str]) -> list[GtBox3D]:
    return [b for b in gt_in_front_fov(gts.get(fid, [])) if b.center[0] <= x_max and b.object_type in classes]


def evaluate_objects(objects_by_frame: dict[int, list[Object3D]], gts: dict[int, list[GtBox3D]], cfg: dict,
                     max_dist: float, frames: list[int] | None = None) -> dict:
    """프레임별 매칭을 모아 전체 통계를 낸다. Returns: n_gt, n_pred_ok, n_matched, center/depth 통계, status 분포, match_rate."""
    x_max = float(cfg["fusion"]["roi"]["x_max"])
    classes = list(cfg["evaluation"]["classes"])
    frames = sorted(objects_by_frame) if frames is None else frames
    P, G, matches, offset = [], [], [], 0
    status = collections.Counter()
    n_gt = n_pred = 0
    for fid in frames:
        objs = objects_by_frame.get(fid, [])
        for o in objs:
            status[o.status] += 1
        preds = [o for o in objs if o.status == "ok"]
        gt = gt_for_eval(gts, fid, x_max, classes)
        n_gt += len(gt)
        n_pred += len(preds)
        if preds and gt:
            m = match_by_center_distance(preds, gt, max_dist)
            pc = np.array([o.center for o in preds]); gc = np.array([b.center for b in gt])
            for pi, gi, d in m:
                P.append(pc[pi]); G.append(gc[gi]); matches.append((len(P) - 1, len(G) - 1, d))
    P = np.array(P).reshape(-1, 3); G = np.array(G).reshape(-1, 3)
    ce = center_errors(matches, P, G) if matches else center_errors([], np.zeros((0, 3)), np.zeros((0, 3)))
    de = depth_errors(matches, P, G) if matches else depth_errors([], np.zeros((0, 3)), np.zeros((0, 3)))
    return {
        "n_gt": n_gt, "n_pred_ok": n_pred, "n_matched": len(matches),
        "recall": len(matches) / n_gt if n_gt else float("nan"),
        "precision": len(matches) / n_pred if n_pred else float("nan"),
        "center": ce, "depth": de, "status": dict(status),
    }


def fmt_row(name: str, r: dict) -> str:
    c, d = r["center"], r["depth"]
    return (f"| {name} | {r['n_gt']} | {r['n_pred_ok']} | {r['n_matched']} | {r['recall']:.2f} | {r['precision']:.2f} "
            f"| {c['mean_bev']:.2f} | {c['median_bev']:.2f} | {c['mean_3d']:.2f} | {d['mean_abs']:.2f} | {d['bias']:+.2f} |")


HEADER = ("| 변형 | GT | 예측(ok) | 매칭 | 재현율 | 정밀도 | BEV 오차 mean | BEV median | 3D mean | 깊이 |오차| mean | 깊이 bias |\n"
          "|---|---|---|---|---|---|---|---|---|---|---|")


def run_ablation(cfg: dict, overrides: dict) -> dict[int, list[Object3D]]:
    """cfg["fusion"] 의 일부 키를 바꿔 clustered 융합을 전 프레임에 다시 돌린다 (구성 요소별 기여 확인용, ~10 s)."""
    import copy

    from perceptrack3d.data.kitti_loader import KittiDataset
    from perceptrack3d.detection.detector import load_detections_json
    from perceptrack3d.fusion.frustum_fusion import fuse_frame_clustered
    from perceptrack3d.geometry.calibration import KittiCalibration

    c = copy.deepcopy(cfg)
    for key, val in overrides.items():           # "surface_offset.enable" 처럼 점으로 중첩 키 지정
        d = c["fusion"]
        parts = key.split(".")
        for k in parts[:-1]:
            d = d[k]
        d[parts[-1]] = val
    ds = KittiDataset(c)
    calib = KittiCalibration.from_dir(c["dataset"]["calib_dir"])
    dets = load_detections_json(Path(c["outputs"]["dir"]) / "phase4" / "detections.json")
    return {f: fuse_frame_clustered(ds.load_frame(f)["points"], dets.get(f, []), calib, calib.image_shape, c)
            for f in ds.frame_ids()}


ABLATIONS = [
    ("B0 clustered, 오프셋 없음 (클러스터 중앙값 그대로)", {"surface_offset.enable": False}),
    ("B1 clustered + 오프셋, cluster_select=nearest", {"cluster_select": "nearest"}),
    ("B2 clustered + 오프셋, 이전 클러스터 파라미터 (eps 0.6, min_points 8)", {"cluster.eps": 0.6, "cluster.min_points": 8}),
]


def main() -> None:
    cfg = load_config()
    out_root = Path(cfg["outputs"]["dir"])
    gts = gt_boxes_by_frame(cfg["dataset"]["tracklets"])
    variants = {
        "A raw (Phase 5)": load_objects_json(out_root / "phase5" / "objects_raw.json"),
        "B clustered (Phase 6)": load_objects_json(out_root / "phase6" / "objects_clustered.json"),
    }
    d_cfg = float(cfg["evaluation"]["match_distance_m"])
    lines = ["# Phase 5 vs Phase 6 융합 비교 (raw vs clustered)", "",
             f"시퀀스 {cfg['dataset']['date']}_drive_{cfg['dataset']['drive']}_sync, 297 프레임. "
             f"GT = tracklet 중 카메라 FOV 안(`gt_in_front_fov`), x ≤ {cfg['fusion']['roi']['x_max']} m, 클래스 {cfg['evaluation']['classes']}. "
             "예측 = `status == \"ok\"` 만. 매칭 = BEV 중심 거리 헝가리안 1:1. 오차 단위 m. 깊이 bias = mean(pred_x − gt_x) (음수 = GT 보다 가깝게 추정).",
             "", "생성: `scripts/compare_fusion.py` (raw/clustered JSON 은 `scripts/run_pipeline.py`).", ""]

    for thr in (d_cfg, 4.0):
        lines += [f"## 전 프레임 집계 (매칭 임계값 {thr:.1f} m)", "", HEADER]
        for name, objs in variants.items():
            lines.append(fmt_row(name, evaluate_objects(objs, gts, cfg, thr)))
        lines.append("")

    lines += [f"## 구성 요소별 기여 (ablation, 전 프레임, 매칭 임계값 {d_cfg:.1f} m)", "",
              "B 의 각 요소를 하나씩 끈 결과. `run_ablation` 으로 즉석 계산 (configs/kitti.yaml 의 fusion 키만 바꿈).", "", HEADER]
    for name, over in ABLATIONS:
        lines.append(fmt_row(name, evaluate_objects(run_ablation(cfg, over), gts, cfg, d_cfg)))
    lines.append(fmt_row("B  clustered (현재 설정: 오프셋 on, largest, eps 0.8, min_points 5)", evaluate_objects(variants["B clustered (Phase 6)"], gts, cfg, d_cfg)))
    lines.append("")

    lines += [f"## 대표 프레임 (매칭 임계값 {d_cfg:.1f} m)", ""]
    for fid in REP_FRAMES:
        lines += [f"### frame {fid}", "", HEADER]
        for name, objs in variants.items():
            r = evaluate_objects(objs, gts, cfg, d_cfg, frames=[fid])
            lines.append(fmt_row(name, r))
        lines.append("")
        # 객체별 상세
        lines += ["| 변형 | 검출 (class conf) | status | reason | n_points | center (x, y, z) | size (l, w, h) |", "|---|---|---|---|---|---|---|"]
        for name, objs in variants.items():
            for o in objs.get(fid, []):
                c = "-" if o.center is None else "(" + ", ".join(f"{v:.2f}" for v in o.center) + ")"
                s = "-" if o.size is None else "(" + ", ".join(f"{v:.2f}" for v in o.size) + ")"
                lines.append(f"| {name.split()[0]} | {o.class_name} {o.confidence:.2f} | {o.status} | {o.reason} | {o.n_points} | {c} | {s} |")
        gt = gt_for_eval(gts, fid, float(cfg["fusion"]["roi"]["x_max"]), list(cfg["evaluation"]["classes"]))
        lines.append("")
        lines.append("GT (FOV 안): " + "; ".join(f"gt{b.tracklet_id} {b.object_type} ({b.center[0]:.2f}, {b.center[1]:.2f}, {b.center[2]:.2f}) l={b.size[0]:.2f}" for b in gt))
        lines.append("")

    lines += ["## status 분포 (전 프레임, 검출 1325건)", "", "| 변형 | ok | empty | sparse | invalid |", "|---|---|---|---|---|"]
    for name, objs in variants.items():
        st = evaluate_objects(objs, gts, cfg, d_cfg)["status"]
        lines.append(f"| {name} | {st.get('ok', 0)} | {st.get('empty', 0)} | {st.get('sparse', 0)} | {st.get('invalid', 0)} |")
    lines.append("")

    path = out_root / "phase6" / "comparison.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    print(f"\nsaved {path}")


if __name__ == "__main__":
    main()
