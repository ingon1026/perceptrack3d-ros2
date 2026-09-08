"""Phase 8 정량 평가: 변형 A/B/C 의 3D 결과를 tracklet GT 와 프레임별로 매칭해 지표를 집계한다.

입력
    preds_by_frame : {frame_id: [Object3D, ...]}   (변형 A raw / B clustered — status == "ok" 만 사용)
                     {frame_id: [Track, ...]} 또는 {frame_id: [Track.as_dict() 결과 dict, ...]}  (변형 C tracked — confirmed 만 사용)
    gts_by_frame   : {frame_id: [GtBox3D, ...]}    (evaluation.tracklets.gt_boxes_by_frame)
    cfg            : load_config() 결과. evaluation.match_distance_m / classes / frames, fusion.roi.x_max 를 읽는다.
출력
    evaluate_variant(...) -> dict  (구조는 함수 독스트링 참고)

좌표계: 모든 중심은 Velodyne (x 전방, y 좌, z 상, m). 매칭·오차는 BEV(x, y) 기준, "깊이" 는 x.

매칭 규칙 (scripts/compare_fusion.py 와 동일 — 팀 E 의 comparison.md 수치와 맞추기 위해):
    1. GT 필터: 카메라 FOV 안 (gt_in_front_fov), x <= fusion.roi.x_max, object_type ∈ evaluation.classes
    2. 예측 필터: A/B 는 status == "ok" 이고 center 가 있는 것, C 는 confirmed 트랙 (coasting 포함, include_coasting=False 면 제외)
    3. 프레임마다 BEV 중심 거리 헝가리안 1:1 매칭, 임계값 max_dist (m)
    4. TP = 매칭 쌍, FP = 미매칭 예측, FN = 미매칭 GT
"""
from __future__ import annotations

import collections
import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

from perceptrack3d.evaluation.metrics import (
    bev_iou,
    count_id_switches,
    track_fragmentation,
    match_by_center_distance,
)
from perceptrack3d.evaluation.tracklets import gt_in_front_fov
from perceptrack3d.types import GtBox3D, Object3D, Track

RANGE_BINS: list[tuple[float, float]] = [(0.0, 20.0), (20.0, 40.0), (40.0, 70.0)]   # GT 의 BEV 거리 구간 (m)
IOU_THRESHOLD = 0.5                   # KITTI object eval 의 Pedestrian/Cyclist 기준 (공식 Car 기준은 0.7). 우리 AABB 크기는 표면만 반영해 0.7 은 거의 0 이라 0.5 로 보고
MT_RATIO, ML_RATIO = 0.8, 0.2         # CLEAR MOT: mostly tracked ≥ 80 %, mostly lost ≤ 20 %


@dataclass
class PredRecord:
    """평가 입력을 통일한 예측 1개. Object3D / Track / dict 를 모두 이 형태로 바꾼다."""

    center: np.ndarray            # (3,) Velodyne m
    size: np.ndarray | None       # (3,) [l, w, h] 또는 None (A 는 크기 없음)
    yaw: float | None
    class_name: str
    track_id: int | None          # C 만. A/B 는 None
    coasting: bool = False        # C 에서 misses > 0 (이 프레임에 관측 없이 예측만 한 상태)


def _range_bin_label(lo: float, hi: float) -> str:
    return f"{lo:.0f}-{hi:.0f}"


def range_label(r: float) -> str | None:
    """GT/예측의 BEV 거리 r (m) 가 속한 구간 이름. 구간 밖(70 m 초과) 이면 None."""
    for lo, hi in RANGE_BINS:
        if lo <= r < hi or (hi == RANGE_BINS[-1][1] and r == hi):
            return _range_bin_label(lo, hi)
    return None


def to_pred_records(items: list, include_coasting: bool = True) -> list[PredRecord]:
    """한 프레임의 예측 리스트를 PredRecord 로 바꾼다. 평가 대상이 아닌 항목은 뺀다.

    - Object3D: status == "ok" 이고 center 가 있어야 함
    - Track: confirmed 만. center = state[:3]
    - dict (tracks.json 한 줄): "confirmed" 가 True 이고 center 가 있어야 함
    """
    out: list[PredRecord] = []
    for it in items:
        if isinstance(it, Object3D):
            if it.status != "ok" or it.center is None:
                continue
            out.append(PredRecord(np.asarray(it.center, float), None if it.size is None else np.asarray(it.size, float),
                                  it.yaw, it.class_name, None))
        elif isinstance(it, Track):
            if not it.confirmed:
                continue
            if not include_coasting and it.misses > 0:
                continue
            out.append(PredRecord(np.asarray(it.position, float), None if it.size is None else np.asarray(it.size, float),
                                  None, it.class_name, int(it.track_id), it.misses > 0))
        elif isinstance(it, dict):
            if not it.get("confirmed", False) or it.get("center") is None:
                continue
            coasting = int(it.get("misses", 0)) > 0
            if not include_coasting and coasting:
                continue
            size = it.get("size")
            out.append(PredRecord(np.asarray(it["center"], float), None if size is None else np.asarray(size, float),
                                  None, str(it.get("class_name", "")), int(it["track_id"]), coasting))
        else:
            raise TypeError(f"지원하지 않는 예측 타입: {type(it).__name__}")
    for p in out:
        assert p.center.shape == (3,), f"center 는 (3,) 이어야 합니다: {p.center.shape}"
    return out


def select_gt(gt_boxes: list[GtBox3D], x_max: float, classes: list[str], fov_deg: float = 81.4) -> list[GtBox3D]:
    """평가에 쓸 GT: 카메라 FOV 안, x <= x_max, 클래스 목록 안. (compare_fusion.gt_for_eval 과 같은 규칙)"""
    return [b for b in gt_in_front_fov(gt_boxes, fov_deg) if b.center[0] <= x_max and b.object_type in classes]


def eval_settings(cfg: dict, max_dist: float | None = None, frames: list[int] | None = None) -> dict:
    """cfg 에서 평가 설정을 뽑아 결과에 함께 저장할 dict 로 만든다 (재현성)."""
    ev = cfg["evaluation"]
    if frames is None and ev.get("frames"):
        start, end = ev["frames"]
        frames = list(range(int(start), int(end)))
    return {
        "match_distance_m": float(ev["match_distance_m"] if max_dist is None else max_dist),
        "gt_fov_deg": 81.4,
        "gt_x_max_m": float(cfg["fusion"]["roi"]["x_max"]),
        "classes": list(ev["classes"]),
        "frames": None if frames is None else [int(frames[0]), int(frames[-1]) + 1],
        "range_bins_m": [list(b) for b in RANGE_BINS],
        "iou_threshold": IOU_THRESHOLD,
        "mt_ratio": MT_RATIO,
        "ml_ratio": ML_RATIO,
    }


def _nan_stats(values: np.ndarray) -> tuple[float, float, float]:
    """(mean, median, p95). 비어 있으면 nan."""
    v = np.asarray(values, dtype=np.float64)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return float("nan"), float("nan"), float("nan")
    return float(v.mean()), float(np.median(v)), float(np.percentile(v, 95))


def _pr(n_matched: int, n_pred: int, n_gt: int) -> tuple[float, float, float]:
    """precision, recall, f1. 분모 0 이면 nan (0.0 이 아니라 '평가 불가' 를 뜻하도록)."""
    p = n_matched / n_pred if n_pred else float("nan")
    r = n_matched / n_gt if n_gt else float("nan")
    f = 2 * p * r / (p + r) if (n_pred and n_gt and (p + r) > 0) else (0.0 if (n_pred and n_gt) else float("nan"))
    return p, r, f


def match_frames(preds_by_frame: dict, gts_by_frame: dict[int, list[GtBox3D]], settings: dict,
                 include_coasting: bool = True) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """프레임별 매칭을 수행해 세 개의 표를 돌려준다.

    Returns:
        pairs : 매칭 쌍 1행 = frame, track_id, gt_id, gt_class, pred_class, gt_range, dist_bev, err_3d, depth_err
                (pred_x − gt_x, 양수 = GT 보다 멀리), iou (size 없으면 nan), coasting
        preds : 예측 1행 = frame, track_id, pred_range, matched, coasting
        gts   : GT 1행 = frame, gt_id, gt_class, gt_range, matched, track_id (매칭된 예측의 track_id, 없으면 -1)
    """
    x_max, classes, fov = settings["gt_x_max_m"], settings["classes"], settings["gt_fov_deg"]
    max_dist = settings["match_distance_m"]
    frames = settings["frames"]
    frame_ids = sorted(set(preds_by_frame) | set(gts_by_frame))
    if frames is not None:
        frame_ids = [f for f in frame_ids if frames[0] <= f < frames[1]]

    pair_rows, pred_rows, gt_rows = [], [], []
    for fid in frame_ids:
        preds = to_pred_records(preds_by_frame.get(fid, []), include_coasting)
        gts = select_gt(gts_by_frame.get(fid, []), x_max, classes, fov)
        P = np.array([p.center for p in preds]).reshape(-1, 3)
        G = np.array([g.center for g in gts]).reshape(-1, 3)
        matches = match_by_center_distance(P, G, max_dist) if (len(P) and len(G)) else []
        matched_p = {pi: (gi, d) for pi, gi, d in matches}
        matched_g = {gi: pi for pi, gi, _ in matches}

        for pi, p in enumerate(preds):
            pred_rows.append({"frame": fid, "track_id": -1 if p.track_id is None else p.track_id,
                              "pred_range": float(np.hypot(p.center[0], p.center[1])),
                              "matched": pi in matched_p, "coasting": p.coasting})
        for gi, g in enumerate(gts):
            pi = matched_g.get(gi)
            gt_rows.append({"frame": fid, "gt_id": g.tracklet_id, "gt_class": g.object_type,
                            "gt_range": float(np.hypot(g.center[0], g.center[1])), "matched": pi is not None,
                            "track_id": -1 if (pi is None or preds[pi].track_id is None) else preds[pi].track_id})
        for pi, gi, d in matches:
            p, g = preds[pi], gts[gi]
            diff = p.center - g.center
            iou = float("nan")
            if p.size is not None:
                iou = bev_iou((p.center, p.size, 0.0 if p.yaw is None else p.yaw), g)
            pair_rows.append({"frame": fid, "track_id": -1 if p.track_id is None else p.track_id,
                              "gt_id": g.tracklet_id, "gt_class": g.object_type, "pred_class": p.class_name,
                              "gt_range": float(np.hypot(g.center[0], g.center[1])),
                              "dist_bev": float(d), "err_3d": float(np.linalg.norm(diff)),
                              "depth_err": float(diff[0]), "iou": iou, "coasting": p.coasting})

    pairs = pd.DataFrame(pair_rows, columns=["frame", "track_id", "gt_id", "gt_class", "pred_class", "gt_range",
                                             "dist_bev", "err_3d", "depth_err", "iou", "coasting"])
    preds = pd.DataFrame(pred_rows, columns=["frame", "track_id", "pred_range", "matched", "coasting"])
    gts = pd.DataFrame(gt_rows, columns=["frame", "gt_id", "gt_class", "gt_range", "matched", "track_id"])
    return pairs, preds, gts


def summarize_pairs(pairs: pd.DataFrame, n_pred: int, n_gt: int, idsw: int = 0) -> dict:
    """매칭 쌍 표 하나를 지표 dict 로 요약한다 (전체·구간·클래스가 같은 함수를 쓴다).

    MOTA = 1 − (FN + FP + IDSW) / n_gt  (CLEAR MOT; A/B 는 IDSW = 0 이라 '검출 MOTA').
    iou_* 는 size 가 있는 매칭 쌍만 (A 는 전부 nan → iou_n = 0).
    """
    n_matched = len(pairs)
    p, r, f = _pr(n_matched, n_pred, n_gt)
    bev_mean, bev_median, bev_p95 = _nan_stats(pairs["dist_bev"].to_numpy() if n_matched else np.array([]))
    d3_mean, d3_median, d3_p95 = _nan_stats(pairs["err_3d"].to_numpy() if n_matched else np.array([]))
    depth = pairs["depth_err"].to_numpy() if n_matched else np.array([])
    dabs_mean, dabs_median, dabs_p95 = _nan_stats(np.abs(depth))
    iou = pairs["iou"].to_numpy() if n_matched else np.array([])
    iou = iou[np.isfinite(iou)]
    return {
        "n_gt": int(n_gt), "n_pred": int(n_pred), "n_matched": int(n_matched),
        "n_fp": int(n_pred - n_matched), "n_fn": int(n_gt - n_matched),
        "precision": p, "recall": r, "f1": f,
        "mota": 1.0 - (n_gt - n_matched + n_pred - n_matched + idsw) / n_gt if n_gt else float("nan"),
        "bev_mean": bev_mean, "bev_median": bev_median, "bev_p95": bev_p95,
        "d3_mean": d3_mean, "d3_median": d3_median, "d3_p95": d3_p95,
        "depth_abs_mean": dabs_mean, "depth_abs_median": dabs_median, "depth_abs_p95": dabs_p95,
        "depth_bias": float(depth.mean()) if depth.size else float("nan"),
        "iou_n": int(iou.size),
        "iou_mean": float(iou.mean()) if iou.size else float("nan"),
        "iou_ge05_frac": float((iou >= IOU_THRESHOLD).mean()) if iou.size else float("nan"),
    }


def tracking_metrics(pairs: pd.DataFrame, preds: pd.DataFrame, gts: pd.DataFrame) -> dict:
    """변형 C 전용 추적 지표.

    - idsw : count_id_switches — 같은 GT 가 직전에 매칭됐던 트랙과 다른 트랙에 붙으면 1
    - frag : track_fragmentation — GT 의 매칭 프레임 사이에 빈 구간이 생길 때마다 1
    - coverage (GT tracklet 별): n_present = 평가 GT 로 존재한 프레임 수,
        cov_any = 어떤 트랙이든 매칭된 프레임 비율, cov_single = 가장 많이 매칭된 **단일** track_id 의 프레임 비율
    - MT / PT / ML : cov_single ≥ 0.8 / 그 사이 / ≤ 0.2 인 GT 수 (단일 ID 기준 = 지시서의 근사 정의)
    """
    frames = sorted(gts["frame"].unique().tolist())          # 매칭 쌍은 GT 가 있는 프레임에만 존재한다
    assignments = [[(int(t), int(g)) for t, g in pairs.loc[pairs["frame"] == f, ["track_id", "gt_id"]].itertuples(index=False)]
                   for f in frames]
    idsw = count_id_switches(assignments)
    frag = track_fragmentation(assignments)

    coverage: dict[int, dict] = {}
    for gid, grp in gts.groupby("gt_id"):
        n_present = len(grp)
        matched = grp[grp["matched"]]
        counts = collections.Counter(matched["track_id"].tolist())
        best_id, best_n = (counts.most_common(1)[0] if counts else (-1, 0))
        coverage[int(gid)] = {
            "gt_class": str(grp["gt_class"].iloc[0]), "n_present": int(n_present),
            "first_frame": int(grp["frame"].min()), "last_frame": int(grp["frame"].max()),
            "n_matched": int(len(matched)), "n_ids": int(len(counts)),
            "best_track_id": int(best_id), "cov_any": len(matched) / n_present, "cov_single": best_n / n_present,
        }
    cov_single = np.array([c["cov_single"] for c in coverage.values()])
    cov_any = np.array([c["cov_any"] for c in coverage.values()])
    n_ids = int(preds.loc[preds["track_id"] >= 0, "track_id"].nunique())
    return {
        "idsw": int(idsw), "frag": int(frag),
        "n_track_ids": n_ids, "n_gt_tracklets": int(len(coverage)),
        "mt": int((cov_single >= MT_RATIO).sum()), "ml": int((cov_single <= ML_RATIO).sum()),
        "pt": int(((cov_single < MT_RATIO) & (cov_single > ML_RATIO)).sum()),
        "mean_cov_single": float(cov_single.mean()) if cov_single.size else float("nan"),
        "mean_cov_any": float(cov_any.mean()) if cov_any.size else float("nan"),
        "n_coasting_pred": int(preds["coasting"].sum()),
        "n_coasting_matched": int(pairs["coasting"].sum()) if len(pairs) else 0,
        "coverage": coverage,
    }


def evaluate_variant(preds_by_frame: dict, gts_by_frame: dict[int, list[GtBox3D]], cfg: dict, variant_name: str,
                     max_dist: float | None = None, frames: list[int] | None = None,
                     include_coasting: bool = True) -> dict:
    """변형 하나를 GT 와 비교해 지표를 모두 계산한다.

    Args:
        preds_by_frame : 모듈 독스트링의 세 형식 중 하나
        gts_by_frame   : gt_boxes_by_frame() 결과 (필터 전, 모든 GT)
        cfg            : load_config()
        variant_name   : 결과에 붙일 이름 ("A_raw" 등)
        max_dist       : 매칭 임계값 (m). None 이면 cfg evaluation.match_distance_m
        frames         : 평가할 프레임 목록. None 이면 cfg evaluation.frames (null = 전체)
        include_coasting : C 에서 관측 없이 예측만 한 트랙도 예측으로 셀지 (기본 True = tracks.json 그대로)
    Returns: {
        "variant", "settings" (평가 설정), "has_tracks" (bool),
        "overall"  : summarize_pairs 의 dict (n_gt, n_pred, n_matched, precision, recall, f1, mota, bev_*, d3_*, depth_*, iou_*),
        "by_range" : {"0-20": {...같은 키..., "n_pred" 는 예측 자신의 거리 기준}, "20-40", "40-70"},
        "by_class" : {클래스: {...}} — n_gt == 0 인 클래스는 지표가 nan ("n/a"),
        "tracking" : tracking_metrics 결과 (트랙 ID 가 없으면 None),
        "per_frame": DataFrame(frame, n_gt, n_pred, n_matched),
        "pairs", "preds", "gts": match_frames 의 표 (플롯·디버깅용)
    }
    """
    settings = eval_settings(cfg, max_dist, frames)
    pairs, preds, gts = match_frames(preds_by_frame, gts_by_frame, settings, include_coasting)
    has_tracks = bool(len(preds)) and bool((preds["track_id"] >= 0).any())

    tracking = tracking_metrics(pairs, preds, gts) if has_tracks else None
    idsw = tracking["idsw"] if tracking else 0
    overall = summarize_pairs(pairs, len(preds), len(gts), idsw)

    by_range = {}
    for lo, hi in RANGE_BINS:
        label = _range_bin_label(lo, hi)
        in_bin_gt = gts["gt_range"].apply(range_label) == label
        in_bin_pred = preds["pred_range"].apply(range_label) == label
        in_bin_pair = pairs["gt_range"].apply(range_label) == label
        d = summarize_pairs(pairs[in_bin_pair], int(in_bin_pred.sum()), int(in_bin_gt.sum()))
        # 매칭 쌍은 GT 거리로, 예측은 자신의 거리로 구간을 나누므로 둘이 다른 구간에 들 수 있다.
        # → FP/precision 은 "그 구간에 있는 예측 중 미매칭 비율" 로 다시 세고, MOTA 는 구간별로 정의하지 않는다.
        d["n_fp"] = int((in_bin_pred & ~preds["matched"]).sum())
        d["precision"] = 1.0 - d["n_fp"] / d["n_pred"] if d["n_pred"] else float("nan")
        d["mota"] = float("nan")
        by_range[label] = d

    by_class = {}
    for cls in settings["classes"]:
        g = gts[gts["gt_class"] == cls]
        pr = pairs[pairs["gt_class"] == cls]
        d = summarize_pairs(pr, len(pr), len(g))     # 예측 클래스(COCO) 와 GT 클래스(KITTI) 가 달라 클래스별 FP 는 정의하지 않는다 → recall·오차만 의미 있음
        d["n_pred"], d["n_fp"], d["precision"], d["f1"], d["mota"] = 0, 0, float("nan"), float("nan"), float("nan")
        by_class[cls] = d

    per_frame = (pd.DataFrame({"frame": sorted(set(gts["frame"]) | set(preds["frame"]))})
                 .merge(gts.groupby("frame").size().rename("n_gt"), on="frame", how="left")
                 .merge(preds.groupby("frame").size().rename("n_pred"), on="frame", how="left")
                 .merge(pairs.groupby("frame").size().rename("n_matched"), on="frame", how="left")
                 .fillna(0).astype(int))

    return {"variant": variant_name, "settings": settings, "has_tracks": has_tracks, "include_coasting": include_coasting,
            "overall": overall, "by_range": by_range, "by_class": by_class, "tracking": tracking,
            "per_frame": per_frame, "pairs": pairs, "preds": preds, "gts": gts}


def results_to_long(results: list[dict]) -> pd.DataFrame:
    """evaluate_variant 결과 여러 개를 long 형식 표로 편다: variant, max_dist_m, section, group, metric, value."""
    rows = []
    for r in results:
        thr = r["settings"]["match_distance_m"]
        for k, v in r["overall"].items():
            rows.append((r["variant"], thr, "overall", "all", k, v))
        for grp, d in r["by_range"].items():
            for k, v in d.items():
                rows.append((r["variant"], thr, "range", grp, k, v))
        for grp, d in r["by_class"].items():
            for k, v in d.items():
                rows.append((r["variant"], thr, "class", grp, k, v))
        if r["tracking"]:
            for k, v in r["tracking"].items():
                if k != "coverage":
                    rows.append((r["variant"], thr, "tracking", "all", k, v))
    return pd.DataFrame(rows, columns=["variant", "max_dist_m", "section", "group", "metric", "value"])


def fmt(v, nd: int = 2) -> str:
    """표용 숫자 포맷. nan → "n/a"."""
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "n/a"
    if isinstance(v, (int, np.integer)):
        return str(int(v))
    return f"{v:.{nd}f}"
