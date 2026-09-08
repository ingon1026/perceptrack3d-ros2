"""evaluation/evaluate.py 테스트: 합성 GT/예측 3 프레임으로 손계산과 비교한다.

합성 장면 (Velodyne, m). GT 는 카메라 FOV 안(x > 0, |y| < x·tan 40.7°), x ≤ 70, 클래스 Car/Van.
    프레임 0: GT g0 (10, 0)  g1 (30, 5)       예측 p (10.5, 0) [g0 에 0.5 m], p (30, 5.8) [g1 에 0.8 m]
    프레임 1: GT g0 (11, 0)  g1 (31, 5)       예측 p (11.3, 0) [g0], p (50, 0) [FP]            → g1 FN
    프레임 2: GT g0 (12, 0)  g1 (32, 5)       예측 p (12, 0.4) [g0], p (32, 5) [g1, 0 m]
    손계산: n_gt 6, n_pred 6, TP 5, FP 1, FN 1 → precision 5/6, recall 5/6.
            BEV 오차 [0.5, 0.8, 0.3, 0.4, 0.0] → mean 0.4, median 0.4.
            깊이 오차 pred_x − gt_x = [0.5, 0, 0.3, 0, 0] → bias 0.16.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from perceptrack3d.evaluation.evaluate import (
    evaluate_variant,
    fmt,
    range_label,
    results_to_long,
    select_gt,
    to_pred_records,
)
from perceptrack3d.types import GtBox3D, Object3D, Track

CFG = {
    "evaluation": {"match_distance_m": 2.0, "classes": ["Car", "Van", "Pedestrian"], "frames": None},
    "fusion": {"roi": {"x_max": 70.0}},
}


def _gt(fid, tid, x, y, cls="Car", l=4.0, w=1.8, h=1.5, yaw=0.0):
    return GtBox3D(frame_id=fid, tracklet_id=tid, object_type=cls, center=np.array([x, y, -0.8]),
                   size=np.array([l, w, h]), yaw=yaw)


def _obj(fid, x, y, size=None, status="ok", cls="car"):
    return Object3D(frame_id=fid, class_name=cls, confidence=0.9, xyxy=np.zeros(4),
                    center=None if status != "ok" else np.array([x, y, -0.8]),
                    size=None if size is None else np.array(size), yaw=0.0, status=status)


def _track_dict(fid, tid, x, y, misses=0, confirmed=True, size=None):
    return {"frame_id": fid, "track_id": tid, "class_name": "car", "center": [x, y, -0.8], "velocity": [0, 0, 0],
            "size": size, "hits": 5, "age": 5, "misses": misses, "confirmed": confirmed}


@pytest.fixture
def gts():
    return {0: [_gt(0, 0, 10, 0), _gt(0, 1, 30, 5, "Van")],
            1: [_gt(1, 0, 11, 0), _gt(1, 1, 31, 5, "Van")],
            2: [_gt(2, 0, 12, 0), _gt(2, 1, 32, 5, "Van")]}


@pytest.fixture
def objects():
    return {0: [_obj(0, 10.5, 0), _obj(0, 30, 5.8)],
            1: [_obj(1, 11.3, 0), _obj(1, 50, 0), _obj(1, 0, 0, status="empty")],   # empty 는 예측에서 제외
            2: [_obj(2, 12, 0.4), _obj(2, 32, 5)]}


# ---------- 검출 지표 (변형 A/B 형식) ----------
def test_precision_recall_and_errors_match_hand_calculation(gts, objects):
    r = evaluate_variant(objects, gts, CFG, "synthetic")
    o = r["overall"]
    assert (o["n_gt"], o["n_pred"], o["n_matched"], o["n_fp"], o["n_fn"]) == (6, 6, 5, 1, 1)
    assert o["precision"] == pytest.approx(5 / 6)
    assert o["recall"] == pytest.approx(5 / 6)
    assert o["f1"] == pytest.approx(5 / 6)
    assert o["mota"] == pytest.approx(1 - 2 / 6)
    assert o["bev_mean"] == pytest.approx(0.4)
    assert o["bev_median"] == pytest.approx(0.4)
    assert o["depth_bias"] == pytest.approx(0.16)
    assert o["depth_abs_mean"] == pytest.approx(0.16)
    assert r["tracking"] is None and not r["has_tracks"]


def test_iou_is_nan_when_predictions_have_no_size(gts, objects):
    o = evaluate_variant(objects, gts, CFG, "A")["overall"]
    assert o["iou_n"] == 0
    assert math.isnan(o["iou_mean"]) and math.isnan(o["iou_ge05_frac"])
    assert fmt(o["iou_mean"]) == "n/a"


def test_iou_computed_when_size_present(gts):
    # GT 와 같은 중심·크기의 AABB 예측 → IoU 1.0; 프레임 1 은 GT 절반만 겹치는 박스 (x 로 2 m 이동 → IoU 1/3)
    objs = {0: [_obj(0, 10, 0, size=[4.0, 1.8, 1.5])], 1: [_obj(1, 13, 0, size=[4.0, 1.8, 1.5])]}
    o = evaluate_variant(objs, gts, CFG, "B")["overall"]
    assert o["iou_n"] == 2
    assert o["iou_mean"] == pytest.approx((1.0 + 1 / 3) / 2)
    assert o["iou_ge05_frac"] == pytest.approx(0.5)


def test_by_range_and_by_class(gts, objects):
    r = evaluate_variant(objects, gts, CFG, "A")
    br = r["by_range"]
    assert br["0-20"]["n_gt"] == 3 and br["0-20"]["recall"] == pytest.approx(1.0)
    assert br["20-40"]["n_gt"] == 3 and br["20-40"]["recall"] == pytest.approx(2 / 3)
    assert br["40-70"]["n_gt"] == 0 and math.isnan(br["40-70"]["recall"])
    assert br["40-70"]["n_pred"] == 1 and br["40-70"]["precision"] == pytest.approx(0.0)   # (50, 0) FP 는 예측 거리 기준 40-70 구간
    bc = r["by_class"]
    assert bc["Car"]["n_gt"] == 3 and bc["Car"]["recall"] == pytest.approx(1.0)
    assert bc["Van"]["n_gt"] == 3 and bc["Van"]["recall"] == pytest.approx(2 / 3)
    assert bc["Pedestrian"]["n_gt"] == 0 and math.isnan(bc["Pedestrian"]["recall"])


def test_threshold_changes_matching(gts, objects):
    tight = evaluate_variant(objects, gts, CFG, "A", max_dist=0.45)["overall"]     # 0.5, 0.8 오차 쌍은 탈락
    assert tight["n_matched"] == 3
    assert tight["bev_mean"] == pytest.approx((0.3 + 0.4 + 0.0) / 3)


def test_empty_frames_and_no_predictions(gts):
    r = evaluate_variant({}, gts, CFG, "none")
    o = r["overall"]
    assert (o["n_gt"], o["n_pred"], o["n_matched"]) == (6, 0, 0)
    assert o["recall"] == 0.0 and math.isnan(o["precision"]) and math.isnan(o["bev_mean"])
    assert len(r["per_frame"]) == 3 and r["per_frame"]["n_matched"].sum() == 0
    # GT 없는 프레임 + 예측만 있는 프레임 → 모두 FP
    r2 = evaluate_variant({7: [_obj(7, 10, 0)]}, {}, CFG, "fp_only")
    assert (r2["overall"]["n_gt"], r2["overall"]["n_pred"], r2["overall"]["n_fp"]) == (0, 1, 1)
    assert math.isnan(r2["overall"]["recall"]) and math.isnan(r2["overall"]["mota"])


def test_frames_subset(gts, objects):
    o = evaluate_variant(objects, gts, CFG, "A", frames=[0, 1])["overall"]
    assert (o["n_gt"], o["n_pred"], o["n_matched"]) == (4, 4, 3)


# ---------- 추적 지표 (변형 C 형식) ----------
def test_tracking_metrics_id_switch_and_fragmentation(gts):
    # g0 는 프레임 0,1 에 트랙 1, 프레임 2 에 트랙 7 → IDSW 1. g1 은 프레임 0 과 2 에만 매칭 → 단절 1 (프레임 1 은 coasting 예측이 멀리 있음)
    tracks = {0: [_track_dict(0, 1, 10, 0), _track_dict(0, 2, 30, 5)],
              1: [_track_dict(1, 1, 11, 0), _track_dict(1, 2, 40, 5, misses=1), _track_dict(1, 9, 5, 0, confirmed=False)],
              2: [_track_dict(2, 7, 12, 0), _track_dict(2, 2, 32, 5)]}
    r = evaluate_variant(tracks, gts, CFG, "C")
    assert r["has_tracks"]
    t = r["tracking"]
    assert t["idsw"] == 1
    assert t["frag"] == 1
    assert t["n_track_ids"] == 3 and t["n_gt_tracklets"] == 2      # tentative 트랙 9 는 제외
    assert t["n_coasting_pred"] == 1 and t["n_coasting_matched"] == 0
    cov = t["coverage"]
    assert cov[0]["cov_any"] == pytest.approx(1.0) and cov[0]["cov_single"] == pytest.approx(2 / 3) and cov[0]["n_ids"] == 2
    assert cov[1]["cov_any"] == pytest.approx(2 / 3) and cov[1]["best_track_id"] == 2
    assert (t["mt"], t["pt"], t["ml"]) == (0, 2, 0)
    assert r["overall"]["mota"] == pytest.approx(1 - (1 + 1 + 1) / 6)      # FN 1 (g1@1), FP 1 (coasting), IDSW 1


def test_include_coasting_false_drops_predicted_only_tracks(gts):
    tracks = {0: [_track_dict(0, 1, 10, 0, misses=2)]}
    assert evaluate_variant(tracks, gts, CFG, "C")["overall"]["n_pred"] == 1
    assert evaluate_variant(tracks, gts, CFG, "C", include_coasting=False)["overall"]["n_pred"] == 0


def test_track_dataclass_input(gts):
    tr = Track(track_id=3, class_name="car", state=np.array([10.0, 0, -0.8, 1, 0, 0]), covariance=np.eye(6), confirmed=True)
    o = evaluate_variant({0: [tr]}, gts, CFG, "C")["overall"]
    assert o["n_pred"] == 1 and o["n_matched"] == 1
    tr.confirmed = False
    assert evaluate_variant({0: [tr]}, gts, CFG, "C")["overall"]["n_pred"] == 0


# ---------- 보조 함수 ----------
def test_to_pred_records_rejects_unknown_type():
    with pytest.raises(TypeError):
        to_pred_records([object()])


def test_select_gt_filters_fov_range_and_class():
    boxes = [_gt(0, 0, 10, 0), _gt(0, 1, -5, 0), _gt(0, 2, 10, 20), _gt(0, 3, 80, 0), _gt(0, 4, 10, 0, cls="Tram")]
    kept = select_gt(boxes, x_max=70.0, classes=["Car"])
    assert [b.tracklet_id for b in kept] == [0]


def test_range_label_bins():
    assert range_label(0.0) == "0-20" and range_label(19.99) == "0-20"
    assert range_label(20.0) == "20-40" and range_label(40.0) == "40-70"
    assert range_label(70.0) == "40-70" and range_label(70.01) is None


def test_results_to_long_has_expected_columns(gts, objects):
    r = evaluate_variant(objects, gts, CFG, "A")
    df = results_to_long([r])
    assert list(df.columns) == ["variant", "max_dist_m", "section", "group", "metric", "value"]
    assert set(df["section"]) == {"overall", "range", "class"}
    row = df[(df["section"] == "overall") & (df["metric"] == "recall")]
    assert len(row) == 1 and row["value"].iloc[0] == pytest.approx(5 / 6)
    assert df.loc[(df["section"] == "class") & (df["group"] == "Pedestrian") & (df["metric"] == "recall"), "value"].isna().all()
