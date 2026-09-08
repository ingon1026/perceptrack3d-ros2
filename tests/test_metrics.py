"""evaluation/metrics.py 테스트."""
from __future__ import annotations

import math

import numpy as np
import pytest

from perceptrack3d.evaluation.metrics import (
    bev_iou,
    center_errors,
    convex_intersection,
    count_id_switches,
    depth_errors,
    match_by_center_distance,
    polygon_area,
    precision_recall,
    summarize_runtime,
    track_fragmentation,
)
from perceptrack3d.types import GtBox3D, Object3D


def _box(x, y, l=4.0, w=2.0, yaw=0.0):
    return (np.array([x, y, 0.0]), np.array([l, w, 1.5]), yaw)


# ---------- BEV IoU ----------
def test_bev_iou_identical_is_one():
    assert bev_iou(_box(3, 1, yaw=0.3), _box(3, 1, yaw=0.3)) == pytest.approx(1.0)


def test_bev_iou_disjoint_is_zero():
    assert bev_iou(_box(0, 0), _box(10, 0)) == 0.0
    assert bev_iou(_box(0, 0), _box(4.0, 0)) == pytest.approx(0.0)        # 변이 맞닿음


def test_bev_iou_half_overlap_is_one_third():
    # 4x2 박스를 x 로 2 만큼 밀면 교집합 2x2=4, 합집합 8+8-4=12 → 1/3
    assert bev_iou(_box(0, 0), _box(2, 0)) == pytest.approx(1 / 3)


def test_bev_iou_rotated_square_90deg_is_one():
    assert bev_iou(_box(0, 0, l=2, w=2, yaw=0.0), _box(0, 0, l=2, w=2, yaw=math.pi / 2)) == pytest.approx(1.0)


def test_bev_iou_rotated_square_45deg():
    # 단위 정사각형(변 2) 과 45° 회전한 같은 정사각형: 교집합 = 정팔각형 넓이 8(√2−1) ≈ 3.3137, 합집합 = 8 − 3.3137
    inter = 8 * (math.sqrt(2) - 1)
    expect = inter / (8 - inter)
    assert bev_iou(_box(0, 0, l=2, w=2), _box(0, 0, l=2, w=2, yaw=math.pi / 4)) == pytest.approx(expect, rel=1e-9)


def test_bev_iou_accepts_dataclasses_and_none_yaw():
    gt = GtBox3D(frame_id=0, tracklet_id=0, object_type="Car", center=np.array([5.0, 0, 0]),
                 size=np.array([4.0, 2.0, 1.5]), yaw=0.0)
    obj = Object3D(frame_id=0, class_name="car", confidence=0.9, xyxy=np.zeros(4),
                   center=np.array([5.0, 0, 0]), size=np.array([4.0, 2.0, 1.5]), yaw=None)
    assert bev_iou(gt, obj) == pytest.approx(1.0)


def test_convex_intersection_orientation_independent():
    sq = np.array([[0, 0], [2, 0], [2, 2], [0, 2]], float)
    other = np.array([[1, 1], [3, 1], [3, 3], [1, 3]], float)
    assert polygon_area(convex_intersection(sq, other)) == pytest.approx(1.0)
    assert polygon_area(convex_intersection(sq[::-1], other[::-1])) == pytest.approx(1.0)
    assert len(convex_intersection(sq, other + 10)) == 0


# ---------- 매칭 ----------
def test_match_threshold_and_one_to_one():
    preds = np.array([[10.0, 0.0, 0.0], [20.0, 0.0, 0.0], [50.0, 0.0, 0.0]])
    gts = np.array([[10.5, 0.2, -1.0], [21.0, 0.0, 0.0], [30.0, 0.0, 0.0]])
    m = match_by_center_distance(preds, gts, max_dist=2.0)
    assert [(p, g) for p, g, _ in m] == [(0, 0), (1, 1)]              # 3번 예측(50m) 은 임계값 밖
    assert m[0][2] == pytest.approx(math.hypot(0.5, 0.2))              # BEV 거리 (z 무시)
    assert match_by_center_distance(preds, gts, max_dist=0.1) == []
    assert match_by_center_distance(np.zeros((0, 3)), gts, 2.0) == []


def test_match_prefers_global_minimum():
    # 예측 0 은 GT 0/1 모두 가깝지만, 총합 최소는 (0→0, 1→1)
    preds = np.array([[0.0, 0.0, 0], [1.2, 0.0, 0]])
    gts = np.array([[0.0, 0.0, 0], [1.0, 0.0, 0]])
    assert [(p, g) for p, g, _ in match_by_center_distance(preds, gts, 2.0)] == [(0, 0), (1, 1)]


def test_match_accepts_dataclasses():
    obj = Object3D(frame_id=0, class_name="car", confidence=1.0, xyxy=np.zeros(4), center=np.array([5.0, 1.0, 0.0]))
    gt = GtBox3D(frame_id=0, tracklet_id=3, object_type="Car", center=np.array([5.5, 1.0, 0.5]),
                 size=np.ones(3), yaw=0.0)
    m = match_by_center_distance([obj], [gt], 2.0)
    assert m == [(0, 0, pytest.approx(0.5))]
    with pytest.raises(ValueError):
        match_by_center_distance([Object3D(0, "car", 1.0, np.zeros(4), center=None, status="empty")], [gt], 2.0)


# ---------- 오차 통계 ----------
def test_center_and_depth_errors():
    preds = np.array([[11.0, 0.0, 0.0], [19.0, 1.0, 0.0]])
    gts = np.array([[10.0, 0.0, 0.0], [20.0, 0.0, 0.0]])
    matches = [(0, 0, 1.0), (1, 1, math.sqrt(2))]
    ce = center_errors(matches, preds, gts)
    assert ce["n"] == 2
    assert ce["mean_bev"] == pytest.approx((1 + math.sqrt(2)) / 2) and ce["mean_3d"] == pytest.approx(ce["mean_bev"])
    de = depth_errors(matches, preds, gts)
    assert de["mean_abs"] == pytest.approx(1.0) and de["bias"] == pytest.approx(0.0)   # +1, -1 → 편향 0
    de2 = depth_errors([(0, 0, 1.0)], preds, gts)
    assert de2["bias"] == pytest.approx(+1.0)                                          # 예측이 더 멀면 양수
    de3 = depth_errors([(1, 1, 1.0)], preds, gts)
    assert de3["bias"] == pytest.approx(-1.0)
    assert math.isnan(center_errors([], preds, gts)["mean_3d"]) and center_errors([], preds, gts)["n"] == 0


# ---------- precision / recall ----------
def test_precision_recall():
    pr = precision_recall(n_matched=6, n_pred=8, n_gt=12)
    assert pr["precision"] == pytest.approx(0.75) and pr["recall"] == pytest.approx(0.5)
    assert pr["f1"] == pytest.approx(2 * 0.75 * 0.5 / 1.25)
    assert precision_recall(0, 0, 0) == {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    with pytest.raises(ValueError):
        precision_recall(5, 4, 10)


# ---------- ID 스위치 / 단절 ----------
def test_count_id_switches_and_fragmentation():
    # (track_id, gt_id) — GT 7: 트랙 1 → 1 → 2 → 2 → 1 (스위치 2회). GT 8: 트랙 5 유지, 프레임 2 에서 미검출 (단절 1회)
    assignments = [
        [(1, 7), (5, 8)],
        [(1, 7), (5, 8)],
        [(2, 7)],
        [(2, 7), (5, 8)],
        [(1, 7), (5, 8)],
    ]
    assert count_id_switches(assignments) == 2
    assert track_fragmentation(assignments) == 1
    assert count_id_switches([[(1, 7)], [], [(1, 7)]]) == 0            # 미검출 후 같은 트랙 복귀는 스위치 아님
    assert track_fragmentation([[(1, 7)], [], [(1, 7)]]) == 1
    assert count_id_switches([]) == 0 and track_fragmentation([]) == 0


# ---------- 런타임 ----------
def test_summarize_runtime():
    s = summarize_runtime({"detect": [10.0, 20.0, 30.0], "empty": []})
    assert "empty" not in s
    d = s["detect"]
    assert d["n"] == 3 and d["mean_ms"] == pytest.approx(20.0) and d["median_ms"] == pytest.approx(20.0)
    assert d["p95_ms"] == pytest.approx(29.0) and d["fps"] == pytest.approx(50.0)
