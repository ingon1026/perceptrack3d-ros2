"""Phase 5–6 융합 모듈 테스트. 합성 데이터 위주로 빠르게, 마지막에 실제 프레임 0 통합 테스트 1개."""
from __future__ import annotations

import math

import numpy as np
import pytest

from perceptrack3d.fusion.frustum_fusion import (
    dedup_objects,
    fuse_frame_clustered,
    fuse_frame_raw,
    surface_to_center_offset,
)
from perceptrack3d.fusion.point_filters import (
    aabb_from_points,
    cluster_points,
    frustum_mask,
    obb_from_points_bev,
    remove_ground,
    roi_filter,
    shrink_box,
    uv_in_box,
)
from perceptrack3d.geometry.calibration import KittiCalibration
from perceptrack3d.geometry.transforms import make_rigid
from perceptrack3d.types import Detection2D, Object3D

RNG = np.random.default_rng(0)

ROI = dict(x_min=0.0, x_max=70.0, y_abs_max=40.0, z_min=-3.0, z_max=3.0)
GROUND = dict(method="ransac", distance_threshold=0.2, ransac_n=3, num_iterations=200)
CLUSTER = dict(eps=0.8, min_points=5)


def make_fusion_cfg(**over) -> dict:
    f = dict(min_points=5, depth_percentile=30, depth_band_m=1.5, box_shrink=0.1, roi=ROI, ground=GROUND, cluster=CLUSTER,
             dedup_distance_m=1.0, cluster_select="largest", use_obb=False, max_extent_m=15.0,
             surface_offset=dict(enable=False, half_depth_m={"car": 2.0}))
    f.update(over)
    return {"fusion": f}


def synthetic_calib() -> KittiCalibration:
    """Velodyne (x 전방, y 좌, z 상) → cam (x 우, y 아래, z 전방) 축 치환만 있는 가상 캘리브레이션 + 단순 핀홀 P."""
    R = np.array([[0.0, -1.0, 0.0], [0.0, 0.0, -1.0], [1.0, 0.0, 0.0]])
    T = make_rigid(R, np.zeros(3))
    P = np.array([[700.0, 0.0, 621.0, 0.0], [0.0, 700.0, 187.5, 0.0], [0.0, 0.0, 1.0, 0.0]])
    return KittiCalibration(T_velo_to_cam=T, R_rect_00=np.eye(4), P_rect_02=P, image_size=(1242, 375))


def det(xyxy, class_name="car", conf=0.9, frame_id=0) -> Detection2D:
    return Detection2D(frame_id=frame_id, class_name=class_name, class_id=2, confidence=conf,
                       xyxy=np.asarray(xyxy, dtype=np.float32))


def box_points(center, size, n=200, rng=None) -> np.ndarray:
    """center 주변 size 크기의 상자 안에 균일 분포한 n 개 점 (N, 3)."""
    rng = RNG if rng is None else rng
    return np.asarray(center) + (rng.random((n, 3)) - 0.5) * np.asarray(size)


def project(calib, pts):
    h = np.hstack([pts, np.ones((len(pts), 1))]) @ calib.P_velo_to_img.T
    return h[:, :2] / h[:, 2:3]


# ---------------------------------------------------------------- point_filters

def test_roi_filter_boundaries():
    pts = np.array([
        [0.0, 0.0, 0.0],        # x_min 경계 → 포함
        [70.0, 40.0, 3.0],      # 모든 상한 경계 → 포함
        [-0.1, 0.0, 0.0],       # 뒤 → 제외
        [10.0, 40.1, 0.0],      # 옆 → 제외
        [10.0, 0.0, -3.1],      # 아래 → 제외
        [70.1, 0.0, 0.0],       # 멀리 → 제외
    ])
    np.testing.assert_array_equal(roi_filter(pts, ROI), [True, True, False, False, False, False])


def test_roi_filter_accepts_n4_and_rejects_bad_shape():
    pts4 = np.hstack([box_points([10, 0, 0], [2, 2, 2], 50), np.zeros((50, 1))])
    assert roi_filter(pts4, ROI).all()
    with pytest.raises(ValueError):
        roi_filter(np.zeros((5, 2)), ROI)


def test_remove_ground_keeps_box_points_only():
    gx, gy = np.meshgrid(np.linspace(1, 40, 60), np.linspace(-15, 15, 40))
    ground = np.stack([gx.ravel(), gy.ravel(), np.full(gx.size, -1.7) + RNG.normal(0, 0.02, gx.size)], axis=1)
    car = box_points([15.0, 2.0, -0.6], [4.0, 1.8, 1.5], 300)          # 바닥 z ≈ -1.35 (지면 +0.35 m, 실제 차 하부 높이)
    pts = np.vstack([ground, car])
    mask, plane = remove_ground(pts, GROUND)
    assert mask.shape == (len(pts),) and plane.shape == (4,)
    assert abs(-plane[3] / plane[2] + 1.7) < 0.05                       # 평면 z ≈ -1.7 (원점에서)
    assert plane[2] > 0.99                                              # 법선이 위쪽
    assert mask[len(ground):].mean() > 0.9                              # 상자 점은 대부분 남고
    assert mask[:len(ground)].mean() < 0.02                             # 지면 점은 대부분 제거


def test_remove_ground_deterministic_and_small_input():
    pts = np.vstack([box_points([10, 0, -1.7], [20, 20, 0.02], 500), box_points([10, 0, 0], [2, 2, 1], 100)])
    m1, p1 = remove_ground(pts, GROUND)
    m2, p2 = remove_ground(pts, GROUND)
    np.testing.assert_array_equal(m1, m2)
    np.testing.assert_allclose(p1, p2)
    m, p = remove_ground(pts[:2], GROUND)                                # 점이 너무 적으면 제거하지 않음
    assert m.all() and np.isnan(p).all()


def test_shrink_box_and_uv_in_box():
    np.testing.assert_allclose(shrink_box([0, 0, 100, 50], 0.1), [10, 5, 90, 45])
    np.testing.assert_allclose(shrink_box([0, 0, 100, 50], 0.0), [0, 0, 100, 50])
    with pytest.raises(ValueError):
        shrink_box([0, 0, 10, 10], 0.5)
    uv = np.array([[10, 5], [90, 45], [9.9, 5], [50, 45.1]])
    np.testing.assert_array_equal(uv_in_box(uv, [10, 5, 90, 45]), [True, True, False, False])


def test_frustum_mask_includes_center_point_excludes_outside():
    calib = synthetic_calib()
    pts = np.array([
        [20.0, 0.0, 0.0],       # 카메라 정면 → 픽셀 (621, 187.5) → 중앙 박스 안
        [20.0, 5.0, 0.0],       # 왼쪽으로 5 m → u = 621 - 700*5/20 = 446 → 박스 밖
        [-20.0, 0.0, 0.0],      # 카메라 뒤 → 제외
        [20.0, 0.0, 10.0],      # 위로 10 m → v < 0, 이미지 밖
    ])
    xyxy = np.array([571, 137, 671, 237])                               # 중앙 100x100 박스
    m = frustum_mask(pts, xyxy, calib, calib.image_shape, shrink=0.1)
    np.testing.assert_array_equal(m, [True, False, False, False])
    # shrink 를 0 으로 하면 박스 가장자리에 걸친 점이 들어온다: u = 621 - 700*y/20 = 575 → y = 1.314
    edge = np.array([[20.0, 1.30, 0.0]])
    assert frustum_mask(edge, xyxy, calib, calib.image_shape, shrink=0.0)[0]
    assert not frustum_mask(edge, xyxy, calib, calib.image_shape, shrink=0.1)[0]


def test_cluster_points_two_blobs():
    a = box_points([10, 0, 0], [1.0, 1.0, 1.0], 60)
    b = box_points([10, 6, 0], [1.0, 1.0, 1.0], 60)
    lone = np.array([[10.0, 3.0, 0.0]])                                  # 고립점 → noise(-1)
    labels = cluster_points(np.vstack([a, b, lone]), CLUSTER)
    assert labels.shape == (121,)
    assert labels[-1] == -1
    assert len(set(labels[:60])) == 1 and len(set(labels[60:120])) == 1
    assert labels[0] != labels[60]
    assert len(cluster_points(np.zeros((0, 3)), CLUSTER)) == 0


def test_aabb_from_points_recovers_box():
    c, s = aabb_from_points(box_points([10, -2, 0.5], [4.0, 1.8, 1.5], 2000))
    np.testing.assert_allclose(c, [10, -2, 0.5], atol=0.05)
    np.testing.assert_allclose(s, [4.0, 1.8, 1.5], atol=0.05)
    with pytest.raises(ValueError):
        aabb_from_points(np.zeros((0, 3)))


@pytest.mark.parametrize("yaw_true", [0.0, 0.5, -1.2, 1.4])
def test_obb_from_points_bev_recovers_size_and_yaw(yaw_true):
    local = box_points([0, 0, 0], [4.0, 1.8, 1.5], 3000)
    c, s = math.cos(yaw_true), math.sin(yaw_true)
    rot = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    pts = local @ rot.T + np.array([20.0, 5.0, 0.5])
    center, size, yaw = obb_from_points_bev(pts)
    np.testing.assert_allclose(center, [20.0, 5.0, 0.5], atol=0.05)
    np.testing.assert_allclose(size, [4.0, 1.8, 1.5], atol=0.08)
    d = (yaw - yaw_true + math.pi / 2) % math.pi - math.pi / 2           # ±π 대칭 (앞뒤 구분 불가)
    assert abs(d) < 0.03


def test_obb_axis_aligned_matches_aabb():
    pts = box_points([10, 0, 0], [4.0, 1.8, 1.5], 2000)
    ca, sa = aabb_from_points(pts)
    co, so, yaw = obb_from_points_bev(pts)
    np.testing.assert_allclose(co, ca, atol=0.05)
    np.testing.assert_allclose(so, sa, atol=0.08)
    assert abs(yaw) < 0.05 or abs(abs(yaw) - math.pi / 2) < 0.05


# ---------------------------------------------------------------- frustum_fusion (합성)

def scene(calib, car_center=(20.0, 0.0, 0.0), n_car=200, with_wall=True, n_wall=100):
    """카메라 정면 car_center 에 상자(차), 그 뒤 40 m 에 벽(배경) 을 둔 합성 장면과 차를 감싸는 2D 박스.

    n_wall 기본 100: 실제 LiDAR 는 거리²에 반비례해 성기므로 3배 먼 배경은 박스 안 점의 소수(< 30%) 다.
    n_wall = 600 으로 주면 "배경이 다수" 인 극단적 오염 장면이 된다 (raw 백분위와 largest 규칙이 실패하는 사례).
    """
    rng = np.random.default_rng(1)                                       # 테스트 실행 순서와 무관하게 같은 장면
    car = box_points(car_center, [4.0, 1.8, 1.5], n_car, rng)
    uv = project(calib, car)
    x1, y1 = uv.min(axis=0) - 5
    x2, y2 = uv.max(axis=0) + 5
    # 박스를 10% 축소해도 차가 들어가도록 여유를 두고 만든다
    w, h = x2 - x1, y2 - y1
    xyxy = np.array([x1 - 0.15 * w, y1 - 0.15 * h, x2 + 0.15 * w, y2 + 0.15 * h])
    pts = [car]
    if with_wall:
        wall = box_points([car_center[0] + 40.0, car_center[1], 0.0], [0.2, 12.0, 4.0], n_wall, rng)   # 뒤쪽 벽
        pts.append(wall)
    return np.vstack(pts), xyxy


def test_fuse_frame_raw_empty_sparse_ok():
    calib = synthetic_calib()
    cfg = make_fusion_cfg()
    pts, xyxy = scene(calib)
    # 빈 박스: 점이 하나도 없는 곳
    objs = fuse_frame_raw(pts, [det([0, 0, 50, 50])], calib, calib.image_shape, cfg)
    assert objs[0].status == "empty" and objs[0].center is None
    # 희소: 차 점 3개만
    few = pts[:3]
    objs = fuse_frame_raw(few, [det(xyxy)], calib, calib.image_shape, cfg)
    assert objs[0].status == "sparse" and objs[0].center is None and objs[0].n_points == 3
    # 정상: 벽(60 m) 점은 깊이 백분위·밴드로 걸러져 중심이 차(18~22 m) 안에 온다.
    # 하위 30% 깊이를 대표로 쓰므로 설계상 가까운 쪽으로 치우친다 (x ≈ 19~20). 이 편향이 Phase 6 표면 오프셋의 동기다.
    objs = fuse_frame_raw(pts, [det(xyxy)], calib, calib.image_shape, cfg)
    o = objs[0]
    assert o.status == "ok" and o.method == "raw" and o.size is None
    assert 18.5 < o.center[0] < 21.0
    np.testing.assert_allclose(o.center[1:], [0.0, 0.0], atol=0.5)
    assert o.n_points > 50 and o.point_indices is not None and len(o.point_indices) == o.n_points
    assert not o.truncated


def test_fuse_frame_raw_rejects_bad_points_shape():
    calib = synthetic_calib()
    with pytest.raises(ValueError):
        fuse_frame_raw(np.zeros((10, 2)), [], calib, calib.image_shape, make_fusion_cfg())


def test_fuse_frame_raw_no_detections_returns_empty_list():
    calib = synthetic_calib()
    pts, _ = scene(calib)
    assert fuse_frame_raw(pts, [], calib, calib.image_shape, make_fusion_cfg()) == []


def test_dedup_keeps_higher_confidence():
    calib = synthetic_calib()
    cfg = make_fusion_cfg()
    pts, xyxy = scene(calib)
    dets = [det(xyxy, "truck", 0.6), det(xyxy + np.array([2, 2, -2, -2]), "car", 0.9)]
    objs = fuse_frame_raw(pts, dets, calib, calib.image_shape, cfg)
    assert [o.status for o in objs] == ["invalid", "ok"]
    assert objs[0].reason.startswith("duplicate") and objs[0].center is None
    # 멀리 떨어진 두 객체는 유지
    a = Object3D(0, "car", 0.9, np.zeros(4), center=np.array([10.0, 0.0, 0.0]))
    b = Object3D(0, "car", 0.8, np.zeros(4), center=np.array([10.0, 3.0, 0.0]))
    assert all(o.status == "ok" for o in dedup_objects([a, b], 1.0))


def test_fuse_frame_clustered_selects_car_not_wall():
    calib = synthetic_calib()
    cfg = make_fusion_cfg()
    pts, xyxy = scene(calib)
    ground = np.stack([RNG.uniform(1, 70, 3000), RNG.uniform(-20, 20, 3000), np.full(3000, -1.7)], axis=1)
    pts = np.vstack([pts, ground])
    o = fuse_frame_clustered(pts, [det(xyxy)], calib, calib.image_shape, cfg)[0]
    assert o.status == "ok" and o.method == "clustered"
    np.testing.assert_allclose(o.center, [20.0, 0.0, 0.0], atol=0.5)
    np.testing.assert_allclose(o.size, [4.0, 1.8, 1.5], atol=0.3)
    assert o.yaw == 0.0
    # 클러스터 점은 모두 차 점 (인덱스 < 200)
    assert (o.point_indices < 200).all()


def test_dense_background_failure_and_nearest_rule():
    """배경(벽) 점이 차보다 3배 많은 극단 장면: raw 백분위와 largest 는 벽에 끌려가고, nearest(min_ratio) 는 차를 고른다."""
    calib = synthetic_calib()
    pts, xyxy = scene(calib, n_wall=600)
    raw = fuse_frame_raw(pts, [det(xyxy)], calib, calib.image_shape, make_fusion_cfg())[0]
    assert raw.status == "ok" and raw.center[0] > 20.5                  # 하위 30% 깊이가 이미 차 뒤쪽 → 오염
    largest = fuse_frame_clustered(pts, [det(xyxy)], calib, calib.image_shape, make_fusion_cfg())[0]
    assert largest.status == "ok" and largest.center[0] > 50            # 점 수 최대 = 벽
    nearest = fuse_frame_clustered(pts, [det(xyxy)], calib, calib.image_shape,
                                   make_fusion_cfg(cluster_select="nearest", cluster_min_ratio=0.3))[0]
    np.testing.assert_allclose(nearest.center, [20.0, 0.0, 0.0], atol=0.5)


def test_fuse_frame_clustered_statuses_and_obb():
    calib = synthetic_calib()
    pts, xyxy = scene(calib, with_wall=False)
    # 지면 밖(ROI z 아래) 에 있는 점만 → ROI 필터 후 비어 있음
    below = pts.copy(); below[:, 2] = -5.0
    o = fuse_frame_clustered(below, [det(xyxy)], calib, calib.image_shape, make_fusion_cfg())[0]
    assert o.status == "empty"
    # 너무 큰 클러스터 → oversize invalid
    o = fuse_frame_clustered(pts, [det(xyxy)], calib, calib.image_shape, make_fusion_cfg(max_extent_m=1.0))[0]
    assert o.status == "invalid" and o.reason.startswith("oversize")
    # OBB 옵션 → yaw 가 float 이고 size 가 복원됨
    o = fuse_frame_clustered(pts, [det(xyxy)], calib, calib.image_shape, make_fusion_cfg(use_obb=True))[0]
    assert o.status == "ok" and isinstance(o.yaw, float)
    np.testing.assert_allclose(sorted(o.size[:2]), [1.8, 4.0], atol=0.3)


def test_surface_offset_pushes_along_ray():
    # 뒷면만 보이는 차: x 방향 퍼짐 0.4 m → push = 2.0 - 0.2 = 1.8 m 전방으로
    pts = box_points([20.0, 0.0, 0.0], [0.4, 1.8, 1.5], 300)
    c = surface_to_center_offset(np.array([20.0, 0.0, 0.0]), pts, 2.0)
    np.testing.assert_allclose(c, [21.8, 0.0, 0.0], atol=0.05)
    # 옆면까지 다 보이면(퍼짐 4 m ≥ 2·half_depth) 밀지 않는다
    pts = box_points([20.0, 0.0, 0.0], [4.0, 1.8, 1.5], 300)
    np.testing.assert_allclose(surface_to_center_offset(np.array([20.0, 0.0, 0.0]), pts, 2.0), [20.0, 0.0, 0.0], atol=0.05)
    # half_depth 0 → 그대로
    np.testing.assert_allclose(surface_to_center_offset(np.array([20.0, 3.0, 0.0]), pts, 0.0), [20.0, 3.0, 0.0])


def test_truncated_flag():
    calib = synthetic_calib()
    pts, _ = scene(calib)
    objs = fuse_frame_raw(pts, [det([0, 100, 200, 300]), det([300, 100, 500, 300])], calib, calib.image_shape, make_fusion_cfg())
    assert objs[0].truncated and not objs[1].truncated


# ---------------------------------------------------------------- 실제 데이터 통합

def test_real_frame0_both_methods(cfg, require_dataset):
    from pathlib import Path

    from perceptrack3d.data.kitti_loader import KittiDataset
    from perceptrack3d.detection.detector import load_detections_json

    det_path = Path(cfg["outputs"]["dir"]) / "phase4" / "detections.json"
    if not det_path.is_file():
        pytest.skip("outputs/phase4/detections.json 이 없습니다 (Phase 4 먼저 실행)")
    ds = KittiDataset(cfg)
    calib = KittiCalibration.from_dir(cfg["dataset"]["calib_dir"])
    fr = ds.load_frame(0)
    dets = load_detections_json(det_path)[0]
    for fn in (fuse_frame_raw, fuse_frame_clustered):
        objs = fn(fr["points"], dets, calib, fr["image"].shape[:2], cfg)
        assert len(objs) == len(dets)
        ok = [o for o in objs if o.status == "ok"]
        assert len(ok) >= 1
        for o in ok:
            assert o.center.shape == (3,) and 0 < o.center[0] < 70      # 카메라 앞, ROI 안
            assert o.n_points >= cfg["fusion"]["min_points"]
