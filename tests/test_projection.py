"""Phase 3: LiDAR → 이미지 투영 테스트."""
import numpy as np
import pytest

from perceptrack3d.data.kitti_loader import KittiDataset
from perceptrack3d.geometry.calibration import KittiCalibration
from perceptrack3d.geometry.projection import project_velo_to_image


@pytest.fixture(scope="module")
def calib(cfg, dataset_available):
    if not dataset_available:
        pytest.skip("KITTI 데이터셋 없음")
    return KittiCalibration.from_dir(cfg["dataset"]["calib_dir"])


def test_point_behind_camera_is_masked_out(calib):
    pts_velo = calib.rect_to_velo(np.array([[0.0, 0.0, -10.0], [1.0, -1.0, -3.0]]))
    uv, depth, mask = project_velo_to_image(pts_velo, calib, calib.image_shape)
    assert mask.shape == (2,) and not mask.any()
    assert uv.shape == (0, 2) and depth.shape == (0,)


def test_known_point_on_optical_axis(calib):
    """rect 프레임 (0, 0, 10) 을 velo 로 보냈다가 투영하면 P_rect_02 @ [0,0,10,1] 과 같아야 한다."""
    p_rect = np.array([[0.0, 0.0, 10.0]])
    pts_velo = calib.rect_to_velo(p_rect)
    uv, depth, mask = project_velo_to_image(pts_velo, calib, calib.image_shape)
    y = calib.P_rect_02 @ np.array([0.0, 0.0, 10.0, 1.0])
    expected_uv = y[:2] / y[2]
    assert mask.all()
    np.testing.assert_allclose(uv[0], expected_uv, atol=1e-3)
    np.testing.assert_allclose(depth[0], y[2], atol=1e-3)
    # 광축 위 점은 주점 (cx, cy) 근처에 맺힌다 (tx/z 만큼 u 가 어긋남)
    cx, cy = calib.P_rect_02[0, 2], calib.P_rect_02[1, 2]
    assert abs(uv[0, 0] - cx) < 10 and abs(uv[0, 1] - cy) < 1


def test_synthetic_grid_bounds(calib):
    """rect 프레임에 격자를 만들어, 경계 안/밖 판정이 수작업 계산과 같은지 확인."""
    H, W = calib.image_shape
    xs = np.linspace(-15, 15, 31)
    ys = np.linspace(-5, 5, 11)
    X, Y = np.meshgrid(xs, ys)
    p_rect = np.stack([X.ravel(), Y.ravel(), np.full(X.size, 12.0)], axis=1)
    uv, depth, mask = project_velo_to_image(calib.rect_to_velo(p_rect), calib, (H, W))
    y = (calib.P_rect_02 @ np.hstack([p_rect, np.ones((len(p_rect), 1))]).T).T
    u_all, v_all = y[:, 0] / y[:, 2], y[:, 1] / y[:, 2]
    expected = (u_all >= 0) & (u_all < W) & (v_all >= 0) & (v_all < H)
    np.testing.assert_array_equal(mask, expected)
    assert 0 < mask.sum() < len(mask)          # 안쪽/바깥쪽 둘 다 존재
    np.testing.assert_allclose(uv[:, 0], u_all[mask], atol=1e-3)
    np.testing.assert_allclose(uv[:, 1], v_all[mask], atol=1e-3)


@pytest.mark.usefixtures("require_dataset")
def test_frame0_real_points(cfg, calib):
    fr = KittiDataset(cfg).load_frame(0)
    pts = fr["points"]
    H, W = fr["image"].shape[:2]
    uv, depth, mask = project_velo_to_image(pts, calib, (H, W))
    assert uv.dtype == np.float32 and depth.dtype == np.float32 and mask.dtype == bool
    assert mask.shape == (len(pts),)
    assert mask.sum() == len(uv) == len(depth)
    assert 1_000 < len(uv) < len(pts)                       # 일부만 이미지 안
    assert (uv[:, 0] >= 0).all() and (uv[:, 0] < W).all()
    assert (uv[:, 1] >= 0).all() and (uv[:, 1] < H).all()
    assert (depth > 0).all()
    # 카메라 뒤(velo x < -1) 점은 절대 포함되지 않는다
    assert not mask[pts[:, 0] < -1.0].any()


def test_accepts_n3_and_n4(calib):
    p4 = np.array([[10.0, 0.0, -1.0, 0.5], [12.0, 1.0, 0.0, 0.1]], dtype=np.float32)
    uv4, d4, m4 = project_velo_to_image(p4, calib, calib.image_shape)
    uv3, d3, m3 = project_velo_to_image(p4[:, :3], calib, calib.image_shape)
    np.testing.assert_array_equal(m4, m3)
    np.testing.assert_allclose(uv4, uv3)
    np.testing.assert_allclose(d4, d3)


def test_rejects_bad_shape(calib):
    with pytest.raises(ValueError):
        project_velo_to_image(np.zeros((5, 2)), calib, calib.image_shape)
    with pytest.raises(ValueError):
        project_velo_to_image(np.zeros(3), calib, calib.image_shape)
