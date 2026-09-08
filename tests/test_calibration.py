"""Phase 2: 캘리브레이션 파싱과 강체 변환 테스트."""
import numpy as np
import pytest

from perceptrack3d.geometry.calibration import KittiCalibration, read_calib_file
from perceptrack3d.geometry.transforms import (
    apply_transform,
    invert_rigid,
    make_rigid,
    nearest_rotation,
    to_homogeneous,
)


def _rot_z(theta: float) -> np.ndarray:
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


# ---- transforms.py (데이터 불필요) -------------------------------------------
def test_to_homogeneous():
    pts = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    h = to_homogeneous(pts)
    assert h.shape == (2, 4)
    np.testing.assert_array_equal(h[:, 3], 1.0)
    np.testing.assert_array_equal(h[:, :3], pts)


def test_to_homogeneous_rejects_bad_shape():
    with pytest.raises(ValueError):
        to_homogeneous(np.zeros((5, 4)))


def test_apply_transform_known():
    T = make_rigid(_rot_z(np.pi / 2), [1.0, 0.0, 0.0])      # z 축 90° 회전 후 x 로 1 이동
    out = apply_transform(T, np.array([[1.0, 0.0, 0.0]]))
    np.testing.assert_allclose(out, [[1.0, 1.0, 0.0]], atol=1e-12)


def test_invert_rigid_is_inverse():
    rng = np.random.default_rng(0)
    T = make_rigid(nearest_rotation(rng.standard_normal((3, 3))), rng.standard_normal(3))
    np.testing.assert_allclose(invert_rigid(T) @ T, np.eye(4), atol=1e-12)
    np.testing.assert_allclose(T @ invert_rigid(T), np.eye(4), atol=1e-12)


def test_nearest_rotation_is_orthonormal_and_proper():
    rng = np.random.default_rng(1)
    R = nearest_rotation(rng.standard_normal((3, 3)))
    np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-12)
    assert np.isclose(np.linalg.det(R), 1.0)


def test_nearest_rotation_barely_changes_rounded_rotation():
    R = _rot_z(0.3)
    R_rounded = np.round(R, 6)
    assert np.abs(nearest_rotation(R_rounded) - R_rounded).max() < 1e-5


# ---- calibration.py ----------------------------------------------------------
def test_read_calib_file_skips_non_numeric(tmp_path):
    p = tmp_path / "c.txt"
    p.write_text("calib_time: 15-Mar-2012 11:37:16\nR: 1 0 0 0 1 0 0 0 1\nT: 1 2 3\n")
    d = read_calib_file(p)
    assert "calib_time" not in d
    assert d["R"].shape == (9,) and d["T"].shape == (3,)


def test_read_calib_file_missing_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        read_calib_file(tmp_path / "nope.txt")


def test_calibration_rejects_bad_shapes():
    with pytest.raises(ValueError):
        KittiCalibration(np.eye(3), np.eye(4), np.zeros((3, 4)))
    with pytest.raises(ValueError):
        KittiCalibration(np.eye(4), np.eye(4), np.zeros((4, 4)))


@pytest.fixture(scope="module")
def calib(cfg, dataset_available):
    if not dataset_available:
        pytest.skip("KITTI 데이터셋 없음")
    return KittiCalibration.from_dir(cfg["dataset"]["calib_dir"])


class TestRealCalibration:

    def test_shapes(self, calib):
        assert calib.T_velo_to_cam.shape == (4, 4)
        assert calib.R_rect_00.shape == (4, 4)
        assert calib.P_rect_02.shape == (3, 4)
        assert calib.P_velo_to_img.shape == (3, 4)
        assert calib.image_size == (1242, 375)
        assert calib.image_shape == (375, 1242)

    def test_rotation_orthonormal(self, calib):
        for M in (calib.T_velo_to_cam, calib.R_rect_00, calib.T_velo_to_rect):
            R = M[:3, :3]
            np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-9)
            assert np.isclose(np.linalg.det(R), 1.0, atol=1e-9)
            np.testing.assert_array_equal(M[3], [0, 0, 0, 1])
        np.testing.assert_allclose(calib.R_rect_00[:3, 3], 0.0)   # 정류는 회전만

    def test_round_trip(self, calib):
        rng = np.random.default_rng(0)
        pts = rng.uniform([-80, -40, -3], [80, 40, 3], size=(1000, 3))
        back = calib.rect_to_velo(calib.velo_to_rect(pts))
        np.testing.assert_allclose(back, pts, atol=1e-6)

    def test_invert_rigid_on_real_matrices(self, calib):
        for T in (calib.T_velo_to_cam, calib.T_velo_to_rect):
            np.testing.assert_allclose(invert_rigid(T) @ T, np.eye(4), atol=1e-9)

    def test_axis_conventions(self, calib):
        """Velodyne 전방(+x) → rect 전방(+z), Velodyne 좌(+y) → rect 좌(-x), Velodyne 상(+z) → rect 상(-y)."""
        rect = calib.velo_to_rect(np.array([[10.0, 0.0, 0.0], [10.0, 5.0, 0.0], [10.0, 0.0, 3.0]]))
        assert 9.0 < rect[0, 2] < 11.0 and abs(rect[0, 0]) < 0.5 and abs(rect[0, 1]) < 0.5
        assert rect[1, 0] < -4.0          # 왼쪽 5 m → 카메라 x(우) 음수
        assert rect[2, 1] < -2.0          # 위로 3 m → 카메라 y(아래) 음수

    def test_velodyne_origin_is_behind_camera(self, calib):
        """KITTI 센서 배치: Velodyne 은 카메라보다 약 0.27 m 뒤에 있으므로 velo 원점의 rect z 는 음수."""
        rect = calib.velo_to_rect(np.zeros((1, 3)))[0]
        assert -0.4 < rect[2] < -0.2

    def test_from_dir_missing_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            KittiCalibration.from_dir(tmp_path)
