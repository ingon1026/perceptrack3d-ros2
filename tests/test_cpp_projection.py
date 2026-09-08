"""Phase 10: C++ 확장(pt3d_cpp) 과 Python 기준 구현의 수치 일치 테스트.

확장 모듈이 빌드되지 않은 환경에서는 전체를 skip 한다 (scripts/build_cpp.sh 로 빌드).
기준: mask 완전 일치, uv allclose(atol 1e-3), depth allclose(rtol 1e-5). ROI/frustum 은 mask 완전 일치.
"""
import json
from pathlib import Path

import numpy as np
import pytest

from perceptrack3d.data.kitti_loader import KittiDataset
from perceptrack3d.geometry.calibration import KittiCalibration
from perceptrack3d.geometry.projection import project_velo_to_image

cpp = pytest.importorskip(
    "perceptrack3d.geometry.projection_cpp",
    reason="C++ 확장 모듈 pt3d_cpp 가 없습니다. scripts/build_cpp.sh 로 빌드하세요.",
)

try:  # 팀 E 의 Python 기준 필터 (있으면 그것도 기준으로 비교)
    from perceptrack3d.fusion import point_filters as pf
except ImportError:
    pf = None


# ---- 참조 구현 (numpy) ------------------------------------------------------------------------------
def roi_ref(pts: np.ndarray, roi: dict) -> np.ndarray:
    x, y, z = pts[:, 0], pts[:, 1], pts[:, 2]
    return (x >= roi["x_min"]) & (x <= roi["x_max"]) & (np.abs(y) <= roi["y_abs_max"]) & (z >= roi["z_min"]) & (z <= roi["z_max"])


def frustum_ref(pts: np.ndarray, boxes: np.ndarray, calib, shape) -> np.ndarray:
    """Python 투영 출력(uv float32)에 포함 경계 박스 조건을 건 뒤 원본 인덱스로 되돌린다. (K, N) bool."""
    uv, _, mask = project_velo_to_image(pts, calib, shape)
    idx = np.flatnonzero(mask)
    out = np.zeros((len(boxes), len(pts)), dtype=bool)
    for k, (x1, y1, x2, y2) in enumerate(np.asarray(boxes, dtype=np.float64)):
        inb = (uv[:, 0] >= x1) & (uv[:, 0] <= x2) & (uv[:, 1] >= y1) & (uv[:, 1] <= y2)
        out[k, idx[inb]] = True
    return out


# ---- fixtures -----------------------------------------------------------------------------------------
@pytest.fixture(scope="module")
def calib(cfg, dataset_available):
    if not dataset_available:
        pytest.skip("KITTI 데이터셋 없음")
    return KittiCalibration.from_dir(cfg["dataset"]["calib_dir"])


@pytest.fixture(scope="module")
def frame0_points(cfg, dataset_available):
    if not dataset_available:
        pytest.skip("KITTI 데이터셋 없음")
    return KittiDataset(cfg).load_frame(0)["points"]  # (N, 4) float32 C-연속


@pytest.fixture(scope="module")
def frame0_boxes(cfg):
    """Phase 4 검출 결과가 있으면 실제 박스, 없으면 이미지 안 임의 박스."""
    path = Path(cfg["outputs"]["dir"]) / "phase4" / "detections.json"
    if path.is_file():
        dets = json.loads(path.read_text()).get("0", [])
        if dets:
            return np.array([d["xyxy"] for d in dets], dtype=np.float64)
    return np.array([[255.7, 174.3, 302.6, 201.1], [569.8, 173.5, 609.6, 208.1], [0.0, 0.0, 1242.0, 375.0]])


# ---- 투영 ---------------------------------------------------------------------------------------------
def test_projection_frame0_matches_python(frame0_points, calib):
    shape = calib.image_shape
    uv_p, d_p, m_p = project_velo_to_image(frame0_points, calib, shape)
    uv_c, d_c, m_c = cpp.project_velo_to_image_cpp(frame0_points, calib, shape)
    assert uv_c.dtype == np.float32 and d_c.dtype == np.float32 and m_c.dtype == bool
    assert m_c.shape == (len(frame0_points),) and uv_c.shape == uv_p.shape and d_c.shape == d_p.shape
    np.testing.assert_array_equal(m_c, m_p)
    np.testing.assert_allclose(uv_c, uv_p, atol=1e-3)
    np.testing.assert_allclose(d_c, d_p, rtol=1e-5)
    assert m_c.sum() == len(uv_c) == len(d_c) > 1_000


def test_projection_n4_and_n3_inputs_agree(frame0_points, calib):
    """(N,4) 를 그대로 넘긴 결과(stride 뷰) 와 (N,3) 복사본을 넘긴 결과가 같아야 한다."""
    a = cpp.project_velo_to_image_cpp(frame0_points, calib, calib.image_shape)
    b = cpp.project_velo_to_image_cpp(np.ascontiguousarray(frame0_points[:, :3]), calib, calib.image_shape)
    for x, y in zip(a, b):
        np.testing.assert_array_equal(x, y)


def test_projection_synthetic_grid_matches_python(calib):
    """float64 격자 입력(변환 경로) 에서도 경계 판정이 Python 과 같은지 확인."""
    H, W = calib.image_shape
    X, Y = np.meshgrid(np.linspace(-15, 15, 31), np.linspace(-5, 5, 11))
    p_rect = np.stack([X.ravel(), Y.ravel(), np.full(X.size, 12.0)], axis=1)
    pts_velo = calib.rect_to_velo(p_rect)
    uv_p, d_p, m_p = project_velo_to_image(pts_velo, calib, (H, W))
    uv_c, d_c, m_c = cpp.project_velo_to_image_cpp(pts_velo, calib, (H, W))
    np.testing.assert_array_equal(m_c, m_p)
    assert 0 < m_c.sum() < len(m_c)
    np.testing.assert_allclose(uv_c, uv_p, atol=1e-3)
    np.testing.assert_allclose(d_c, d_p, rtol=1e-5)


def test_projection_behind_camera_and_empty(calib):
    pts_velo = calib.rect_to_velo(np.array([[0.0, 0.0, -10.0], [1.0, -1.0, -3.0]]))
    uv, depth, mask = cpp.project_velo_to_image_cpp(pts_velo, calib, calib.image_shape)
    assert mask.shape == (2,) and not mask.any() and uv.shape == (0, 2) and depth.shape == (0,)
    uv, depth, mask = cpp.project_velo_to_image_cpp(np.zeros((0, 4), np.float32), calib, calib.image_shape)
    assert mask.shape == (0,) and uv.shape == (0, 2) and depth.shape == (0,)


def test_projection_rejects_bad_shape(calib):
    with pytest.raises(ValueError):
        cpp.project_velo_to_image_cpp(np.zeros((5, 2)), calib, calib.image_shape)
    with pytest.raises(ValueError):
        cpp.project_velo_to_image_cpp(np.zeros(3), calib, calib.image_shape)


# ---- ROI ----------------------------------------------------------------------------------------------
def test_roi_frame0_matches_reference(cfg, frame0_points):
    roi = cfg["fusion"]["roi"]
    m_c = cpp.roi_mask_cpp(frame0_points, roi)
    assert m_c.dtype == bool and m_c.shape == (len(frame0_points),)
    np.testing.assert_array_equal(m_c, roi_ref(frame0_points, roi))
    assert 0 < m_c.sum() < len(m_c)


def test_roi_boundaries_inclusive():
    roi = {"x_min": 0.0, "x_max": 70.0, "y_abs_max": 40.0, "z_min": -3.0, "z_max": 3.0}
    pts = np.array([
        [0.0, 0.0, 0.0], [70.0, 0.0, 0.0], [70.001, 0.0, 0.0], [-0.001, 0.0, 0.0],
        [10.0, 40.0, 0.0], [10.0, -40.0, 0.0], [10.0, -40.001, 0.0],
        [10.0, 0.0, -3.0], [10.0, 0.0, 3.0], [10.0, 0.0, 3.001],
    ], dtype=np.float32)
    expected = np.array([1, 1, 0, 0, 1, 1, 0, 1, 1, 0], dtype=bool)
    np.testing.assert_array_equal(cpp.roi_mask_cpp(pts, roi), expected)
    np.testing.assert_array_equal(roi_ref(pts, roi), expected)


@pytest.mark.skipif(pf is None or not hasattr(pf, "roi_filter"), reason="팀 E point_filters.roi_filter 없음")
def test_roi_matches_team_e(cfg, frame0_points):
    np.testing.assert_array_equal(cpp.roi_mask_cpp(frame0_points, cfg["fusion"]["roi"]), pf.roi_filter(frame0_points, cfg["fusion"]["roi"]))


# ---- frustum ------------------------------------------------------------------------------------------
def test_frustum_frame0_matches_reference(frame0_points, frame0_boxes, calib):
    shape = calib.image_shape
    m_c = cpp.frustum_masks_cpp(frame0_points, frame0_boxes, calib, shape)
    assert m_c.dtype == bool and m_c.shape == (len(frame0_boxes), len(frame0_points))
    np.testing.assert_array_equal(m_c, frustum_ref(frame0_points, frame0_boxes, calib, shape))
    assert m_c.any(axis=1).any()  # 적어도 한 박스에는 점이 있다


def test_frustum_shrink_matches_reference(frame0_points, frame0_boxes, calib):
    shape = calib.image_shape
    shrunk = cpp.shrink_boxes(frame0_boxes, 0.1)
    assert (shrunk[:, 0] > frame0_boxes[:, 0]).all() and (shrunk[:, 2] < frame0_boxes[:, 2]).all()
    m_c = cpp.frustum_masks_cpp(frame0_points, frame0_boxes, calib, shape, shrink=0.1)
    np.testing.assert_array_equal(m_c, frustum_ref(frame0_points, shrunk, calib, shape))


def test_frustum_no_boxes(frame0_points, calib):
    m = cpp.frustum_masks_cpp(frame0_points, np.zeros((0, 4)), calib, calib.image_shape)
    assert m.shape == (0, len(frame0_points))


@pytest.mark.skipif(pf is None or not hasattr(pf, "frustum_mask"), reason="팀 E point_filters.frustum_mask 없음")
@pytest.mark.parametrize("shrink", [0.0, 0.1])
def test_frustum_matches_team_e(frame0_points, frame0_boxes, calib, shrink):
    shape = calib.image_shape
    m_c = cpp.frustum_masks_cpp(frame0_points, frame0_boxes, calib, shape, shrink=shrink)
    for k, box in enumerate(frame0_boxes):
        m_e = pf.frustum_mask(frame0_points, box, calib, shape, shrink)
        np.testing.assert_array_equal(m_c[k], m_e, err_msg=f"box {k} shrink={shrink}")


# ---- 팀 E 회신 반영 케이스 ---------------------------------------------------------------------------
def _synthetic_calib() -> KittiCalibration:
    """tests/test_fusion.py 의 synthetic_calib 와 동일: 축 치환만 있는 velo→cam + 단순 핀홀 (f=700, c=(621, 187.5))."""
    from perceptrack3d.geometry.transforms import make_rigid

    R = np.array([[0.0, -1.0, 0.0], [0.0, 0.0, -1.0], [1.0, 0.0, 0.0]])
    P = np.array([[700.0, 0.0, 621.0, 0.0], [0.0, 700.0, 187.5, 0.0], [0.0, 0.0, 1.0, 0.0]])
    return KittiCalibration(T_velo_to_cam=make_rigid(R, np.zeros(3)), R_rect_00=np.eye(4), P_rect_02=P, image_size=(1242, 375))


def test_frustum_edge_point_shrink_rule():
    """팀 E test_frustum_mask_includes_center_point_excludes_outside 와 같은 케이스. 경계 점 (20, 1.30, 0) 은 shrink 0 포함, 0.1 제외."""
    calib = _synthetic_calib()
    pts = np.array([[20.0, 0.0, 0.0], [20.0, 5.0, 0.0], [-20.0, 0.0, 0.0], [20.0, 0.0, 10.0]])
    box = np.array([[571, 137, 671, 237]], dtype=np.float64)
    np.testing.assert_array_equal(cpp.frustum_masks_cpp(pts, box, calib, calib.image_shape, shrink=0.1)[0], [True, False, False, False])
    edge = np.array([[20.0, 1.30, 0.0]])
    assert cpp.frustum_masks_cpp(edge, box, calib, calib.image_shape, shrink=0.0)[0, 0]
    assert not cpp.frustum_masks_cpp(edge, box, calib, calib.image_shape, shrink=0.1)[0, 0]
    if pf is not None and hasattr(pf, "frustum_mask"):
        for s in (0.0, 0.1):
            np.testing.assert_array_equal(cpp.frustum_masks_cpp(edge, box, calib, calib.image_shape, shrink=s)[0],
                                          pf.frustum_mask(edge, box[0], calib, calib.image_shape, s))


def test_shrink_rejects_out_of_range():
    for s in (-0.1, 0.5, 1.0):
        with pytest.raises(ValueError):
            cpp.shrink_boxes(np.array([[0, 0, 10, 10]]), s)


@pytest.mark.skipif(pf is None or not hasattr(pf, "roi_filter"), reason="팀 E point_filters.roi_filter 없음")
def test_roi_nonrepresentable_thresholds_match_team_e(frame0_points):
    """0.1, 30.3 처럼 float32 로 정확히 표현되지 않는 임계값에서도 팀 E (float64 비교) 와 완전 일치해야 한다."""
    roi = {"x_min": 0.1, "x_max": 30.3, "y_abs_max": 10.1, "z_min": -1.7, "z_max": 0.3}
    m_c = cpp.roi_mask_cpp(frame0_points, roi)
    m_e = pf.roi_filter(frame0_points, roi)
    np.testing.assert_array_equal(m_c, m_e)
    assert 0 < m_c.sum() < len(m_c)
