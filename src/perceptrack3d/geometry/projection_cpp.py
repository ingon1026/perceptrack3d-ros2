"""C++ 확장 모듈(pt3d_cpp) 의 얇은 파이썬 래퍼 (Phase 10).

Python 기준 구현(geometry/projection.py, fusion/point_filters.py)과 같은 계약을 유지하되 내부는 C++ 로 계산한다.
확장 모듈은 scripts/build_cpp.sh 로 빌드하며 cpp/build/ 에 놓인다. 이 파일은 그 경로를 sys.path 에 추가해 불러온다.

zero-copy 규칙
- 로더가 주는 (N, 4) float32 C-연속 배열을 **자르지 않고** 그대로 넘긴다. pts[:, :3] 은 비연속 뷰라 확장 모듈이
  복사하게 되지만, (N, 4) 전체를 넘기면 C++ 쪽이 행 간격(stride) 4 로 앞 3열만 읽는다.
- float64 나 비연속 배열을 넘기면 확장 모듈이 float32 C-연속으로 변환(복사)한 뒤 계산한다. 동작은 같고 느릴 뿐이다.
"""
from __future__ import annotations

import sys

import numpy as np

from perceptrack3d.config import REPO_ROOT
from perceptrack3d.geometry.calibration import KittiCalibration

CPP_BUILD_DIR = REPO_ROOT / "cpp" / "build"
if CPP_BUILD_DIR.is_dir() and str(CPP_BUILD_DIR) not in sys.path:
    sys.path.append(str(CPP_BUILD_DIR))

try:
    import pt3d_cpp as _cpp
except ImportError as e:  # 빌드 안 됨, 파이썬 버전 불일치 등
    raise ImportError(
        "C++ 확장 모듈 pt3d_cpp 를 불러올 수 없습니다. 먼저 빌드하세요:\n"
        "    scripts/build_cpp.sh\n"
        f"(기대 위치: {CPP_BUILD_DIR}/pt3d_cpp.cpython-*.so, 인터프리터: {sys.executable})\n"
        f"원인: {e}"
    ) from e


def _as_points(points_velo: np.ndarray) -> np.ndarray:
    pts = np.asarray(points_velo)
    if pts.ndim != 2 or pts.shape[1] not in (3, 4):
        raise ValueError(f"points 는 (N, 3) 또는 (N, 4) 이어야 합니다. 받은 shape: {pts.shape}")
    return pts


def project_velo_to_image_cpp(
    pts_velo: np.ndarray,
    calib: KittiCalibration,
    image_shape: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """geometry.projection.project_velo_to_image 와 같은 입력·출력 (uv (M,2) float32, depth (M,) float32, mask (N,) bool)."""
    if len(image_shape) < 2:
        raise ValueError(f"image_shape 은 (H, W) 이어야 합니다: {image_shape}")
    H, W = int(image_shape[0]), int(image_shape[1])
    uv, depth, mask = _cpp.project_velo_to_image(_as_points(pts_velo), calib.P_velo_to_img, H, W)
    return uv, depth, mask.view(np.bool_)  # uint8 0/1 → bool 재해석 (복사 없음)


def roi_mask_cpp(points_velo: np.ndarray, roi_cfg: dict) -> np.ndarray:
    """fusion.point_filters.roi_filter 와 같은 규칙: x_min<=x<=x_max, |y|<=y_abs_max, z_min<=z<=z_max. 반환 (N,) bool."""
    mask = _cpp.roi_mask(
        _as_points(points_velo),
        float(roi_cfg["x_min"]), float(roi_cfg["x_max"]), float(roi_cfg["y_abs_max"]),
        float(roi_cfg["z_min"]), float(roi_cfg["z_max"]),
    )
    return mask.view(np.bool_)


def shrink_boxes(boxes_xyxy: np.ndarray, shrink: float) -> np.ndarray:
    """(K, 4) 박스의 각 변을 안쪽으로 (너비·높이 × shrink) 만큼 옮긴다. shrink=0 이면 그대로. float64 반환.

    팀 E point_filters.shrink_box 와 같은 수식·범위 검사 (shrink ∈ [0, 0.5), 아니면 ValueError).
    """
    if not 0.0 <= shrink < 0.5:
        raise ValueError(f"shrink 는 [0, 0.5) 이어야 합니다: {shrink}")
    b = np.asarray(boxes_xyxy, dtype=np.float64).reshape(-1, 4).copy()
    if shrink:
        w = (b[:, 2] - b[:, 0]) * shrink
        h = (b[:, 3] - b[:, 1]) * shrink
        b[:, 0] += w
        b[:, 2] -= w
        b[:, 1] += h
        b[:, 3] -= h
    return b


def frustum_masks_cpp(
    points_velo: np.ndarray,
    boxes_xyxy: np.ndarray,
    calib: KittiCalibration,
    image_shape: tuple[int, int],
    shrink: float = 0.0,
) -> np.ndarray:
    """K 개 2D 박스 각각에 대한 frustum 마스크 (K, N) bool. 투영은 C++ 안에서 한 번만 수행한다.

    점 i 가 박스 k 안: project_velo_to_image 의 mask 가 True 이고, 축소된 박스에 대해 x1<=u<=x2, y1<=v<=y2 (포함 경계).
    """
    H, W = int(image_shape[0]), int(image_shape[1])
    boxes = shrink_boxes(boxes_xyxy, shrink)
    masks = _cpp.frustum_masks(_as_points(points_velo), calib.P_velo_to_img, H, W, boxes)
    return masks.view(np.bool_)
