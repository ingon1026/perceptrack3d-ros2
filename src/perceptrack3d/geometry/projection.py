"""LiDAR 점 → image_02 픽셀 투영 (Phase 3).

변환 체인
    Velodyne 점 (x, y, z)
      → 동차화 [x, y, z, 1]
      → P_velo_to_img (3x4) 곱  = P_rect_02 @ R_rect_00 @ T_velo_to_cam @ [x, y, z, 1]
      → [u·d, v·d, d]         (d = 카메라 앞 깊이, m)
      → d > 0 인 점만 남김      (카메라 뒤 점은 나누면 뒤집힌 가짜 픽셀이 되므로 반드시 먼저 제거)
      → (u, v) = (u·d / d, v·d / d)
      → 0 ≤ u < W, 0 ≤ v < H 인 점만 남김
"""
from __future__ import annotations

import numpy as np

from perceptrack3d.geometry.calibration import KittiCalibration
from perceptrack3d.geometry.transforms import to_homogeneous


def project_velo_to_image(
    pts_velo: np.ndarray,
    calib: KittiCalibration,
    image_shape: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Velodyne 점을 image_02 픽셀로 투영한다.

    Args:
        pts_velo: (N, 3) 또는 (N, 4) float. 열 = [x, y, z(, reflectance)], Velodyne 프레임, m.
                  4열이면 reflectance 는 무시한다.
        calib: KittiCalibration (P_velo_to_img 사용).
        image_shape: (H, W). `image.shape[:2]` 를 그대로 넣는다. (W, H) 순서가 아님에 주의.

    Returns:
        uv:    (M, 2) float32 픽셀 좌표 (u, v). u ∈ [0, W), v ∈ [0, H). 정수화하려면 astype(int) 로 내림.
        depth: (M,) float32. 투영 동차 좌표의 세 번째 성분 = rect 프레임 z + P_rect_02[2, 3] (≈ 2.7 mm 차이).
               카메라 2 광학축 방향의 깊이(m), 항상 > 0.
        mask:  (N,) bool. 원본 점 중 카메라 앞이면서 이미지 안에 들어온 점이 True. mask.sum() == M.
               uv[i] 는 pts_velo[mask][i] 에 대응한다.
    """
    pts = np.asarray(pts_velo)
    if pts.ndim != 2 or pts.shape[1] not in (3, 4):
        raise ValueError(f"pts_velo 는 (N, 3) 또는 (N, 4) 이어야 합니다. 받은 shape: {pts.shape}")
    if len(image_shape) < 2:
        raise ValueError(f"image_shape 은 (H, W) 이어야 합니다: {image_shape}")
    H, W = int(image_shape[0]), int(image_shape[1])

    pts_h = to_homogeneous(pts[:, :3])                       # (N, 4) float64
    proj = pts_h @ calib.P_velo_to_img.T                      # (N, 3) = [u·d, v·d, d]
    d = proj[:, 2]

    in_front = d > 0                                          # 1) 카메라 앞
    u = np.full(len(d), np.nan)
    v = np.full(len(d), np.nan)
    u[in_front] = proj[in_front, 0] / d[in_front]             # 2) 깊이로 나누기 (앞쪽 점만)
    v[in_front] = proj[in_front, 1] / d[in_front]

    in_image = in_front & (u >= 0) & (u < W) & (v >= 0) & (v < H)   # 3) 이미지 경계
    uv = np.stack([u[in_image], v[in_image]], axis=1).astype(np.float32)
    depth = d[in_image].astype(np.float32)
    return uv, depth, in_image
