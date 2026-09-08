"""투영된 LiDAR 점을 이미지 위에 그리는 유틸 (Phase 3).

색은 깊이(카메라 앞 거리, m)에 따라 순차형 컬러맵으로 칠한다. 프레임마다 색 의미가 같도록
깊이 범위를 고정(d_min, d_max)해서 정규화한다.
"""
from __future__ import annotations

import cv2
import matplotlib
import numpy as np

DEFAULT_DEPTH_RANGE = (0.0, 70.0)   # m. 이 범위 밖은 양 끝 색으로 포화


def depth_to_color(
    depth: np.ndarray,
    d_min: float = DEFAULT_DEPTH_RANGE[0],
    d_max: float = DEFAULT_DEPTH_RANGE[1],
    cmap: str = "plasma_r",
) -> np.ndarray:
    """(M,) 깊이(m) → (M, 3) uint8 **BGR** 색. 가까울수록 밝은 노랑, 멀수록 어두운 보라(plasma 역순).

    OpenCV 로 그릴 것이므로 BGR 순서로 돌려준다.
    """
    depth = np.asarray(depth, dtype=np.float64).reshape(-1)
    t = np.clip((depth - d_min) / max(d_max - d_min, 1e-9), 0.0, 1.0)
    rgba = matplotlib.colormaps[cmap](t)                    # (M, 4) float 0~1, RGBA
    rgb = (rgba[:, :3] * 255).astype(np.uint8)
    return rgb[:, ::-1].copy()                               # RGB → BGR


def draw_projected_points(
    image_bgr: np.ndarray,
    uv: np.ndarray,
    depth: np.ndarray,
    radius: int = 1,
    d_min: float = DEFAULT_DEPTH_RANGE[0],
    d_max: float = DEFAULT_DEPTH_RANGE[1],
) -> np.ndarray:
    """(H, W, 3) BGR 이미지 복사본 위에 (M, 2) 픽셀 uv 를 깊이 색으로 찍어 돌려준다. 원본은 바꾸지 않는다.

    uv 는 float 이어도 되며, 내부에서 내림(int) 하여 픽셀 좌표로 쓴다.
    """
    if image_bgr.ndim != 3 or image_bgr.shape[2] != 3:
        raise ValueError(f"image_bgr 는 (H, W, 3) 이어야 합니다: {image_bgr.shape}")
    uv = np.asarray(uv)
    if uv.ndim != 2 or uv.shape[1] != 2 or len(uv) != len(depth):
        raise ValueError(f"uv 는 (M, 2), depth 는 (M,) 이어야 합니다: {uv.shape}, {np.shape(depth)}")
    out = image_bgr.copy()
    colors = depth_to_color(depth, d_min, d_max)
    H, W = out.shape[:2]
    for (u, v), c in zip(uv.astype(int), colors):
        if 0 <= u < W and 0 <= v < H:
            cv2.circle(out, (int(u), int(v)), radius, (int(c[0]), int(c[1]), int(c[2])), -1)
    return out
