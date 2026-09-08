"""좌표계에 독립적인 순수 기하 변환 함수 (Phase 2).

규약
- 점 배열: (N, 3) — 행 하나가 점 하나 [x, y, z]. 단위 m.
- 강체 변환(rigid transform): (4, 4) 동차 행렬
      T = [[R, t],
           [0, 1]]      R: (3, 3) 회전(직교, det = +1), t: (3,) 이동
  p' = R p + t 를 동차 좌표로 쓰면  [p'; 1] = T @ [p; 1]  이 된다.
- 계산은 모두 float64 로 한다 (float32 LiDAR 점을 넣어도 내부에서 승격). 반환도 float64.
"""
from __future__ import annotations

import numpy as np


def _check_points(pts: np.ndarray) -> np.ndarray:
    pts = np.asarray(pts, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError(f"점 배열은 (N, 3) 이어야 합니다. 받은 shape: {pts.shape}")
    return pts


def _check_transform(T: np.ndarray) -> np.ndarray:
    T = np.asarray(T, dtype=np.float64)
    if T.shape != (4, 4):
        raise ValueError(f"변환 행렬은 (4, 4) 이어야 합니다. 받은 shape: {T.shape}")
    return T


def to_homogeneous(pts: np.ndarray) -> np.ndarray:
    """(N, 3) → (N, 4). 각 점 뒤에 1 을 붙여 동차 좌표(homogeneous coordinates)로 만든다.

    동차 좌표를 쓰면 회전(행렬 곱)과 이동(덧셈)을 4x4 행렬 곱 하나로 표현할 수 있다.
    """
    pts = _check_points(pts)
    return np.hstack([pts, np.ones((pts.shape[0], 1), dtype=np.float64)])


def apply_transform(T: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """(4, 4) 강체 변환 T 를 (N, 3) 점들에 적용해 (N, 3) float64 를 돌려준다.

    수식:  p' = R p + t   ⇔   [p'; 1] = T @ [p; 1]
    N 개 점을 한 번에 처리하려고 행 벡터 형태로 (pts_h @ T.T) 를 계산한다.
    """
    T = _check_transform(T)
    pts_h = to_homogeneous(pts)
    return (pts_h @ T.T)[:, :3]


def invert_rigid(T: np.ndarray) -> np.ndarray:
    """강체 변환의 역행렬. np.linalg.inv 대신 회전의 직교성을 이용한다.

    T = [[R, t], [0, 1]]  이면  T⁻¹ = [[Rᵀ, -Rᵀ t], [0, 1]]
    (R 이 직교이므로 R⁻¹ = Rᵀ). 수치적으로 안정적이고 "회전을 되돌리고 이동을 빼는" 의미가 드러난다.
    전제: R 이 정확히 직교여야 한다. 텍스트 파일에서 읽은 회전은 nearest_rotation() 으로 먼저 보정한다.
    """
    T = _check_transform(T)
    R, t = T[:3, :3], T[:3, 3]
    T_inv = np.eye(4, dtype=np.float64)
    T_inv[:3, :3] = R.T
    T_inv[:3, 3] = -R.T @ t
    return T_inv


def nearest_rotation(R: np.ndarray) -> np.ndarray:
    """(3, 3) 행렬을 가장 가까운 회전 행렬로 투영한다 (SVD: R = U Σ Vᵀ → U Vᵀ, det = +1 보정).

    왜 필요한가: 캘리브레이션 텍스트 파일의 회전은 유효숫자 7자리로 반올림되어 R Rᵀ ≠ I (오차 ~1e-7) 이다.
    그대로 쓰면 Rᵀ 로 만든 역변환의 round-trip 오차가 먼 점(70 m)에서 ~2e-6 m 가 되어
    "정확한 역변환" 이라는 강체 변환의 성질이 깨진다. 보정 후 값 변화는 1e-7 수준이라 투영에는 영향이 없다.
    """
    R = np.asarray(R, dtype=np.float64)
    if R.shape != (3, 3):
        raise ValueError(f"회전 행렬은 (3, 3) 이어야 합니다. 받은 shape: {R.shape}")
    U, _, Vt = np.linalg.svd(R)
    R_o = U @ Vt
    if np.linalg.det(R_o) < 0:  # 반사(reflection)가 나오면 마지막 축을 뒤집어 회전으로 만든다
        U[:, -1] *= -1
        R_o = U @ Vt
    return R_o


def make_rigid(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """(3, 3) 회전과 (3,) 이동으로 (4, 4) 동차 강체 변환을 만든다."""
    R = np.asarray(R, dtype=np.float64)
    t = np.asarray(t, dtype=np.float64).reshape(-1)
    if R.shape != (3, 3) or t.shape != (3,):
        raise ValueError(f"R 은 (3, 3), t 는 (3,) 이어야 합니다. 받은 shape: {R.shape}, {t.shape}")
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = t
    return T
