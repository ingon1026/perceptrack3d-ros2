"""트랙 ↔ 관측 데이터 연관(association): 비용 행렬 + 게이팅 + 헝가리안 할당.

좌표계: Velodyne (m). 트랙의 예측 위치 H x⁻ 와 관측 z 사이의 거리를 비용으로 쓴다.

비용 정의
    mahalanobis : d² = (z − H x⁻)ᵀ S⁻¹ (z − H x⁻),  S = H P⁻ Hᵀ + R   (chi² 3 자유도 분포)
    euclidean   : d  = ‖z − H x⁻‖₂
게이트를 넘는 쌍은 np.inf. 헝가리안(scipy.optimize.linear_sum_assignment)은 inf 를 못 다루므로
큰 유한값으로 바꿔 풀고, 결과 중 원래 비용이 inf(또는 gate 초과)인 쌍은 사후에 제거한다.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment

# 관측 행렬 H = [I3 | 0]: 상태 [px, py, pz, vx, vy, vz] 에서 위치만 관측한다.
H_POS = np.hstack([np.eye(3), np.zeros((3, 3))])


def build_cost_matrix(
    tracks: list,
    objects: list,
    gating: str = "mahalanobis",
    gate_mahalanobis: float = 9.21,
    gate_euclidean: float = 3.0,
    R: np.ndarray | None = None,
    match_same_class: bool = False,
) -> np.ndarray:
    """(T, D) 비용 행렬. tracks[i] 와 objects[j] 의 거리. 게이트 밖·클래스 불일치는 np.inf.

    tracks  : Track 리스트 (state (6,), covariance (6,6) 는 이미 predict 된 값이어야 한다)
    objects : center (3,) 가 있는 Object3D 리스트
    gating  : "mahalanobis" (R (3,3) 필수, 비용 = d²) | "euclidean" (비용 = d, m)
    """
    T, D = len(tracks), len(objects)
    cost = np.full((T, D), np.inf, dtype=float)
    if T == 0 or D == 0:
        return cost
    if gating not in ("mahalanobis", "euclidean"):
        raise ValueError(f"gating 은 'mahalanobis' 또는 'euclidean' 이어야 합니다: {gating!r}")
    if gating == "mahalanobis" and R is None:
        raise ValueError("gating='mahalanobis' 에는 관측 잡음 R (3,3) 이 필요합니다")

    Z = np.stack([np.asarray(o.center, dtype=float) for o in objects])   # (D, 3)
    assert Z.shape == (D, 3), Z.shape

    for i, tr in enumerate(tracks):
        x, P = tr.state, tr.covariance
        diff = Z - (H_POS @ x)                                            # (D, 3) 혁신 z − H x⁻
        if gating == "mahalanobis":
            S = H_POS @ P @ H_POS.T + R                                   # (3, 3)
            S_inv = np.linalg.inv(S)
            d = np.einsum("dj,jk,dk->d", diff, S_inv, diff)               # (D,)  d² = diffᵀ S⁻¹ diff
            gate = gate_mahalanobis
        else:
            d = np.linalg.norm(diff, axis=1)                              # (D,)  유클리드 거리 (m)
            gate = gate_euclidean
        row = np.where(d <= gate, d, np.inf)
        if match_same_class:
            same = np.array([o.class_name == tr.class_name for o in objects])
            row = np.where(same, row, np.inf)
        cost[i] = row
    return cost


def hungarian_assign(
    cost: np.ndarray, gate: float | None = None
) -> tuple[list[tuple[int, int]], list[int], list[int]]:
    """총비용 최소 일대일 할당. 반환 (matches, unmatched_tracks, unmatched_objects).

    cost : (T, D). np.inf 인 쌍은 절대 매칭되지 않는다. gate 를 주면 cost > gate 인 쌍도 제거한다.
    matches 는 (track_idx, object_idx) 리스트, 미매칭 인덱스는 오름차순.
    """
    cost = np.asarray(cost, dtype=float)
    assert cost.ndim == 2, cost.shape
    T, D = cost.shape
    if T == 0 or D == 0:
        return [], list(range(T)), list(range(D))

    finite = np.isfinite(cost)
    # inf 를 "어떤 유한 비용의 합보다도 큰" 값으로 치환 → 솔버는 가능한 한 inf 쌍을 피한다.
    big = max(1e6, 10.0 * float(cost[finite].sum())) if finite.any() else 1e6
    safe = np.where(finite, cost, big)
    rows, cols = linear_sum_assignment(safe)

    matches: list[tuple[int, int]] = []
    for i, j in zip(rows.tolist(), cols.tolist()):
        if not finite[i, j]:
            continue
        if gate is not None and cost[i, j] > gate:
            continue
        matches.append((i, j))

    matched_t = {i for i, _ in matches}
    matched_o = {j for _, j in matches}
    unmatched_tracks = [i for i in range(T) if i not in matched_t]
    unmatched_objects = [j for j in range(D) if j not in matched_o]
    return matches, unmatched_tracks, unmatched_objects
