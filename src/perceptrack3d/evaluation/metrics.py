"""Phase 8 평가 메트릭 라이브러리. 모두 순수 함수이며 입력은 numpy 배열 또는 types 의 dataclass 이다.

좌표계: 모든 중심/박스는 Velodyne (x 전방, y 좌, z 상, m). "BEV 거리" 는 (x, y) 평면의 유클리드 거리,
"깊이(depth)" 는 Velodyne x 축(전방) 값이다.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment

from perceptrack3d.evaluation.tracklets import box_corners_bev

# 게이트 밖 쌍에 넣는 큰 비용. np.inf 는 행 전체가 inf 일 때 linear_sum_assignment 가 실패하므로 유한값을 쓴다.
_BIG_COST = 1e6


def _as_centers(items) -> np.ndarray:
    """(N,3) 배열 또는 Object3D/Track/GtBox3D 리스트를 (N,3) float64 로 통일한다."""
    if isinstance(items, np.ndarray):
        arr = items.astype(np.float64)
    else:
        rows = []
        for it in items:
            c = it.position if hasattr(it, "position") else it.center   # Track 은 position, 나머지는 center
            if c is None:
                raise ValueError("center 가 None 인 객체가 있습니다. status != 'ok' 인 예측은 먼저 걸러 주세요")
            rows.append(np.asarray(c, dtype=np.float64))
        arr = np.array(rows, dtype=np.float64).reshape(-1, 3)
    assert arr.ndim == 2 and arr.shape[1] == 3, f"centers 는 (N, 3) 이어야 합니다: {arr.shape}"
    return arr


def match_by_center_distance(pred_centers, gt_centers, max_dist: float) -> list[tuple[int, int, float]]:
    """BEV(x, y) 중심 거리로 예측 ↔ GT 를 1:1 매칭한다 (헝가리안).

    Args:
        pred_centers: (P, 3) 배열 또는 Object3D/Track 리스트 (center 가 None 이면 ValueError)
        gt_centers:   (G, 3) 배열 또는 GtBox3D 리스트
        max_dist:     BEV 거리 임계값 (m). 이보다 먼 쌍은 매칭하지 않는다.
    Returns: [(pred_idx, gt_idx, bev_dist), ...]. 거리 합이 최소가 되는 1:1 할당 중 임계값 이하만.
    """
    P, G = _as_centers(pred_centers), _as_centers(gt_centers)
    if len(P) == 0 or len(G) == 0:
        return []
    dist = np.linalg.norm(P[:, None, :2] - G[None, :, :2], axis=2)   # (P, G)
    cost = np.where(dist <= max_dist, dist, _BIG_COST)
    rows, cols = linear_sum_assignment(cost)
    return [(int(r), int(c), float(dist[r, c])) for r, c in zip(rows, cols) if dist[r, c] <= max_dist]


def _stats(values: np.ndarray) -> tuple[float, float, float]:
    return float(np.mean(values)), float(np.median(values)), float(np.percentile(values, 95))


def center_errors(matches: list[tuple[int, int, float]], pred_centers, gt_centers) -> dict:
    """매칭된 쌍의 중심 오차 통계.

    Returns: {"n", "mean_3d", "median_3d", "p95_3d", "mean_bev", "median_bev", "p95_bev"} (m).
        3d = ||p - g||, bev = ||p[:2] - g[:2]||. n == 0 이면 통계는 nan.
    """
    P, G = _as_centers(pred_centers), _as_centers(gt_centers)
    if not matches:
        return {"n": 0, **{f"{s}_{k}": float("nan") for k in ("3d", "bev") for s in ("mean", "median", "p95")}}
    pi = np.array([m[0] for m in matches]); gi = np.array([m[1] for m in matches])
    diff = P[pi] - G[gi]
    d3 = np.linalg.norm(diff, axis=1)
    dbev = np.linalg.norm(diff[:, :2], axis=1)
    m3, md3, p3 = _stats(d3)
    mb, mdb, pb = _stats(dbev)
    return {"n": len(matches), "mean_3d": m3, "median_3d": md3, "p95_3d": p3,
            "mean_bev": mb, "median_bev": mdb, "p95_bev": pb}


def depth_errors(matches: list[tuple[int, int, float]], pred_centers, gt_centers) -> dict:
    """깊이(Velodyne x, 전방) 오차 통계.

    Returns: {"n", "mean_abs", "median_abs", "p95_abs", "bias"} (m).
        err = pred_x - gt_x. bias = mean(err) (부호 있음, 양수면 예측이 GT 보다 멀리 있음).
    """
    P, G = _as_centers(pred_centers), _as_centers(gt_centers)
    if not matches:
        return {"n": 0, "mean_abs": float("nan"), "median_abs": float("nan"), "p95_abs": float("nan"), "bias": float("nan")}
    pi = np.array([m[0] for m in matches]); gi = np.array([m[1] for m in matches])
    err = P[pi, 0] - G[gi, 0]
    mean_abs, median_abs, p95_abs = _stats(np.abs(err))
    return {"n": len(matches), "mean_abs": mean_abs, "median_abs": median_abs, "p95_abs": p95_abs,
            "bias": float(np.mean(err))}


def polygon_area(poly: np.ndarray) -> float:
    """(K, 2) 다각형의 넓이 (신발끈 공식, 절댓값)."""
    x, y = poly[:, 0], poly[:, 1]
    return 0.5 * abs(float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def convex_intersection(subject: np.ndarray, clip: np.ndarray) -> np.ndarray:
    """Sutherland–Hodgman 클리핑: 볼록 다각형 clip 으로 subject 를 잘라 교집합 다각형을 돌려준다.

    Args:
        subject, clip: (K, 2). 두 다각형 모두 볼록해야 하며 꼭짓점 방향(시계/반시계)은 무관하다.
    Returns: (M, 2) 교집합 다각형. 교집합이 없으면 (0, 2).
    알고리즘: clip 의 각 변(반평면)에 대해 subject 의 변을 순회하며 안쪽 점은 유지, 경계를 넘는 변은 교점을 추가한다.
    """
    clip = np.asarray(clip, dtype=np.float64)
    # clip 을 반시계(CCW) 로 통일해야 "왼쪽 = 안쪽" 판정이 성립한다.
    signed = 0.5 * (np.dot(clip[:, 0], np.roll(clip[:, 1], -1)) - np.dot(clip[:, 1], np.roll(clip[:, 0], -1)))
    if signed < 0:
        clip = clip[::-1]
    output = [tuple(p) for p in np.asarray(subject, dtype=np.float64)]
    for i in range(len(clip)):
        a, b = clip[i], clip[(i + 1) % len(clip)]
        edge = b - a
        if not output:
            break
        inputs, output = output, []

        def inside(p):
            return edge[0] * (p[1] - a[1]) - edge[1] * (p[0] - a[0]) >= 0.0

        def intersect(p, q):
            # 선분 p→q 와 직선 a→b 의 교점
            r, s = q[0] - p[0], q[1] - p[1]
            denom = edge[0] * s - edge[1] * r
            t = (edge[0] * (a[1] - p[1]) - edge[1] * (a[0] - p[0])) / denom
            return (p[0] + t * r, p[1] + t * s)

        prev = inputs[-1]
        for cur in inputs:
            if inside(cur):
                if not inside(prev):
                    output.append(intersect(prev, cur))
                output.append(cur)
            elif inside(prev):
                output.append(intersect(prev, cur))
            prev = cur
    return np.array(output, dtype=np.float64).reshape(-1, 2)


def _bev_corners(box) -> np.ndarray:
    """GtBox3D/Object3D (center, size, yaw 속성) 또는 (center, size, yaw) 튜플 → (4, 2) 모서리."""
    if hasattr(box, "center"):
        yaw = 0.0 if box.yaw is None else float(box.yaw)
        return box_corners_bev(box.center, box.size, yaw)
    center, size, yaw = box
    return box_corners_bev(center, size, float(yaw))


def bev_iou(box_a, box_b) -> float:
    """두 회전 사각형의 BEV IoU.

    Args: 각 박스는 GtBox3D/Object3D 처럼 center(≥2,), size(≥2, [l, w, ...]), yaw 속성을 가진 객체이거나
        (center, size, yaw) 튜플. Object3D.yaw 가 None 이면 0 (AABB) 으로 본다.
    Returns: 교집합 / 합집합 ∈ [0, 1]. 두 박스 넓이가 모두 0 이면 0.
    구현 선택: 외부 의존성(shapely) 없이 Sutherland–Hodgman 클리핑(convex_intersection)으로 교집합 넓이를 구한다.
        사각형은 항상 볼록하므로 정확하며, 의존성을 늘리지 않아 재현성이 좋고 학습 목적상 알고리즘이 드러난다.
    """
    ca, cb = _bev_corners(box_a), _bev_corners(box_b)
    area_a, area_b = polygon_area(ca), polygon_area(cb)
    inter_poly = convex_intersection(ca, cb)
    inter = polygon_area(inter_poly) if len(inter_poly) >= 3 else 0.0
    union = area_a + area_b - inter
    return float(inter / union) if union > 0 else 0.0


def precision_recall(n_matched: int, n_pred: int, n_gt: int) -> dict:
    """precision = TP/n_pred, recall = TP/n_gt, f1 = 2PR/(P+R). 분모가 0 이면 0.0."""
    if n_matched > n_pred or n_matched > n_gt:
        raise ValueError(f"n_matched({n_matched}) 가 n_pred({n_pred}) 또는 n_gt({n_gt}) 보다 큽니다")
    precision = n_matched / n_pred if n_pred else 0.0
    recall = n_matched / n_gt if n_gt else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


def count_id_switches(assignments: list[list[tuple[int, int]]]) -> int:
    """ID 스위치 수. assignments[f] = 프레임 f 의 [(track_id, gt_tracklet_id), ...] (매칭된 쌍만).

    같은 GT 가 **직전에 매칭됐던** 트랙과 다른 트랙에 매칭되면 1회로 센다 (CLEAR MOT 의 IDSW 정의; 프레임이 연속하지
    않아도 마지막 매칭 트랙과 비교한다).
    """
    last_track: dict[int, int] = {}
    n = 0
    for frame in assignments:
        for track_id, gt_id in frame:
            if gt_id in last_track and last_track[gt_id] != track_id:
                n += 1
            last_track[gt_id] = track_id
    return n


def track_fragmentation(assignments: list[list[tuple[int, int]]]) -> int:
    """단절(fragmentation) 수. GT 가 매칭된 프레임 사이에 매칭되지 않은 구간이 생길 때마다 1회.

    예: GT 7 이 프레임 [0, 1, 2, 5, 6] 에서 매칭 → 2 와 5 사이 공백 1개 → 1. GT 는 그 사이에도 존재한다고 가정한다.
    """
    frames_by_gt: dict[int, set[int]] = {}
    for f, frame in enumerate(assignments):
        for _, gt_id in frame:
            frames_by_gt.setdefault(gt_id, set()).add(f)
    n = 0
    for frames in frames_by_gt.values():
        seq = sorted(frames)
        n += sum(1 for a, b in zip(seq, seq[1:]) if b - a > 1)
    return n


def summarize_runtime(stage_times: dict[str, list[float]]) -> dict:
    """단계별 실행 시간 요약. 입력 단위는 **밀리초(ms)** 이다.

    Returns: {stage: {"n", "mean_ms", "median_ms", "p95_ms", "fps"}}. fps = 1000 / mean_ms. 비어 있는 단계는 건너뛴다.
    """
    out = {}
    for stage, times in stage_times.items():
        t = np.asarray(times, dtype=np.float64)
        if t.size == 0:
            continue
        mean_ms, median_ms, p95_ms = _stats(t)
        out[stage] = {"n": int(t.size), "mean_ms": mean_ms, "median_ms": median_ms, "p95_ms": p95_ms,
                      "fps": 1000.0 / mean_ms if mean_ms > 0 else float("inf")}
    return out
