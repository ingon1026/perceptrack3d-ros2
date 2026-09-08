"""LiDAR 점 필터·클러스터·박스 추정 순수 함수 모음 (Phase 5–6).

모든 입력 점은 **Velodyne 좌표계** (x 전방, y 좌, z 상, 단위 m) 이며 (N, 3) 또는 (N, 4) 배열이다.
(N, 4) 이면 마지막 열(reflectance) 은 무시한다. 반환 mask 는 항상 입력과 같은 길이 N 의 bool 배열이라
`points[mask]` 로 바로 걸러 쓸 수 있다. 함수는 상태를 갖지 않는다 (같은 입력 → 같은 출력).

Phase 6 처리 순서 (frustum_fusion.fuse_frame_clustered)
    ROI 필터 → 지면 제거 → 2D 박스 프러스텀 → DBSCAN → 클러스터 선택 → 중심/AABB/OBB
"""
from __future__ import annotations

import math

import numpy as np
import open3d as o3d

from perceptrack3d.geometry.calibration import KittiCalibration
from perceptrack3d.geometry.projection import project_velo_to_image

# RANSAC 의 무작위 표본 추출 시드. numpy 구현은 이 값으로 완전히 결정적이고, Open3D 경로는 호출 전에 같은 값으로 시드를 건다.
RANSAC_SEED = 0


def _xyz(points: np.ndarray) -> np.ndarray:
    """(N, 3|4) → (N, 3) float64. 모듈 경계 shape 검증."""
    pts = np.asarray(points)
    if pts.ndim != 2 or pts.shape[1] not in (3, 4):
        raise ValueError(f"points 는 (N, 3) 또는 (N, 4) 이어야 합니다. 받은 shape: {pts.shape}")
    return pts[:, :3].astype(np.float64, copy=False)


# ----------------------------------------------------------------------------- ROI

def roi_filter(points_velo: np.ndarray, roi_cfg: dict) -> np.ndarray:
    """관심 영역(ROI) 안의 점만 True 인 (N,) bool mask.

    roi_cfg 키 (configs/kitti.yaml fusion.roi): x_min, x_max, y_abs_max, z_min, z_max (m, Velodyne).
    조건 (경계 포함): x_min <= x <= x_max, |y| <= y_abs_max, z_min <= z <= z_max.
    x_min = 0 은 "카메라 뒤(자차 뒤) 점 제거" 를 뜻한다. 카메라가 보는 곳은 x > 0 뿐이므로 뒤쪽 점은 융합에 필요 없다.
    """
    p = _xyz(points_velo)
    return (
        (p[:, 0] >= float(roi_cfg["x_min"])) & (p[:, 0] <= float(roi_cfg["x_max"]))
        & (np.abs(p[:, 1]) <= float(roi_cfg["y_abs_max"]))
        & (p[:, 2] >= float(roi_cfg["z_min"])) & (p[:, 2] <= float(roi_cfg["z_max"]))
    )


# ----------------------------------------------------------------------------- 지면 제거

def ransac_plane(points_xyz: np.ndarray, distance_threshold: float, num_iterations: int,
                 seed: int = RANSAC_SEED, chunk: int = 50, max_score_points: int = 12000) -> tuple[np.ndarray, np.ndarray]:
    """numpy 로 구현한 결정적(seeded) RANSAC 평면 적합. Returns (plane (4,) [a, b, c, d] 단위 법선, inlier mask (N,)).

    알고리즘 (Fischler & Bolles 1981)
        1. 점 3개를 무작위로 뽑아 평면 가설을 만든다: n = (b − a) × (c − a) / |·|,  d = −n·a
        2. 점의 평면 거리 |n·p + d| 가 distance_threshold 이하인 점(inlier) 수를 센다
        3. num_iterations 번 반복해 inlier 가 가장 많은 가설을 고른다
        4. 그 inlier 들로 최소제곱 평면(중심 이동 후 SVD 의 최소 특이벡터 = 법선) 을 다시 맞추고 inlier 를 다시 센다
    구현: 가설 chunk 개를 한 번에 행렬곱 (M, 3) @ (3, chunk) 으로 평가한다. 가설 점수는 seed 로 뽑은 부분집합
          (max_score_points 개, 기본 12 000) 으로 매기고, 최종 재적합·inlier 판정은 전체 점으로 한다 (N=60 000, 200 가설 → ~20 ms).
    Open3D segment_plane 대신 쓰는 이유: Open3D 0.19 는 시드를 고정해도 병렬 실행 때문에 실행마다 결과가 달라진다.
    """
    p = np.asarray(points_xyz, dtype=np.float64)
    N = len(p)
    rng = np.random.default_rng(seed)
    score = p if N <= max_score_points else p[rng.choice(N, max_score_points, replace=False)]
    score32 = score.astype(np.float32)
    best_count, best_plane = -1, None
    for start in range(0, num_iterations, chunk):
        k = min(chunk, num_iterations - start)
        idx = rng.integers(0, N, size=(k, 3))
        a, b, c = p[idx[:, 0]], p[idx[:, 1]], p[idx[:, 2]]
        nrm = np.cross(b - a, c - a)                                  # (k, 3) 법선 (정규화 전)
        length = np.linalg.norm(nrm, axis=1)
        ok = length > 1e-9                                            # 세 점이 한 직선 위면 평면이 정의되지 않음
        if not ok.any():
            continue
        nrm = nrm[ok] / length[ok, None]
        d = -np.einsum("ij,ij->i", nrm, a[ok])                        # (k',)
        dist = np.abs(score32 @ nrm.T.astype(np.float32) + d.astype(np.float32))   # (M, k') 점-평면 거리
        counts = (dist <= distance_threshold).sum(axis=0)
        j = int(np.argmax(counts))
        if counts[j] > best_count:
            best_count, best_plane = int(counts[j]), np.append(nrm[j], d[j])
    if best_plane is None:
        return np.full(4, np.nan), np.zeros(N, dtype=bool)
    # 최소제곱 재적합: inlier 점의 중심을 지나고, 퍼짐이 가장 작은 방향(최소 특이벡터) 이 법선
    inl = np.abs(p @ best_plane[:3] + best_plane[3]) <= distance_threshold
    q = p[inl]
    centroid = q.mean(axis=0)
    _, _, vt = np.linalg.svd(q - centroid, full_matrices=False)
    nrm = vt[-1]
    plane = np.append(nrm, -nrm @ centroid)
    inl = np.abs(p @ plane[:3] + plane[3]) <= distance_threshold
    return plane, inl


def remove_ground(points_velo: np.ndarray, ground_cfg: dict) -> tuple[np.ndarray, np.ndarray]:
    """RANSAC 평면 적합으로 지면 점을 찾아 제거한다.

    Args:
        points_velo: (N, 3|4) Velodyne 점. 보통 roi_filter 를 통과한 점을 넣는다 (멀리 있는 점·건물 벽이
                     평면 후보를 오염시키는 것을 줄이고 속도도 빨라진다).
        ground_cfg: fusion.ground — method ("ransac" = numpy 결정적 구현(기본), "open3d" = Open3D segment_plane),
                    distance_threshold (m, 평면에서 이 거리 안이면 inlier), ransac_n (한 번에 뽑는 점 수, 평면은 3),
                    num_iterations (반복 횟수), 선택 키 min_normal_z (기본 0.8).
    Returns:
        (non_ground_mask (N,) bool — 지면이 **아닌** 점이 True,
         plane (4,) float64 [a, b, c, d] — 평면식 a·x + b·y + c·z + d = 0, 법선 (a, b, c) 는 단위 벡터이고 c > 0 (위쪽) 으로 정규화)
    수식: 점 p 의 평면 거리 = |a·x + b·y + c·z + d| (법선이 단위 벡터일 때). 거리 <= distance_threshold 이면 inlier(지면).
    안전장치: 찾은 평면의 법선 z 성분이 min_normal_z 보다 작으면 (벽 같은 수직면을 잡은 경우) 지면을 제거하지 않고
             mask 전체 True, plane 은 nan 으로 돌려준다. 점이 ransac_n 보다 적어도 같다.
    KITTI 에서 Velodyne 은 지면에서 약 1.73 m 높이에 있으므로 z ≈ -1.73 부근에 평면이 나와야 정상이다.
    """
    p = _xyz(points_velo)
    n = len(p)
    ransac_n = int(ground_cfg.get("ransac_n", 3))
    min_normal_z = float(ground_cfg.get("min_normal_z", 0.8))
    method = str(ground_cfg.get("method", "ransac"))
    thr, n_iter = float(ground_cfg["distance_threshold"]), int(ground_cfg["num_iterations"])
    nan_plane = np.full(4, np.nan)
    if n < max(ransac_n, 3):
        return np.ones(n, dtype=bool), nan_plane

    if method == "open3d":
        o3d.utility.random.seed(RANSAC_SEED)          # 주의: 0.19 는 병렬 실행이라 시드를 고정해도 완전히 결정적이지 않다
        pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(p))
        plane, inliers = pcd.segment_plane(distance_threshold=thr, ransac_n=ransac_n, num_iterations=n_iter)
        plane = np.asarray(plane, dtype=np.float64).reshape(4)
        ground = np.zeros(n, dtype=bool)
        ground[np.asarray(inliers, dtype=np.int64)] = True
    elif method == "ransac":
        plane, ground = ransac_plane(p, thr, n_iter)
    else:
        raise ValueError(f"fusion.ground.method 는 'ransac' 또는 'open3d' 이어야 합니다: {method!r}")

    if np.isnan(plane).any():
        return np.ones(n, dtype=bool), nan_plane
    if plane[2] < 0:                       # 법선을 항상 위(+z) 로 향하게 통일 (a, b, c, d 모두 부호 반전)
        plane = -plane
    if plane[2] < min_normal_z:            # 수직에 가까운 면 → 지면이 아님
        return np.ones(n, dtype=bool), nan_plane
    return ~ground, plane


# ----------------------------------------------------------------------------- 프러스텀

def shrink_box(xyxy: np.ndarray, shrink: float) -> np.ndarray:
    """2D 박스를 각 변에서 shrink 비율만큼 안쪽으로 줄인다.

    (x1, y1, x2, y2) → 폭 w, 높이 h 에 대해 x1 + shrink·w, y1 + shrink·h, x2 - shrink·w, y2 - shrink·h.
    shrink = 0.1 이면 가로·세로 각각 80% 크기의 가운데 박스가 된다. 박스 가장자리에 걸리는 배경(도로·건물) 점을 줄이는 목적.
    """
    x1, y1, x2, y2 = (float(v) for v in np.asarray(xyxy).reshape(4))
    if not 0.0 <= shrink < 0.5:
        raise ValueError(f"shrink 는 [0, 0.5) 이어야 합니다: {shrink}")
    w, h = x2 - x1, y2 - y1
    return np.array([x1 + shrink * w, y1 + shrink * h, x2 - shrink * w, y2 - shrink * h], dtype=np.float64)


def uv_in_box(uv: np.ndarray, xyxy: np.ndarray) -> np.ndarray:
    """(M, 2) 픽셀 좌표 중 박스 (x1, y1, x2, y2) 안(경계 포함)에 있는 것이 True 인 (M,) bool."""
    uv = np.asarray(uv)
    if uv.ndim != 2 or uv.shape[1] != 2:
        raise ValueError(f"uv 는 (M, 2) 이어야 합니다: {uv.shape}")
    x1, y1, x2, y2 = (float(v) for v in np.asarray(xyxy).reshape(4))
    return (uv[:, 0] >= x1) & (uv[:, 0] <= x2) & (uv[:, 1] >= y1) & (uv[:, 1] <= y2)


def frustum_mask(
    points_velo: np.ndarray,
    xyxy: np.ndarray,
    calib: KittiCalibration,
    image_shape: tuple[int, int],
    shrink: float = 0.0,
) -> np.ndarray:
    """2D 박스가 만드는 3D 프러스텀(절두체) 안의 점이 True 인 (N,) bool mask.

    프러스텀 = 카메라 중심에서 2D 박스의 네 모서리를 지나 뻗어 나가는 사각뿔. 어떤 3D 점이 이 안에 있다는 것은
    "카메라 앞에 있고, 이미지에 투영했을 때 (shrink 적용한) 박스 안에 떨어진다" 와 같은 말이다.
    그래서 3D 기하 대신 투영(project_velo_to_image) 으로 판정한다.

    Args:
        points_velo: (N, 3|4) Velodyne 점
        xyxy: (4,) image_02 픽셀 박스 (x1, y1, x2, y2)
        calib, image_shape: 투영에 필요 (image_shape 은 (H, W))
        shrink: 박스 축소 비율 (shrink_box 참고)
    """
    p = _xyz(points_velo)
    uv, _depth, in_img = project_velo_to_image(p, calib, image_shape)
    mask = np.zeros(len(p), dtype=bool)
    idx = np.flatnonzero(in_img)
    mask[idx[uv_in_box(uv, shrink_box(xyxy, shrink))]] = True
    return mask


# ----------------------------------------------------------------------------- 클러스터

def cluster_points(points_xyz: np.ndarray, cluster_cfg: dict) -> np.ndarray:
    """DBSCAN 으로 점을 덩어리(cluster) 로 나눈다. 반환 (M,) int 라벨, -1 은 noise (어느 덩어리에도 못 들어간 점).

    cluster_cfg 키 (fusion.cluster): eps (m, 이웃 반경), min_points (핵심점이 되기 위한 이웃 수, 자기 자신 포함).
    DBSCAN: 반경 eps 안에 min_points 개 이상의 이웃이 있는 점을 "핵심점" 으로 삼고, 핵심점끼리 eps 안에서 이어지면
    같은 덩어리로 묶는다. 덩어리 개수를 미리 정하지 않아도 되고, 떨어진 물체(차 vs 뒤 건물) 를 잘 분리한다.
    라벨은 0 부터 연속 정수이며 Open3D 구현은 점 순서가 같으면 같은 결과를 낸다 (결정적).
    """
    p = _xyz(points_xyz)
    if len(p) == 0:
        return np.zeros(0, dtype=np.int64)
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(p))
    labels = pcd.cluster_dbscan(eps=float(cluster_cfg["eps"]), min_points=int(cluster_cfg["min_points"]),
                                print_progress=False)
    return np.asarray(labels, dtype=np.int64)


# ----------------------------------------------------------------------------- 3D 박스

def aabb_from_points(points_xyz: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """축 정렬 박스(AABB): 각 축의 min/max 로 만든 박스.

    Returns: (center (3,) = (min + max) / 2, size (3,) = max - min = [l(x), w(y), h(z)]). yaw 는 0 으로 본다.
    한계: 비스듬히 선 차는 박스가 실제보다 커진다 (대각선 방향으로 늘어남). → obb_from_points_bev 로 개선.
    """
    p = _xyz(points_xyz)
    if len(p) == 0:
        raise ValueError("빈 점 집합으로는 박스를 만들 수 없습니다")
    lo, hi = p.min(axis=0), p.max(axis=0)
    return (lo + hi) / 2.0, hi - lo


def obb_from_points_bev(points_xyz: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """BEV(x-y 평면) PCA 로 방향(yaw) 을 찾은 뒤, 그 방향으로 정렬한 박스(OBB, oriented bounding box).

    알고리즘
        1. x, y 좌표의 공분산 행렬 (2×2) 의 고유벡터 중 고유값이 큰 쪽 = 점이 가장 길게 퍼진 방향 = 물체 길이 방향.
           yaw = atan2(v_y, v_x) (Velodyne x 축에서 반시계 방향, rad). [-π/2, π/2) 로 감싼다 (앞뒤 구분 불가).
        2. 점들을 -yaw 만큼 돌려(회전 행렬 Rz(-yaw)) 물체 로컬 프레임으로 옮긴 뒤 aabb_from_points 와 같이 min/max.
        3. 로컬 박스 중심을 +yaw 로 다시 돌려 Velodyne 프레임 중심을 얻는다.
    Returns: (center (3,), size (3,) [l(yaw 방향), w(수직 방향), h(z)], yaw rad)
    한계: LiDAR 는 보이는 면만 찍히므로 (L 자 모양) PCA 방향이 실제 진행 방향과 다를 수 있다. 점이 2개 이하면 yaw = 0.
    """
    p = _xyz(points_xyz)
    if len(p) == 0:
        raise ValueError("빈 점 집합으로는 박스를 만들 수 없습니다")
    if len(p) < 3:
        c, s = aabb_from_points(p)
        return c, s, 0.0
    xy = p[:, :2]
    cov = np.cov(xy - xy.mean(axis=0), rowvar=False)               # (2, 2)
    eigval, eigvec = np.linalg.eigh(cov)                             # 오름차순 고유값
    v = eigvec[:, np.argmax(eigval)]                                 # 최대 분산 방향
    yaw = math.atan2(v[1], v[0])
    yaw = (yaw + math.pi / 2) % math.pi - math.pi / 2                # [-π/2, π/2)
    c, s = math.cos(-yaw), math.sin(-yaw)
    rot = np.array([[c, -s], [s, c]])                                # Rz(-yaw)
    local = xy @ rot.T
    lo, hi = local.min(axis=0), local.max(axis=0)
    center_local = (lo + hi) / 2.0
    rot_back = np.array([[c, s], [-s, c]])                           # Rz(+yaw)
    center_xy = center_local @ rot_back.T
    z_lo, z_hi = p[:, 2].min(), p[:, 2].max()
    center = np.array([center_xy[0], center_xy[1], (z_lo + z_hi) / 2.0])
    size = np.array([hi[0] - lo[0], hi[1] - lo[1], z_hi - z_lo])
    return center, size, float(yaw)
