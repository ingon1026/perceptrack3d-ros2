"""2D 검출 박스 + LiDAR 점 → 3D 객체 (Phase 5 기준선, Phase 6 개선).

입력
    points_velo : (N, 3|4) float32/64, Velodyne 프레임 (x 전방, y 좌, z 상, m)
    detections  : list[Detection2D] — xyxy 는 image_02 픽셀 (x1, y1, x2, y2)
    calib       : KittiCalibration, image_shape = (H, W)
    cfg         : load_config() 결과. cfg["fusion"] 키를 사용한다 (configs/kitti.yaml 주석 참고)
출력
    list[Object3D] — 검출과 같은 순서·같은 개수. center/size 는 **Velodyne 프레임** (m).
    status: "ok" | "empty"(박스 안 점 0) | "sparse"(min_points 미만) | "invalid"(클러스터 실패·비현실적 크기·중복)
    status != "ok" 이면 center = None 이고 reason 에 이유를 적는다.

Phase 5 기준선 (fuse_frame_raw, method = "raw")
    투영 → (축소한) 2D 박스 안의 점 선택 → 깊이(카메라 앞 거리) 의 하위 depth_percentile 값을 대표 깊이로
    → 대표 깊이 ± depth_band_m 안의 점들의 좌표별 **중앙값** 을 중심으로.
    왜 평균이 아니라 백분위·중앙값인가: 2D 박스 안에는 물체 뒤의 건물·나무 점(배경) 이 섞인다. 평균은 배경에 끌려가고,
    중앙값도 배경 점이 절반을 넘으면 틀린다. "가까운 쪽 30%" 를 대표로 잡으면 전경(물체) 을 우선한다.

Phase 6 (fuse_frame_clustered, method = "clustered")
    ROI 필터 → RANSAC 지면 제거 → 프러스텀(축소 박스) → DBSCAN → 클러스터 선택 규칙 → 중앙값 중심 + AABB/OBB 크기.
    클러스터 선택 규칙 (cfg["fusion"]["cluster_select"]):
        "largest" (기본): 점이 가장 많은 클러스터. 축소한 2D 박스 안에서는 물체가 가장 큰 덩어리인 경우가 대부분이고,
                         가까운 곳의 작은 노이즈 덩어리(지면 잔여·기둥·풀) 에 강하다. 전 프레임 스윕에서 nearest 보다 GT 매칭 +7%.
                         단, 박스가 크고 물체가 작으면(멀리 있는 사람 뒤 건물 벽) 배경을 고를 수 있다 → max_extent_m 로 일부 걸러짐.
        "nearest":       깊이 중앙값이 가장 작은(카메라에 가장 가까운) 클러스터. 배경(건물) 은 항상 더 멀다는 가정.

    표면→중심 오프셋 (cfg["fusion"]["surface_offset"], Phase 6 선택 개선):
        LiDAR 는 물체의 **보이는 면** 만 찍는다 (앞차라면 뒷면). 그래서 점 중앙값은 물체 중심이 아니라 센서 쪽 표면에 있고,
        차의 경우 GT 중심보다 약 l/2 ≈ 2 m 가깝게 나온다 (프레임 0/50/150 에서 -1.4 ~ -2.1 m 로 확인).
        보정: BEV 시선 단위벡터 u = c_xy / |c_xy| 에 대해 push = max(0, half_depth[class] - extent_u / 2),  c_xy += u · push.
        extent_u 는 클러스터 점을 u 에 투영한 퍼짐(max - min). 이미 옆면까지 보이면(extent 큼) 덜 민다.
        클래스 prior 에 의존하므로 YOLO 가 car/truck 을 헷갈리면 1 m 정도 오차가 생긴다 (한계).

중복 제거 (두 방법 공통): 서로 다른 2D 박스(예: 같은 차에 truck + bus) 가 같은 3D 위치를 가리키면
    BEV 중심 거리 < cfg["fusion"]["dedup_distance_m"] 인 쌍 중 confidence 낮은 쪽을 status = "invalid", reason = "duplicate" 로 바꾼다.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from perceptrack3d.fusion.point_filters import (
    aabb_from_points,
    cluster_points,
    obb_from_points_bev,
    remove_ground,
    roi_filter,
    shrink_box,
    uv_in_box,
)
from perceptrack3d.geometry.calibration import KittiCalibration
from perceptrack3d.geometry.projection import project_velo_to_image
from perceptrack3d.types import Detection2D, Object3D


def _xyz(points: np.ndarray) -> np.ndarray:
    pts = np.asarray(points)
    if pts.ndim != 2 or pts.shape[1] not in (3, 4):
        raise ValueError(f"points_velo 는 (N, 3) 또는 (N, 4) 이어야 합니다. 받은 shape: {pts.shape}")
    return pts[:, :3].astype(np.float64, copy=False)


def is_truncated(xyxy: np.ndarray, image_shape: tuple[int, int], margin: float = 1.0) -> bool:
    """2D 박스가 이미지 경계(좌/우/아래) 에 margin 픽셀 이내로 닿으면 True. 잘린 물체는 박스 중심이 물체 중심이 아니다."""
    H, W = int(image_shape[0]), int(image_shape[1])
    x1, y1, x2, y2 = (float(v) for v in np.asarray(xyxy).reshape(4))
    return x1 <= margin or x2 >= W - margin or y2 >= H - margin


def _new_object(det: Detection2D, method: str, image_shape) -> Object3D:
    return Object3D(
        frame_id=int(det.frame_id), class_name=det.class_name, confidence=float(det.confidence),
        xyxy=np.asarray(det.xyxy, dtype=np.float32), center=None, method=method,
        truncated=is_truncated(det.xyxy, image_shape),
    )


def dedup_objects(objects: list[Object3D], dist_m: float) -> list[Object3D]:
    """BEV 중심 거리 < dist_m 인 두 "ok" 객체 중 confidence 낮은 쪽을 invalid/duplicate 로 바꾼다 (제자리 수정 후 반환).

    confidence 내림차순으로 돌며, 이미 유지된 객체와 가까우면 탈락시킨다 (탐욕적). 클래스는 보지 않는다:
    person + bicycle 처럼 서로 다른 클래스가 한 물체(KITTI Cyclist) 를 가리키는 경우도 하나로 합쳐지는 효과가 있다.
    """
    ok_idx = sorted((i for i, o in enumerate(objects) if o.status == "ok"),
                    key=lambda i: -objects[i].confidence)
    kept: list[int] = []
    for i in ok_idx:
        c = objects[i].center[:2]
        dup_of = next((j for j in kept if np.linalg.norm(c - objects[j].center[:2]) < dist_m), None)
        if dup_of is None:
            kept.append(i)
        else:
            o = objects[i]
            o.status, o.reason, o.center, o.size, o.yaw = "invalid", f"duplicate_of_{dup_of}", None, None, None
    return objects


# ----------------------------------------------------------------------------- Phase 5

def fuse_frame_raw(
    points_velo: np.ndarray,
    detections: list[Detection2D],
    calib: KittiCalibration,
    image_shape: tuple[int, int],
    cfg: dict,
    timings: dict | None = None,
) -> list[Object3D]:
    """Phase 5 기준선. 모듈 독스트링의 "raw" 절차를 따른다.

    timings 를 dict 로 주면 "projection", "fusion" 단계의 경과 ms 를 채워 준다 (run_pipeline 의 런타임 집계용).
    Object3D.n_points = 중심 계산에 실제로 쓰인 점 수 (깊이 밴드 안), point_indices = 그 점들의 원본 인덱스.
    """
    f = cfg["fusion"]
    p = _xyz(points_velo)
    min_points, pct, band_m, shrink = int(f["min_points"]), float(f["depth_percentile"]), float(f["depth_band_m"]), float(f["box_shrink"])

    t0 = time.perf_counter()
    uv, depth, in_img = project_velo_to_image(p, calib, image_shape)
    idx = np.flatnonzero(in_img)                     # uv[i] ↔ 원본 점 idx[i]
    t1 = time.perf_counter()

    objects: list[Object3D] = []
    for det in detections:
        obj = _new_object(det, "raw", image_shape)
        sel = np.flatnonzero(uv_in_box(uv, shrink_box(det.xyxy, shrink)))   # 박스 안 점 (uv 로컬 인덱스)
        if len(sel) == 0:
            obj.status, obj.reason = "empty", "no_points_in_box"
        elif len(sel) < min_points:
            obj.status, obj.reason = "sparse", f"{len(sel)}<{min_points}_points"
            obj.n_points, obj.point_indices = len(sel), idx[sel]
        else:
            d = depth[sel]
            d_rep = float(np.percentile(d, pct, method="nearest"))   # 대표 깊이: 가까운 쪽 pct% (실제 점 값 → 밴드가 비지 않음)
            used = sel[np.abs(d - d_rep) <= band_m]                  # 대표 깊이 ± band 안의 점
            obj.center = np.median(p[idx[used]], axis=0)
            obj.n_points, obj.point_indices = len(used), idx[used]
        objects.append(obj)
    dedup_objects(objects, float(f.get("dedup_distance_m", 1.0)))
    t2 = time.perf_counter()
    if timings is not None:
        timings["projection"] = (t1 - t0) * 1e3
        timings["fusion"] = (t2 - t1) * 1e3
    return objects


# ----------------------------------------------------------------------------- Phase 6

def fuse_frame_clustered(
    points_velo: np.ndarray,
    detections: list[Detection2D],
    calib: KittiCalibration,
    image_shape: tuple[int, int],
    cfg: dict,
    timings: dict | None = None,
) -> list[Object3D]:
    """Phase 6. 모듈 독스트링의 "clustered" 절차를 따른다.

    cfg["fusion"] 추가 키: cluster_select ("nearest" | "largest"), use_obb (bool), max_extent_m (l 또는 w 가 이보다 크면 invalid),
                         dedup_distance_m. 없으면 기본값 largest / False / 15.0 / 1.0.
    timings 에 "filters"(ROI + 지면 제거), "projection", "cluster"(프러스텀 + DBSCAN + 박스, 검출 전체 합) ms 를 채운다.
    Object3D.n_points = 선택된 클러스터의 점 수, point_indices = 그 점들의 원본 인덱스, size = [l, w, h], yaw (AABB 면 0.0).
    """
    f = cfg["fusion"]
    p = _xyz(points_velo)
    min_points, shrink = int(f["min_points"]), float(f["box_shrink"])
    select = str(f.get("cluster_select", "largest"))
    min_ratio = float(f.get("cluster_min_ratio", 0.0))
    if select not in ("nearest", "largest"):
        raise ValueError(f"fusion.cluster_select 는 'nearest' 또는 'largest' 이어야 합니다: {select!r}")
    use_obb, max_extent = bool(f.get("use_obb", False)), float(f.get("max_extent_m", 15.0))
    so_cfg = f.get("surface_offset") or {}
    half_depth = so_cfg.get("half_depth_m") or {}

    # 1) ROI + 지면 제거 → keep (N,) bool
    t0 = time.perf_counter()
    roi = roi_filter(p, f["roi"])
    roi_idx = np.flatnonzero(roi)
    non_ground, _plane = remove_ground(p[roi_idx], f["ground"])
    keep = np.zeros(len(p), dtype=bool)
    keep[roi_idx[non_ground]] = True
    t1 = time.perf_counter()

    # 2) 투영 (전체 점 한 번)
    uv, depth, in_img = project_velo_to_image(p, calib, image_shape)
    idx = np.flatnonzero(in_img)
    keep_uv = keep[idx]                                              # uv 로컬 인덱스 기준 keep
    t2 = time.perf_counter()

    # 3) 검출마다 프러스텀 → DBSCAN → 클러스터 선택 → 박스
    objects: list[Object3D] = []
    for det in detections:
        obj = _new_object(det, "clustered", image_shape)
        sel = np.flatnonzero(uv_in_box(uv, shrink_box(det.xyxy, shrink)) & keep_uv)
        if len(sel) == 0:
            obj.status, obj.reason = "empty", "no_points_after_filters"
            objects.append(obj)
            continue
        if len(sel) < min_points:
            obj.status, obj.reason = "sparse", f"{len(sel)}<{min_points}_points"
            obj.n_points, obj.point_indices = len(sel), idx[sel]
            objects.append(obj)
            continue
        pts_sel = p[idx[sel]]
        labels = cluster_points(pts_sel, f["cluster"])
        if labels.max() < 0:
            obj.status, obj.reason = "invalid", "no_cluster"
            obj.n_points, obj.point_indices = len(sel), idx[sel]
            objects.append(obj)
            continue
        k = _select_cluster(labels, depth[sel], select, min_ratio)
        member = labels == k
        pts_c = pts_sel[member]
        if use_obb:
            _c, size, yaw = obb_from_points_bev(pts_c)
        else:
            _c, size = aabb_from_points(pts_c)
            yaw = 0.0
        if size[0] > max_extent or size[1] > max_extent:
            obj.status, obj.reason = "invalid", f"oversize_{size[0]:.1f}x{size[1]:.1f}m"
            obj.n_points, obj.point_indices = int(member.sum()), idx[sel[member]]
            objects.append(obj)
            continue
        center = np.median(pts_c, axis=0)                             # 로버스트 중심 (좌표별 중앙값) = 보이는 표면 위
        if so_cfg.get("enable", False):
            center = surface_to_center_offset(center, pts_c, float(half_depth.get(det.class_name, 0.0)))
        obj.center = center
        obj.size, obj.yaw = size, float(yaw)
        obj.n_points, obj.point_indices = int(member.sum()), idx[sel[member]]
        objects.append(obj)
    dedup_objects(objects, float(f.get("dedup_distance_m", 1.0)))
    t3 = time.perf_counter()
    if timings is not None:
        timings["filters"] = (t1 - t0) * 1e3
        timings["projection"] = (t2 - t1) * 1e3
        timings["cluster"] = (t3 - t2) * 1e3
    return objects


def surface_to_center_offset(center: np.ndarray, pts: np.ndarray, half_depth: float) -> np.ndarray:
    """표면 중앙값 center (3,) 를 BEV 시선 방향으로 밀어 물체 중심에 가깝게 만든다 (모듈 독스트링 "표면→중심 오프셋").

    Args: center (3,) Velodyne, pts (K, 3) 클러스터 점, half_depth 클래스 prior (m). 0 이면 그대로 돌려준다.
    Returns: (3,) 새 중심. z 는 바꾸지 않는다.
    """
    c = np.asarray(center, dtype=np.float64).copy()
    r = np.linalg.norm(c[:2])
    if half_depth <= 0.0 or r < 1e-6:
        return c
    u = c[:2] / r                                                    # 센서(원점) → 물체 BEV 단위벡터
    proj = pts[:, :2] @ u
    extent_u = float(proj.max() - proj.min())
    c[:2] += u * max(0.0, half_depth - extent_u / 2.0)
    return c


def _select_cluster(labels: np.ndarray, depth: np.ndarray, rule: str, min_ratio: float = 0.0) -> int:
    """클러스터 라벨(-1 제외) 중 하나를 고른다.

    largest: 점 수 최대 (동률이면 가까운 쪽).
    nearest: 점 수가 (최대 클러스터 × min_ratio) 이상인 후보 중 깊이 중앙값이 가장 작은 것.
             min_ratio = 0 이면 순수 nearest (작은 노이즈 덩어리에 약함), 0.3 이면 "충분히 큰 것 중 가장 가까운 것".
    """
    ids = np.unique(labels[labels >= 0])
    med = np.array([np.median(depth[labels == k]) for k in ids])
    cnt = np.array([(labels == k).sum() for k in ids])
    if rule == "largest":
        best = np.flatnonzero(cnt == cnt.max())
        return int(ids[best[np.argmin(med[best])]])
    eligible = np.flatnonzero(cnt >= min_ratio * cnt.max())
    return int(ids[eligible[np.argmin(med[eligible])]])


# ----------------------------------------------------------------------------- JSON 입출력

def save_objects_json(objects_by_frame: dict[int, list[Object3D]], path: str | Path) -> None:
    """{"0": [Object3D.as_dict()...], ...} 로 저장 (docs/architecture.md 4절). point_indices 는 저장하지 않는다."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {str(fid): [o.as_dict() for o in objects_by_frame[fid]] for fid in sorted(objects_by_frame)}
    with open(path, "w", encoding="utf-8") as fp:
        json.dump(payload, fp, indent=1)


def load_objects_json(path: str | Path) -> dict[int, list[Object3D]]:
    """save_objects_json 의 역. 평가·추적 팀이 융합을 다시 돌리지 않고 결과를 읽을 때 쓴다."""
    with open(path, "r", encoding="utf-8") as fp:
        payload = json.load(fp)
    out: dict[int, list[Object3D]] = {}
    for fid_str, items in payload.items():
        out[int(fid_str)] = [
            Object3D(
                frame_id=int(d["frame_id"]), class_name=str(d["class_name"]), confidence=float(d["confidence"]),
                xyxy=np.asarray(d["xyxy"], dtype=np.float32),
                center=None if d["center"] is None else np.asarray(d["center"], dtype=np.float64),
                size=None if d["size"] is None else np.asarray(d["size"], dtype=np.float64),
                yaw=d.get("yaw"), n_points=int(d.get("n_points", 0)), status=str(d["status"]),
                method=str(d.get("method", "raw")), reason=str(d.get("reason", "")),
                truncated=bool(d.get("truncated", False)),
            )
            for d in items
        ]
    return out
