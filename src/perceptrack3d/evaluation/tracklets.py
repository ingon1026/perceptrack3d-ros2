"""KITTI raw `tracklet_labels.xml` 정답(GT) 파서.

좌표 규약 (KITTI raw devkit 공식 자료로 확인, docs/references.md "팀 D" 절 참고):
- 모든 tracklet 은 **Velodyne 좌표계** (x 전방, y 좌, z 상, m) 로 표현된다. (devkit readme.txt)
- (tx, ty, tz) 는 박스 **바닥면 중심**이다. devkit `run_demoTracklets.m` 이 모서리 z 를 [0, 0, 0, 0, h, h, h, h]
  로 두고 (tx, ty, tz) 를 더하므로 tz 는 바닥 높이다.
- (h, w, l): 같은 스크립트가 x = ±l/2, y = ±w/2, z = 0..h 로 두므로
  l 은 물체 진행 방향(로컬 x), w 는 로컬 y, h 는 z 방향 길이다.
- rz 는 z 축(yaw) 회전. 회전 행렬 R = Rz(rz) 를 로컬 모서리에 곱한 뒤 (tx, ty, tz) 를 더한다. rx, ry 는 항상 0.
- 프레임 번호 = first_frame + pose 인덱스 (`tracklets.h` 의 `getPose`).
- state: 0 UNSET, 1 INTERP(보간), 2 LABELED(직접 라벨). occlusion: -1 UNSET, 0 VISIBLE, 1 PARTLY, 2 FULLY.
  truncation: -1 UNSET, 0 IN_IMAGE, 1 TRUNCATED, 2 OUT_IMAGE, 99 BEHIND_IMAGE. (`tracklets.h` enum)

XML 은 boost serialization 형식이며 표준 라이브러리 `xml.etree.ElementTree` 만 사용한다.
"""
from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

from perceptrack3d.types import GtBox3D

# tracklets.h 의 enum 값 (문서화 및 필터링용)
STATE_UNSET, STATE_INTERP, STATE_LABELED = 0, 1, 2
OCC_UNSET, OCC_VISIBLE, OCC_PARTLY, OCC_FULLY = -1, 0, 1, 2
TRUNC_UNSET, TRUNC_IN_IMAGE, TRUNC_TRUNCATED, TRUNC_OUT_IMAGE, TRUNC_BEHIND_IMAGE = -1, 0, 1, 2, 99

_POSE_FLOAT_KEYS = ("tx", "ty", "tz", "rx", "ry", "rz")
# (태그, 기본값) — 옛 버전 XML 에는 없을 수 있으므로 tracklets.h 의 기본값을 따른다.
_POSE_INT_KEYS = (("state", STATE_UNSET), ("occlusion", OCC_UNSET), ("truncation", TRUNC_IN_IMAGE))


def _text(elem: ET.Element, tag: str) -> str:
    child = elem.find(tag)
    if child is None or child.text is None:
        raise ValueError(f"XML 에 <{tag}> 가 없습니다")
    return child.text.strip()


def _parse_pose(item: ET.Element) -> dict:
    pose = {k: float(_text(item, k)) for k in _POSE_FLOAT_KEYS}
    for k, default in _POSE_INT_KEYS:
        child = item.find(k)
        pose[k] = default if child is None or child.text is None else int(child.text)
    return pose


def load_tracklets(xml_path: str | Path) -> list[dict]:
    """tracklet_labels.xml 을 원본 구조 그대로 읽는다.

    Returns: tracklet dict 의 리스트 (XML 순서, 인덱스 = tracklet_id). 각 dict:
        {"tracklet_id": int, "object_type": str, "h": float, "w": float, "l": float,
         "first_frame": int, "poses": [ {tx, ty, tz, rx, ry, rz (float, m/rad, Velodyne),
                                         state, occlusion, truncation (int)} , ... ]}
        poses[i] 는 프레임 first_frame + i 의 자세. (tx, ty, tz) 는 박스 바닥 중심이다.
    Raises: FileNotFoundError (파일 없음), ValueError (<count> 와 실제 <item> 수 불일치, 태그 누락).
    """
    xml_path = Path(xml_path)
    if not xml_path.is_file():
        raise FileNotFoundError(f"tracklet XML 파일이 없습니다: {xml_path}")
    root = ET.parse(xml_path).getroot()
    tracklets_elem = root.find("tracklets") if root.tag != "tracklets" else root
    if tracklets_elem is None:
        raise ValueError(f"<tracklets> 요소가 없습니다: {xml_path}")

    count = int(_text(tracklets_elem, "count"))
    items = tracklets_elem.findall("item")
    if len(items) != count:
        raise ValueError(f"tracklet <count>={count} 인데 <item> 은 {len(items)}개입니다: {xml_path}")

    tracklets = []
    for tid, item in enumerate(items):
        poses_elem = item.find("poses")
        if poses_elem is None:
            raise ValueError(f"tracklet {tid} 에 <poses> 가 없습니다")
        pose_items = poses_elem.findall("item")
        pose_count = int(_text(poses_elem, "count"))
        if len(pose_items) != pose_count:
            raise ValueError(f"tracklet {tid}: poses <count>={pose_count} 인데 <item> 은 {len(pose_items)}개입니다")
        tracklets.append(
            {
                "tracklet_id": tid,
                "object_type": _text(item, "objectType"),
                "h": float(_text(item, "h")),
                "w": float(_text(item, "w")),
                "l": float(_text(item, "l")),
                "first_frame": int(_text(item, "first_frame")),
                "poses": [_parse_pose(p) for p in pose_items],
            }
        )
    return tracklets


def gt_boxes_by_frame(xml_path: str | Path) -> dict[int, list[GtBox3D]]:
    """XML 을 프레임별 GtBox3D 리스트로 변환한다.

    변환 규칙:
    - center = (tx, ty, tz + h/2): XML 의 **바닥 중심**을 박스 **기하 중심**으로 올린다.
      (융합 결과 Object3D.center 가 기하 중심이므로 같은 기준으로 비교하기 위함)
    - size = [l, w, h] (Velodyne x, y, z 방향 길이, 물체 로컬 프레임 기준. yaw 회전 전)
    - yaw = rz 를 [-pi, pi] 로 감싼 값 (devkit 의 wrapToPi 와 동일)
    - state == UNSET 인 pose 도 포함한다. 필요하면 호출 측에서 occlusion/truncation 으로 거른다.
    Returns: {frame_id: [GtBox3D, ...]} — GT 가 하나도 없는 프레임은 키가 없다.
    """
    by_frame: dict[int, list[GtBox3D]] = {}
    for t in load_tracklets(xml_path):
        size = np.array([t["l"], t["w"], t["h"]], dtype=np.float64)
        for i, p in enumerate(t["poses"]):
            frame_id = t["first_frame"] + i
            center = np.array([p["tx"], p["ty"], p["tz"] + t["h"] / 2.0], dtype=np.float64)
            yaw = math.atan2(math.sin(p["rz"]), math.cos(p["rz"]))
            by_frame.setdefault(frame_id, []).append(
                GtBox3D(
                    frame_id=frame_id,
                    tracklet_id=t["tracklet_id"],
                    object_type=t["object_type"],
                    center=center,
                    size=size.copy(),
                    yaw=yaw,
                    truncation=p["truncation"],
                    occlusion=p["occlusion"],
                )
            )
    return dict(sorted(by_frame.items()))


def box_corners_bev(center: np.ndarray, size: np.ndarray, yaw: float) -> np.ndarray:
    """BEV(x-y 평면) 4 모서리를 돌려준다. 시각화와 회전 IoU 가 공용으로 쓴다.

    Args:
        center: (2,) 또는 (3,) 박스 중심 [x, y(, z)] (Velodyne, m)
        size:   (2,) 또는 (3,) [l, w(, h)] — l 은 yaw 방향(로컬 x), w 는 로컬 y
        yaw:    z 축 회전 (rad)
    Returns: (4, 2) float64, 순서 = 앞좌, 앞우, 뒤우, 뒤좌 (devkit run_demoTracklets.m 과 동일).
    수식: corner = Rz(yaw) @ [±l/2, ±w/2]^T + center[:2]
    """
    center = np.asarray(center, dtype=np.float64)
    size = np.asarray(size, dtype=np.float64)
    assert center.shape[0] >= 2 and size.shape[0] >= 2, "center/size 는 최소 2 성분이어야 합니다"
    l, w = size[0], size[1]
    local = np.array([[l / 2, w / 2], [l / 2, -w / 2], [-l / 2, -w / 2], [-l / 2, w / 2]])
    c, s = math.cos(yaw), math.sin(yaw)
    rot = np.array([[c, -s], [s, c]])
    return local @ rot.T + center[:2]


def points_in_box_mask(points_xyz: np.ndarray, center: np.ndarray, size: np.ndarray, yaw: float) -> np.ndarray:
    """회전 3D 박스 안에 들어가는 점의 bool mask 를 돌려준다 (좌표 규약 검증·GT 점 수 계산용).

    Args:
        points_xyz: (N, 3) 또는 (N, 4) Velodyne 점 (뒤쪽 열은 무시)
        center: (3,) 박스 기하 중심, size: (3,) [l, w, h], yaw: rad
    Returns: (N,) bool
    수식: d = Rz(-yaw) @ (p - center); |d_x| <= l/2, |d_y| <= w/2, |d_z| <= h/2
    """
    pts = np.asarray(points_xyz, dtype=np.float64)
    assert pts.ndim == 2 and pts.shape[1] >= 3, f"points 는 (N, >=3) 이어야 합니다: {pts.shape}"
    d = pts[:, :3] - np.asarray(center, dtype=np.float64)
    c, s = math.cos(-yaw), math.sin(-yaw)
    dx = c * d[:, 0] - s * d[:, 1]
    dy = s * d[:, 0] + c * d[:, 1]
    half = np.asarray(size, dtype=np.float64) / 2.0
    return (np.abs(dx) <= half[0]) & (np.abs(dy) <= half[1]) & (np.abs(d[:, 2]) <= half[2])


def gt_in_front_fov(gt_boxes: list[GtBox3D], fov_deg: float = 81.4) -> list[GtBox3D]:
    """카메라 수평 시야(FOV) 안의 GT 만 남기는 **Velodyne 프레임 근사** 필터.

    GT 는 카메라 뒤·옆 물체도 포함하므로, 카메라 기반 파이프라인을 평가할 때는 시야 안 GT 만 세는 것이 공정하다.
    조건: center.x > 0 and |center.y| < center.x * tan(fov/2).
    근거: image_02 의 P_rect_02 는 fx = 721.54, 이미지 폭 1242 → 수평 FOV = 2·atan(621/721.54) ≈ 81.4°.
    Velodyne 원점과 카메라 원점의 오프셋(약 0.27 m)은 무시한다. 정확한 필터는 캘리브레이션을 사용한 투영으로 대체할 것.
    """
    tan_half = math.tan(math.radians(fov_deg) / 2.0)
    return [b for b in gt_boxes if b.center[0] > 0 and abs(b.center[1]) < b.center[0] * tan_half]
