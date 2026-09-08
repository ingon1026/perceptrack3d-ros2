"""파이프라인 전 단계가 공유하는 데이터 타입.

좌표계 규약 (docs/architecture.md 참고):
- 2D 박스(xyxy): image_02 픽셀 좌표, (x1, y1, x2, y2), float, 원점은 좌상단.
- 3D 위치/크기: **Velodyne 좌표계** (x 전방, y 좌측, z 상향, 단위 m).
  KITTI raw tracklet_labels.xml 도 Velodyne 좌표계이므로 평가와 바로 비교 가능하다.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class Detection2D:
    """YOLO 2D 검출 1개."""

    frame_id: int
    class_name: str            # COCO 이름 (car, truck, bus, person, bicycle, motorcycle)
    class_id: int              # COCO 클래스 id
    confidence: float
    xyxy: np.ndarray           # shape (4,), float32, 픽셀 (x1, y1, x2, y2)

    def as_dict(self) -> dict:
        return {
            "frame_id": self.frame_id,
            "class_name": self.class_name,
            "class_id": self.class_id,
            "confidence": float(self.confidence),
            "xyxy": [float(v) for v in self.xyxy],
        }


@dataclass
class Object3D:
    """2D 검출 + LiDAR 융합으로 얻은 3D 객체 1개 (Velodyne 좌표계)."""

    frame_id: int
    class_name: str
    confidence: float
    xyxy: np.ndarray                     # 원본 2D 박스 (4,)
    center: np.ndarray | None            # (3,) [x, y, z] m, Velodyne. status != "ok" 이면 None
    size: np.ndarray | None = None       # (3,) [l, w, h] m (x, y, z 방향 길이). 없으면 None
    yaw: float | None = None             # z 축 회전 (rad). AABB 기준이면 0.0
    n_points: int = 0                    # 박스/클러스터에 배정된 LiDAR 점 수
    status: str = "ok"                   # "ok" | "empty" | "sparse" | "invalid"
    method: str = "raw"                  # "raw" (Phase5 기준선) | "clustered" (Phase6)
    point_indices: np.ndarray | None = field(default=None, repr=False)  # 원본 점 인덱스 (시각화용)
    reason: str = ""                     # status != "ok" 일 때 이유 (팀 E 추가: "no_points", "duplicate", "no_cluster", ...)
    truncated: bool = False              # 2D 박스가 이미지 경계에 닿음 → 잘린 객체, 중심이 치우칠 수 있음 (팀 E 추가)

    def as_dict(self) -> dict:
        return {
            "frame_id": self.frame_id,
            "class_name": self.class_name,
            "confidence": float(self.confidence),
            "xyxy": [float(v) for v in self.xyxy],
            "center": None if self.center is None else [float(v) for v in self.center],
            "size": None if self.size is None else [float(v) for v in self.size],
            "yaw": self.yaw,
            "n_points": int(self.n_points),
            "status": self.status,
            "method": self.method,
            "reason": self.reason,
            "truncated": bool(self.truncated),
        }


@dataclass
class Track:
    """칼만 필터 트랙 1개. 상태 [x, y, z, vx, vy, vz], Velodyne 좌표계."""

    track_id: int
    class_name: str
    state: np.ndarray                    # (6,)
    covariance: np.ndarray               # (6, 6)
    hits: int = 0                        # 총 매칭 횟수
    age: int = 0                         # 생성 후 경과 프레임
    misses: int = 0                      # 연속 미검출 수
    confirmed: bool = False
    size: np.ndarray | None = None       # 마지막 관측 크기 (3,)
    history: list = field(default_factory=list)   # [(frame_id, x, y, z), ...]

    @property
    def position(self) -> np.ndarray:
        return self.state[:3]

    @property
    def velocity(self) -> np.ndarray:
        return self.state[3:]

    def as_dict(self, frame_id: int) -> dict:
        return {
            "frame_id": frame_id,
            "track_id": self.track_id,
            "class_name": self.class_name,
            "center": [float(v) for v in self.state[:3]],
            "velocity": [float(v) for v in self.state[3:]],
            "size": None if self.size is None else [float(v) for v in self.size],
            "hits": self.hits,
            "age": self.age,
            "misses": self.misses,
            "confirmed": self.confirmed,
        }


@dataclass
class GtBox3D:
    """tracklet_labels.xml 에서 파싱한 정답 3D 박스 1개 (Velodyne 좌표계)."""

    frame_id: int
    tracklet_id: int
    object_type: str                     # Car, Van, Truck, Pedestrian, Cyclist, ...
    center: np.ndarray                   # (3,) 박스 **중심** (xml 의 바닥 중심에서 h/2 올린 값)
    size: np.ndarray                     # (3,) [l, w, h]
    yaw: float                           # rz (rad)
    truncation: int = 0
    occlusion: int = 0
