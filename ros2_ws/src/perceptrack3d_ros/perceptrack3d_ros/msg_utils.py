"""ROS2 메시지 <-> numpy / perceptrack3d 타입 변환 헬퍼 (Phase 9, 팀 H).

이 파일이 파이프라인 타입(Detection2D, Object3D, Track, KittiCalibration) 과 ROS2 메시지 사이의
유일한 경계다. 노드 파일들은 여기 있는 함수만 호출한다.

좌표계 규약
- 3D 값은 전부 Velodyne 프레임 (x 전방, y 좌, z 상, m) = ROS REP-103 규약과 같은 축 방향.
- 이미지 픽셀은 image_02 (u 우, v 아래, 원점 좌상단).
- TF: `TransformStamped(header.frame_id = 부모, child_frame_id = 자식)` 의 transform 은
  "자식 프레임 좌표 → 부모 프레임 좌표" 로 옮기는 강체 변환 T_parent_child 다.

타임스탬프 규약
- 플레이어는 stamp = frame_id / rate_hz (초) 로 결정적 시각을 만든다. 하위 노드는 같은 stamp 를 그대로
  복사하므로 `frame_from_stamp()` 로 언제든 프레임 번호를 되찾을 수 있다.

cv_bridge 는 이 환경(OpenCV 5 + numpy 2.5) 에서 cv2_to_imgmsg 가 KeyError 로 실패해 쓰지 않는다.
sensor_msgs/Image 는 bgr8 인코딩으로 직접 채운다.
"""
from __future__ import annotations

import math

import numpy as np
from builtin_interfaces.msg import Duration as DurationMsg
from builtin_interfaces.msg import Time as TimeMsg
from geometry_msgs.msg import Point, Quaternion, TransformStamped
from scipy.spatial.transform import Rotation
from sensor_msgs.msg import CameraInfo, Image, PointCloud2, PointField
from std_msgs.msg import ColorRGBA, Header
from vision_msgs.msg import (
    Detection2D as Detection2DMsg,
    Detection2DArray,
    Detection3D as Detection3DMsg,
    Detection3DArray,
    ObjectHypothesisWithPose,
)
from visualization_msgs.msg import Marker

from perceptrack3d.geometry.calibration import KittiCalibration
from perceptrack3d.types import Detection2D, Object3D, Track

# COCO 클래스 id (docs/references.md 팀 B 절). Detection2D 메시지에는 이름만 싣고 id 는 여기서 복원한다.
COCO_CLASS_IDS: dict[str, int] = {"person": 0, "bicycle": 1, "car": 2, "motorcycle": 3, "bus": 5, "truck": 7}

# 크기 정보가 없는 객체(method=raw) 를 RViz 에 그릴 때 쓰는 클래스별 대략적 크기 [l, w, h] (m).
DEFAULT_SIZES: dict[str, tuple[float, float, float]] = {
    "car": (4.0, 1.8, 1.5),
    "truck": (7.0, 2.5, 3.0),
    "bus": (10.0, 2.5, 3.0),
    "person": (0.6, 0.6, 1.7),
    "bicycle": (1.7, 0.6, 1.5),
    "motorcycle": (2.0, 0.8, 1.5),
}

# 트랙 id 별 고정 색 (matplotlib tab20 값). id % 20 으로 고른다.
_PALETTE = [
    (0.121, 0.466, 0.705), (0.682, 0.780, 0.909), (1.000, 0.498, 0.054), (1.000, 0.733, 0.470),
    (0.172, 0.627, 0.172), (0.596, 0.874, 0.541), (0.839, 0.152, 0.156), (1.000, 0.596, 0.588),
    (0.580, 0.403, 0.741), (0.772, 0.690, 0.835), (0.549, 0.337, 0.294), (0.768, 0.611, 0.580),
    (0.890, 0.466, 0.760), (0.968, 0.713, 0.823), (0.498, 0.498, 0.498), (0.780, 0.780, 0.780),
    (0.737, 0.741, 0.133), (0.858, 0.858, 0.552), (0.090, 0.745, 0.811), (0.619, 0.854, 0.898),
]


# ============================================================================ 시간 / 프레임 번호

def stamp_from_frame(frame_id: int, rate_hz: float) -> TimeMsg:
    """frame_id → stamp. 초 = frame_id / rate_hz. (예: 10 Hz 에서 프레임 37 → 3.7 s)"""
    total_ns = int(round(frame_id * 1e9 / rate_hz))
    return TimeMsg(sec=total_ns // 1_000_000_000, nanosec=total_ns % 1_000_000_000)


def frame_from_stamp(stamp: TimeMsg, rate_hz: float) -> int:
    """stamp → frame_id (stamp_from_frame 의 역). 반올림하므로 나노초 오차에 안전하다."""
    return int(round((stamp.sec + stamp.nanosec * 1e-9) * rate_hz))


def stamp_equal(a: TimeMsg, b: TimeMsg) -> bool:
    return a.sec == b.sec and a.nanosec == b.nanosec


def stamp_to_str(stamp: TimeMsg) -> str:
    return f"{stamp.sec}.{stamp.nanosec:09d}"


# ============================================================================ 이미지

def image_to_msg(img_bgr: np.ndarray, header: Header) -> Image:
    """(H, W, 3) uint8 BGR → sensor_msgs/Image (encoding='bgr8'). 데이터는 행 우선 연속 바이트."""
    if img_bgr.ndim != 3 or img_bgr.shape[2] != 3 or img_bgr.dtype != np.uint8:
        raise ValueError(f"이미지는 (H, W, 3) uint8 이어야 합니다: {img_bgr.shape} {img_bgr.dtype}")
    h, w = img_bgr.shape[:2]
    msg = Image()
    msg.header = header
    msg.height, msg.width = int(h), int(w)
    msg.encoding = "bgr8"
    msg.is_bigendian = 0
    msg.step = int(w * 3)                       # 한 행의 바이트 수
    msg.data = np.ascontiguousarray(img_bgr).tobytes()
    return msg


def msg_to_image(msg: Image) -> np.ndarray:
    """sensor_msgs/Image (bgr8 또는 rgb8) → (H, W, 3) uint8 BGR. 메시지 버퍼를 복사한다."""
    if msg.encoding not in ("bgr8", "rgb8"):
        raise ValueError(f"지원하지 않는 encoding: {msg.encoding}")
    img = np.frombuffer(bytes(msg.data), dtype=np.uint8).reshape(msg.height, msg.step)[:, : msg.width * 3]
    img = img.reshape(msg.height, msg.width, 3)
    if msg.encoding == "rgb8":
        img = img[:, :, ::-1]
    return np.ascontiguousarray(img)


# ============================================================================ 포인트클라우드

_XYZI_FIELDS = [
    PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
    PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
    PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
    PointField(name="intensity", offset=12, datatype=PointField.FLOAT32, count=1),
]
_PF_DTYPES = {
    PointField.INT8: np.int8, PointField.UINT8: np.uint8, PointField.INT16: np.int16, PointField.UINT16: np.uint16,
    PointField.INT32: np.int32, PointField.UINT32: np.uint32, PointField.FLOAT32: np.float32, PointField.FLOAT64: np.float64,
}


def points_to_msg(points: np.ndarray, header: Header) -> PointCloud2:
    """(N, 4) float32 [x, y, z, reflectance] → PointCloud2 (fields x, y, z, intensity; point_step 16).

    KITTI .bin 의 메모리 배치가 이미 float32 x4 연속이라 tobytes() 한 번으로 끝난다 (per-point 루프 없음).
    """
    pts = np.ascontiguousarray(np.asarray(points, dtype=np.float32))
    if pts.ndim != 2 or pts.shape[1] != 4:
        raise ValueError(f"points 는 (N, 4) 이어야 합니다: {pts.shape}")
    msg = PointCloud2()
    msg.header = header
    msg.height = 1                                  # 비정렬(unorganized) 클라우드
    msg.width = int(pts.shape[0])
    msg.fields = list(_XYZI_FIELDS)
    msg.is_bigendian = False
    msg.point_step = 16
    msg.row_step = 16 * msg.width
    msg.data = pts.tobytes()
    msg.is_dense = True
    return msg


def msg_to_points(msg: PointCloud2) -> np.ndarray:
    """PointCloud2 → (N, 4) float32 [x, y, z, intensity]. intensity 필드가 없으면 0 으로 채운다.

    fields 의 offset/datatype 을 읽어 구조화 dtype 을 만들므로 point_step 이 16 이 아니어도 동작한다.
    """
    names = {f.name: f for f in msg.fields}
    for k in ("x", "y", "z"):
        if k not in names:
            raise ValueError(f"PointCloud2 에 '{k}' 필드가 없습니다: {list(names)}")
    dtype = np.dtype({
        "names": [f.name for f in msg.fields],
        "formats": [_PF_DTYPES[f.datatype] for f in msg.fields],
        "offsets": [f.offset for f in msg.fields],
        "itemsize": msg.point_step,
    })
    n = msg.width * msg.height
    rec = np.frombuffer(bytes(msg.data), dtype=dtype, count=n)
    out = np.zeros((n, 4), dtype=np.float32)
    out[:, 0], out[:, 1], out[:, 2] = rec["x"], rec["y"], rec["z"]
    if "intensity" in names:
        out[:, 3] = rec["intensity"]
    return out


# ============================================================================ CameraInfo / 캘리브레이션

def camera_info_from_calib(calib: KittiCalibration, header: Header) -> CameraInfo:
    """KittiCalibration → sensor_msgs/CameraInfo.

    - K (3x3) = P_rect_02[:, :3]  (정류 이미지의 내부 파라미터 f, cx, cy)
    - P (3x4) = P_rect_02          (rect 프레임 점 → image_02 픽셀)
    - R (3x3) = R_rect_00[:3, :3]  (cam0 → rect 회전)
    - D = 0 (정류(rectified) 이미지라 왜곡이 이미 제거됨), distortion_model = "plumb_bob"
    """
    w, h = calib.image_size
    msg = CameraInfo()
    msg.header = header
    msg.width, msg.height = int(w), int(h)
    msg.distortion_model = "plumb_bob"
    msg.d = [0.0] * 5
    msg.k = [float(v) for v in calib.P_rect_02[:, :3].reshape(-1)]
    msg.r = [float(v) for v in calib.R_rect_00[:3, :3].reshape(-1)]
    msg.p = [float(v) for v in calib.P_rect_02.reshape(-1)]
    return msg


def calibration_from_ros(cam_info: CameraInfo, T_velo_to_cam: np.ndarray) -> KittiCalibration:
    """CameraInfo + TF 로 얻은 (4, 4) 변환 → KittiCalibration.

    T_velo_to_cam 은 TF `camera_02 ← velodyne` (velodyne 점을 camera_02 좌표로 옮기는 변환) 이며,
    camera_02 TF 프레임은 이미 정류(rect) 프레임이므로 R_rect_00 = I 로 둔다.
    결과의 P_velo_to_img = P @ I @ T 가 파일 기반 calib 의 P_rect_02 @ R_rect_00 @ T_velo_to_cam 과 같다.
    """
    P = np.asarray(cam_info.p, dtype=np.float64).reshape(3, 4)
    return KittiCalibration(
        T_velo_to_cam=np.asarray(T_velo_to_cam, dtype=np.float64),
        R_rect_00=np.eye(4),
        P_rect_02=P,
        image_size=(int(cam_info.width), int(cam_info.height)),
    )


# ============================================================================ 회전 / TF

def quat_from_matrix(R: np.ndarray) -> Quaternion:
    """(3, 3) 회전 행렬 → geometry_msgs/Quaternion (x, y, z, w)."""
    x, y, z, w = Rotation.from_matrix(np.asarray(R, dtype=np.float64)).as_quat()
    return Quaternion(x=float(x), y=float(y), z=float(z), w=float(w))


def matrix_from_quat(q: Quaternion) -> np.ndarray:
    """geometry_msgs/Quaternion → (3, 3) 회전 행렬."""
    return Rotation.from_quat([q.x, q.y, q.z, q.w]).as_matrix()


def quat_from_yaw(yaw: float) -> Quaternion:
    """z 축 회전 yaw (rad) → Quaternion. yaw=0 이면 항등."""
    return Quaternion(x=0.0, y=0.0, z=float(math.sin(yaw / 2.0)), w=float(math.cos(yaw / 2.0)))


def transform_msg_from_matrix(T: np.ndarray, parent: str, child: str, stamp: TimeMsg) -> TransformStamped:
    """(4, 4) T_parent_child (자식 좌표 → 부모 좌표) → TransformStamped(header.frame_id=parent, child_frame_id=child)."""
    T = np.asarray(T, dtype=np.float64)
    if T.shape != (4, 4):
        raise ValueError(f"T 는 (4, 4) 이어야 합니다: {T.shape}")
    msg = TransformStamped()
    msg.header.stamp = stamp
    msg.header.frame_id = parent
    msg.child_frame_id = child
    msg.transform.translation.x = float(T[0, 3])
    msg.transform.translation.y = float(T[1, 3])
    msg.transform.translation.z = float(T[2, 3])
    msg.transform.rotation = quat_from_matrix(T[:3, :3])
    return msg


def matrix_from_transform_msg(msg: TransformStamped) -> np.ndarray:
    """TransformStamped → (4, 4) 동차 행렬 (transform_msg_from_matrix 의 역)."""
    T = np.eye(4)
    T[:3, :3] = matrix_from_quat(msg.transform.rotation)
    t = msg.transform.translation
    T[:3, 3] = [t.x, t.y, t.z]
    return T


# ============================================================================ Detection2D

def detections_to_msg(dets: list[Detection2D], header: Header) -> Detection2DArray:
    """list[Detection2D] → vision_msgs/Detection2DArray.

    bbox.center.position = 박스 중심 픽셀, size_x/size_y = 폭/높이 (px).
    results[0].hypothesis.class_id = COCO 이름 (예: 'car'), score = confidence.
    """
    arr = Detection2DArray()
    arr.header = header
    for d in dets:
        x1, y1, x2, y2 = (float(v) for v in d.xyxy)
        m = Detection2DMsg()
        m.header = header
        m.bbox.center.position.x = (x1 + x2) / 2.0
        m.bbox.center.position.y = (y1 + y2) / 2.0
        m.bbox.center.theta = 0.0
        m.bbox.size_x = x2 - x1
        m.bbox.size_y = y2 - y1
        hyp = ObjectHypothesisWithPose()
        hyp.hypothesis.class_id = str(d.class_name)
        hyp.hypothesis.score = float(d.confidence)
        m.results.append(hyp)
        arr.detections.append(m)
    return arr


def msg_to_detections(msg: Detection2DArray, frame_id: int) -> list[Detection2D]:
    """Detection2DArray → list[Detection2D]. xyxy 는 float32 (4,) 픽셀."""
    out: list[Detection2D] = []
    for m in msg.detections:
        cx, cy = m.bbox.center.position.x, m.bbox.center.position.y
        w, h = m.bbox.size_x, m.bbox.size_y
        name = m.results[0].hypothesis.class_id if m.results else "unknown"
        conf = m.results[0].hypothesis.score if m.results else 0.0
        out.append(Detection2D(
            frame_id=int(frame_id),
            class_name=name,
            class_id=COCO_CLASS_IDS.get(name, -1),
            confidence=float(conf),
            xyxy=np.array([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], dtype=np.float32),
        ))
    return out


# ============================================================================ Object3D / Track  <->  Detection3D

def _fill_box3d(m: Detection3DMsg, center: np.ndarray, size: np.ndarray | None, yaw: float | None) -> None:
    m.bbox.center.position.x = float(center[0])
    m.bbox.center.position.y = float(center[1])
    m.bbox.center.position.z = float(center[2])
    m.bbox.center.orientation = quat_from_yaw(yaw or 0.0)
    if size is not None:
        m.bbox.size.x, m.bbox.size.y, m.bbox.size.z = (float(v) for v in size)   # (l, w, h) = (x, y, z 방향 길이)


def objects_to_msg(objects: list[Object3D], header: Header) -> Detection3DArray:
    """list[Object3D] → Detection3DArray. status == 'ok' 이고 center 가 있는 객체만 싣는다.

    header.frame_id 는 'velodyne' 이어야 한다 (Object3D.center 가 Velodyne 프레임).
    results[0].hypothesis.class_id = class_name, score = 2D confidence, pose = 박스 중심.
    bbox.size = [l, w, h]; size 가 없으면(raw) 0 으로 남긴다. id = f"{method}:{n_points}" (디버그용).
    """
    arr = Detection3DArray()
    arr.header = header
    for o in objects:
        if o.status != "ok" or o.center is None:
            continue
        m = Detection3DMsg()
        m.header = header
        m.id = f"{o.method}:{o.n_points}"
        _fill_box3d(m, o.center, o.size, o.yaw)
        hyp = ObjectHypothesisWithPose()
        hyp.hypothesis.class_id = str(o.class_name)
        hyp.hypothesis.score = float(o.confidence)
        hyp.pose.pose = m.bbox.center
        m.results.append(hyp)
        arr.detections.append(m)
    return arr


def msg_to_objects(msg: Detection3DArray, frame_id: int) -> list[Object3D]:
    """Detection3DArray → list[Object3D] (status='ok'). size 가 전부 0 이면 None 으로 되돌린다."""
    out: list[Object3D] = []
    for m in msg.detections:
        p = m.bbox.center.position
        s = m.bbox.size
        size = None if (s.x == 0.0 and s.y == 0.0 and s.z == 0.0) else np.array([s.x, s.y, s.z], dtype=float)
        yaw = float(2.0 * math.atan2(m.bbox.center.orientation.z, m.bbox.center.orientation.w))
        name = m.results[0].hypothesis.class_id if m.results else "unknown"
        conf = m.results[0].hypothesis.score if m.results else 0.0
        method, _, n_pts = m.id.partition(":")
        out.append(Object3D(
            frame_id=int(frame_id), class_name=name, confidence=float(conf),
            xyxy=np.zeros(4, dtype=np.float32),                  # 2D 박스는 3D 메시지에 없다 (추적에는 불필요)
            center=np.array([p.x, p.y, p.z], dtype=float), size=size, yaw=yaw,
            n_points=int(n_pts) if n_pts.isdigit() else 0, status="ok", method=method or "unknown",
        ))
    return out


def tracks_to_msg(tracks: list[Track], header: Header) -> Detection3DArray:
    """list[Track] → Detection3DArray. id = str(track_id), class_id = class_name, score = 1.0.

    bbox.center = 칼만 상태의 위치 (state[:3]), size = 마지막 관측 크기(없으면 0). yaw 는 추적하지 않으므로 0.
    """
    arr = Detection3DArray()
    arr.header = header
    for t in tracks:
        m = Detection3DMsg()
        m.header = header
        m.id = str(int(t.track_id))
        _fill_box3d(m, t.state[:3], t.size, 0.0)
        hyp = ObjectHypothesisWithPose()
        hyp.hypothesis.class_id = str(t.class_name)
        hyp.hypothesis.score = 1.0
        hyp.pose.pose = m.bbox.center
        m.results.append(hyp)
        arr.detections.append(m)
    return arr


def tracks_from_msg(msg: Detection3DArray) -> list[dict]:
    """tracks_to_msg 의 역 (시각화용). 각 항목: {track_id, class_name, center (3,), size (3,)|None}."""
    out = []
    for m in msg.detections:
        p, s = m.bbox.center.position, m.bbox.size
        out.append({
            "track_id": int(m.id) if m.id.lstrip("-").isdigit() else -1,
            "class_name": m.results[0].hypothesis.class_id if m.results else "unknown",
            "center": np.array([p.x, p.y, p.z], dtype=float),
            "size": None if (s.x == 0.0 and s.y == 0.0 and s.z == 0.0) else np.array([s.x, s.y, s.z], dtype=float),
        })
    return out


# ============================================================================ Marker

def color_for_id(i: int, alpha: float = 1.0) -> ColorRGBA:
    r, g, b = _PALETTE[int(i) % len(_PALETTE)]
    return ColorRGBA(r=r, g=g, b=b, a=float(alpha))


def size_or_default(size: np.ndarray | None, class_name: str) -> np.ndarray:
    """size 가 없으면 클래스별 기본 크기. 0 이 섞여 있어도 최소 0.2 m 로 올려 RViz 에서 보이게 한다."""
    if size is None:
        size = np.array(DEFAULT_SIZES.get(class_name, (1.0, 1.0, 1.0)), dtype=float)
    return np.maximum(np.asarray(size, dtype=float), 0.2)


def _base_marker(header: Header, ns: str, mid: int, mtype: int, lifetime_s: float) -> Marker:
    m = Marker()
    m.header = header
    m.ns = ns
    m.id = int(mid)
    m.type = mtype
    m.action = Marker.ADD
    m.lifetime = DurationMsg(sec=int(lifetime_s), nanosec=int((lifetime_s % 1.0) * 1e9))
    m.pose.orientation.w = 1.0
    return m


def cube_marker(header: Header, ns: str, mid: int, center: np.ndarray, size: np.ndarray, yaw: float,
                color: ColorRGBA, lifetime_s: float) -> Marker:
    """반투명 CUBE. scale = (l, w, h), pose = 중심 + z 축 yaw 회전."""
    m = _base_marker(header, ns, mid, Marker.CUBE, lifetime_s)
    m.pose.position.x, m.pose.position.y, m.pose.position.z = (float(v) for v in center)
    m.pose.orientation = quat_from_yaw(yaw)
    m.scale.x, m.scale.y, m.scale.z = (float(v) for v in size)
    m.color = color
    return m


def text_marker(header: Header, ns: str, mid: int, position: np.ndarray, text: str, color: ColorRGBA,
                lifetime_s: float, height_m: float = 0.8) -> Marker:
    """TEXT_VIEW_FACING. scale.z 가 글자 높이(m)."""
    m = _base_marker(header, ns, mid, Marker.TEXT_VIEW_FACING, lifetime_s)
    m.pose.position.x, m.pose.position.y, m.pose.position.z = (float(v) for v in position)
    m.scale.z = float(height_m)
    m.color = color
    m.text = text
    return m


def line_strip_marker(header: Header, ns: str, mid: int, points_xyz: np.ndarray, color: ColorRGBA,
                      lifetime_s: float, width_m: float = 0.1) -> Marker:
    """LINE_STRIP (궤적). points_xyz (K, 3), scale.x 가 선 두께(m)."""
    m = _base_marker(header, ns, mid, Marker.LINE_STRIP, lifetime_s)
    m.scale.x = float(width_m)
    m.color = color
    m.points = [Point(x=float(p[0]), y=float(p[1]), z=float(p[2])) for p in np.asarray(points_xyz, dtype=float)]
    return m
