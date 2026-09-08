"""fusion_node — 2D 검출 + LiDAR 점 → 3D 객체 (Phase 9, 팀 H).

구독 (ApproximateTimeSynchronizer, slop = sync_slop_s)
    /kitti/camera_info              sensor_msgs/CameraInfo      (이미지 크기 + P 행렬)
    /kitti/velodyne_points          sensor_msgs/PointCloud2     (Velodyne 프레임)
    /perceptrack3d/detections_2d    vision_msgs/Detection2DArray
발행
    /perceptrack3d/objects_3d       vision_msgs/Detection3DArray  header.frame_id = velodyne (입력 점군의 frame_id)
        bbox.center.position = 중심 (m), bbox.center.orientation = yaw → 쿼터니언, bbox.size = [l, w, h]
        results[0].hypothesis.class_id = 클래스, .score = 2D confidence. status != "ok" 객체는 싣지 않는다.
    /perceptrack3d/markers_objects  visualization_msgs/MarkerArray (반투명 CUBE + "class x.x m" TEXT, lifetime 0.15 s)

파라미터
    config_path, rate_hz, sync_slop_s, marker_lifetime_s
    method: "raw" (Phase 5 기준선 fuse_frame_raw) | "clustered" (Phase 6 fuse_frame_clustered, 기본)

이미지 자체는 필요 없다 — 융합 함수는 image_shape(H, W) 만 쓰므로 1.4 MB 이미지 대신 CameraInfo 의 width/height 를 쓴다.
"""
from __future__ import annotations

import signal
import time

import message_filters
import numpy as np
import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from sensor_msgs.msg import CameraInfo, PointCloud2
from std_msgs.msg import ColorRGBA, Header
from vision_msgs.msg import Detection2DArray, Detection3DArray
from visualization_msgs.msg import MarkerArray

from perceptrack3d.config import DEFAULT_CONFIG, load_config
from perceptrack3d.fusion.frustum_fusion import fuse_frame_clustered, fuse_frame_raw
from perceptrack3d.visualization.boxes2d import CLASS_COLORS

from perceptrack3d_ros import msg_utils as mu
from perceptrack3d_ros.tf_calib import TfCalibration


def _class_color(name: str, alpha: float) -> ColorRGBA:
    b, g, r = CLASS_COLORS.get(name, (160, 160, 160))       # boxes2d 는 BGR 정수
    return ColorRGBA(r=r / 255.0, g=g / 255.0, b=b / 255.0, a=alpha)


class FusionNode(Node):
    def __init__(self) -> None:
        super().__init__("fusion_node")
        self.declare_parameter("config_path", str(DEFAULT_CONFIG))
        self.declare_parameter("rate_hz", 10.0)
        self.declare_parameter("method", "clustered")
        self.declare_parameter("sync_slop_s", 0.02)
        self.declare_parameter("marker_lifetime_s", 0.15)
        self.cfg = load_config(self.get_parameter("config_path").value)
        self.rate_hz = float(self.get_parameter("rate_hz").value)
        self.method = str(self.get_parameter("method").value)
        if self.method not in ("raw", "clustered"):
            raise ValueError(f"method 는 'raw' 또는 'clustered' 이어야 합니다: {self.method!r}")
        self.fuse = fuse_frame_clustered if self.method == "clustered" else fuse_frame_raw
        self.lifetime = float(self.get_parameter("marker_lifetime_s").value)

        self.calib = TfCalibration(self)
        self.pub_objects = self.create_publisher(Detection3DArray, "/perceptrack3d/objects_3d", 10)
        self.pub_markers = self.create_publisher(MarkerArray, "/perceptrack3d/markers_objects", 10)
        subs = [
            message_filters.Subscriber(self, CameraInfo, "/kitti/camera_info", qos_profile=10),
            message_filters.Subscriber(self, PointCloud2, "/kitti/velodyne_points", qos_profile=10),
            message_filters.Subscriber(self, Detection2DArray, "/perceptrack3d/detections_2d", qos_profile=10),
        ]
        self.sync = message_filters.ApproximateTimeSynchronizer(
            subs, queue_size=10, slop=float(self.get_parameter("sync_slop_s").value))
        self.sync.registerCallback(self._on_frame)
        self._n, self._ms_sum = 0, 0.0
        self.get_logger().info(f"융합 방법: {self.method}")

    def _on_frame(self, info_msg: CameraInfo, points_msg: PointCloud2, dets_msg: Detection2DArray) -> None:
        calib = self.calib.get(info_msg)
        if calib is None:
            return
        frame_id = mu.frame_from_stamp(points_msg.header.stamp, self.rate_hz)
        t0 = time.perf_counter()
        points = mu.msg_to_points(points_msg)
        dets = mu.msg_to_detections(dets_msg, frame_id)
        objects = self.fuse(points, dets, calib, calib.image_shape, self.cfg)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        header = Header(stamp=points_msg.header.stamp, frame_id=points_msg.header.frame_id)   # velodyne
        self.pub_objects.publish(mu.objects_to_msg(objects, header))
        self.pub_markers.publish(self._markers(objects, header))

        self._n += 1
        self._ms_sum += elapsed_ms
        if self._n % 50 == 0:
            n_ok = sum(1 for o in objects if o.status == "ok")
            self.get_logger().info(f"frame {frame_id}: 검출 {len(dets)} → 3D ok {n_ok}, 최근 50 프레임 평균 {self._ms_sum / 50:.1f} ms")
            self._ms_sum = 0.0

    def _markers(self, objects, header: Header) -> MarkerArray:
        arr = MarkerArray()
        for i, o in enumerate(objects):
            if o.status != "ok" or o.center is None:
                continue
            size = mu.size_or_default(o.size, o.class_name)
            yaw = float(o.yaw or 0.0)
            arr.markers.append(mu.cube_marker(header, "object_box", i, o.center, size, yaw, _class_color(o.class_name, 0.4), self.lifetime))
            text_pos = np.array([o.center[0], o.center[1], o.center[2] + size[2] / 2 + 0.5])
            arr.markers.append(mu.text_marker(header, "object_text", i, text_pos, f"{o.class_name} {o.center[0]:.1f}m",
                                              _class_color(o.class_name, 1.0), self.lifetime))
        return arr


def main(args=None) -> None:
    # 종료 규약: Ctrl+C(SIGINT) 를 받으면 rclpy 시그널 핸들러가 컨텍스트를 닫고 spin 이 ExternalShutdownException 을 던진다.
    # 터미널 Ctrl+C 는 프로세스 그룹 전체에 SIGINT 를 보내고 ros2 launch 도 자식에게 SIGINT 를 한 번 더 전달하므로,
    # 컨텍스트를 닫은 뒤에는 SIGINT 를 무시해 두 번째 신호가 인터프리터 종료(atexit) 를 KeyboardInterrupt 로 깨지 않게 한다.
    try:
        rclpy.init(args=args)
        node = FusionNode()
        try:
            rclpy.spin(node)
        finally:
            if rclpy.ok():                      # 컨텍스트가 이미 닫혔으면 destroy_node() 가 실패하므로 건너뛴다
                node.destroy_node()
            rclpy.try_shutdown()
            signal.signal(signal.SIGINT, signal.SIG_IGN)
    except (KeyboardInterrupt, ExternalShutdownException):
        signal.signal(signal.SIGINT, signal.SIG_IGN)


if __name__ == "__main__":
    main()
