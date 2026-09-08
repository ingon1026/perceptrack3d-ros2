"""visualization_node — 2D 박스와 트랙 ID 를 이미지에 그린다 (Phase 9, 팀 H).

구독 (ApproximateTimeSynchronizer 4-way, slop = sync_slop_s)
    /kitti/image_raw                sensor_msgs/Image
    /kitti/camera_info              sensor_msgs/CameraInfo       (트랙 중심 투영용 P)
    /perceptrack3d/detections_2d    vision_msgs/Detection2DArray
    /perceptrack3d/tracks           vision_msgs/Detection3DArray (velodyne 프레임)
발행
    /perceptrack3d/annotated_image  sensor_msgs/Image (header = 입력 이미지 header)

트랙 라벨 위치: 트랙 중심(velodyne) 을 P_velo_to_img 로 투영한 픽셀. 카메라 뒤나 이미지 밖이면 그리지 않는다.

stamp 일관성 검사: 동기화된 4 개 메시지의 stamp 가 **정확히** 같은지 매 프레임 확인한다 (파이프라인의 모든 노드가
입력 header 를 복사한다는 규약의 런타임 검증). 다르면 error 로그 + 카운트. 50 프레임마다 일치 수를 출력한다.
"""
from __future__ import annotations

import signal

import cv2
import message_filters
import numpy as np
import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from sensor_msgs.msg import CameraInfo, Image
from vision_msgs.msg import Detection2DArray, Detection3DArray

from perceptrack3d.config import DEFAULT_CONFIG, load_config
from perceptrack3d.geometry.projection import project_velo_to_image
from perceptrack3d.visualization.boxes2d import draw_detections

from perceptrack3d_ros import msg_utils as mu
from perceptrack3d_ros.tf_calib import TfCalibration


class VisualizationNode(Node):
    def __init__(self) -> None:
        super().__init__("visualization_node")
        self.declare_parameter("config_path", str(DEFAULT_CONFIG))
        self.declare_parameter("rate_hz", 10.0)
        self.declare_parameter("sync_slop_s", 0.02)
        load_config(self.get_parameter("config_path").value)
        self.rate_hz = float(self.get_parameter("rate_hz").value)

        self.calib = TfCalibration(self)
        self.pub = self.create_publisher(Image, "/perceptrack3d/annotated_image", 10)
        subs = [
            message_filters.Subscriber(self, Image, "/kitti/image_raw", qos_profile=10),
            message_filters.Subscriber(self, CameraInfo, "/kitti/camera_info", qos_profile=10),
            message_filters.Subscriber(self, Detection2DArray, "/perceptrack3d/detections_2d", qos_profile=10),
            message_filters.Subscriber(self, Detection3DArray, "/perceptrack3d/tracks", qos_profile=10),
        ]
        self.sync = message_filters.ApproximateTimeSynchronizer(
            subs, queue_size=10, slop=float(self.get_parameter("sync_slop_s").value))
        self.sync.registerCallback(self._on_frame)
        self._n, self._n_stamp_ok = 0, 0

    def _on_frame(self, image_msg: Image, info_msg: CameraInfo, dets_msg: Detection2DArray, tracks_msg: Detection3DArray) -> None:
        stamp = image_msg.header.stamp
        stamps_ok = all(mu.stamp_equal(stamp, m.header.stamp) for m in (info_msg, dets_msg, tracks_msg))
        self._n += 1
        self._n_stamp_ok += int(stamps_ok)
        if not stamps_ok:
            self.get_logger().error(
                f"stamp 불일치: image={mu.stamp_to_str(stamp)} info={mu.stamp_to_str(info_msg.header.stamp)} "
                f"dets={mu.stamp_to_str(dets_msg.header.stamp)} tracks={mu.stamp_to_str(tracks_msg.header.stamp)}")

        frame_id = mu.frame_from_stamp(stamp, self.rate_hz)
        image = mu.msg_to_image(image_msg)
        dets = mu.msg_to_detections(dets_msg, frame_id)
        out = draw_detections(image, dets)

        calib = self.calib.get(info_msg)
        tracks = mu.tracks_from_msg(tracks_msg)
        if calib is not None and tracks:
            centers = np.stack([t["center"] for t in tracks])
            uv, depth, mask = project_velo_to_image(centers, calib, out.shape[:2])
            for (u, v), d, t in zip(uv, depth, [t for t, m in zip(tracks, mask) if m]):
                color = mu.color_for_id(t["track_id"])
                bgr = (int(color.b * 255), int(color.g * 255), int(color.r * 255))
                cv2.circle(out, (int(u), int(v)), 5, bgr, -1)
                cv2.putText(out, f"ID {t['track_id']} {d:.1f}m", (int(u) + 6, int(v) - 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, bgr, 2, cv2.LINE_AA)
        cv2.putText(out, f"frame {frame_id}  dets {len(dets)}  tracks {len(tracks)}", (8, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
        self.pub.publish(mu.image_to_msg(out, image_msg.header))

        if self._n % 50 == 0:
            self.get_logger().info(f"frame {frame_id}: stamp 일치 {self._n_stamp_ok}/{self._n} 프레임, 트랙 {len(tracks)}")


def main(args=None) -> None:
    # 종료 규약: Ctrl+C(SIGINT) 를 받으면 rclpy 시그널 핸들러가 컨텍스트를 닫고 spin 이 ExternalShutdownException 을 던진다.
    # 터미널 Ctrl+C 는 프로세스 그룹 전체에 SIGINT 를 보내고 ros2 launch 도 자식에게 SIGINT 를 한 번 더 전달하므로,
    # 컨텍스트를 닫은 뒤에는 SIGINT 를 무시해 두 번째 신호가 인터프리터 종료(atexit) 를 KeyboardInterrupt 로 깨지 않게 한다.
    try:
        rclpy.init(args=args)
        node = VisualizationNode()
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
