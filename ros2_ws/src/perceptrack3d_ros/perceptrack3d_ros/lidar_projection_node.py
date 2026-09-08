"""lidar_projection_node — LiDAR 점을 image_02 에 투영한 오버레이 이미지를 만든다 (Phase 9, 팀 H).

구독 (message_filters.ApproximateTimeSynchronizer, slop = sync_slop_s)
    /kitti/image_raw        sensor_msgs/Image
    /kitti/velodyne_points  sensor_msgs/PointCloud2
    /kitti/camera_info      sensor_msgs/CameraInfo
발행
    /perceptrack3d/projection_image   sensor_msgs/Image (header = 입력 이미지 header)

캘리브레이션은 파일이 아니라 CameraInfo(P) + TF(camera_02 ← velodyne) 에서 만든다 (tf_calib.TfCalibration).
투영 자체는 Phase 3 의 perceptrack3d.geometry.projection.project_velo_to_image 를 그대로 호출한다.
"""
from __future__ import annotations

import signal
import time

import message_filters
import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from sensor_msgs.msg import CameraInfo, Image, PointCloud2

from perceptrack3d.config import DEFAULT_CONFIG, load_config
from perceptrack3d.geometry.projection import project_velo_to_image
from perceptrack3d.visualization.overlay import draw_projected_points

from perceptrack3d_ros import msg_utils as mu
from perceptrack3d_ros.tf_calib import TfCalibration


class LidarProjectionNode(Node):
    def __init__(self) -> None:
        super().__init__("lidar_projection_node")
        self.declare_parameter("config_path", str(DEFAULT_CONFIG))
        self.declare_parameter("sync_slop_s", 0.02)
        self.declare_parameter("point_radius_px", 1)
        load_config(self.get_parameter("config_path").value)      # 경로 검증용 (이 노드는 cfg 값이 필요 없다)
        self.radius = int(self.get_parameter("point_radius_px").value)

        self.calib = TfCalibration(self)
        self.pub = self.create_publisher(Image, "/perceptrack3d/projection_image", 10)
        subs = [
            message_filters.Subscriber(self, Image, "/kitti/image_raw", qos_profile=10),
            message_filters.Subscriber(self, PointCloud2, "/kitti/velodyne_points", qos_profile=10),
            message_filters.Subscriber(self, CameraInfo, "/kitti/camera_info", qos_profile=10),
        ]
        self.sync = message_filters.ApproximateTimeSynchronizer(
            subs, queue_size=10, slop=float(self.get_parameter("sync_slop_s").value))
        self.sync.registerCallback(self._on_frame)
        self._n, self._ms_sum = 0, 0.0

    def _on_frame(self, image_msg: Image, points_msg: PointCloud2, info_msg: CameraInfo) -> None:
        calib = self.calib.get(info_msg)
        if calib is None:
            return
        t0 = time.perf_counter()
        image = mu.msg_to_image(image_msg)
        points = mu.msg_to_points(points_msg)
        uv, depth, mask = project_velo_to_image(points[:, :3], calib, image.shape[:2])
        overlay = draw_projected_points(image, uv, depth, radius=self.radius)
        self.pub.publish(mu.image_to_msg(overlay, image_msg.header))

        self._n += 1
        self._ms_sum += (time.perf_counter() - t0) * 1000.0
        if self._n % 50 == 0:
            self.get_logger().info(f"stamp {mu.stamp_to_str(image_msg.header.stamp)}: {int(mask.sum())}/{len(points)} 점 투영, "
                                   f"최근 50 프레임 평균 {self._ms_sum / 50:.1f} ms")
            self._ms_sum = 0.0


def main(args=None) -> None:
    # 종료 규약: Ctrl+C(SIGINT) 를 받으면 rclpy 시그널 핸들러가 컨텍스트를 닫고 spin 이 ExternalShutdownException 을 던진다.
    # 터미널 Ctrl+C 는 프로세스 그룹 전체에 SIGINT 를 보내고 ros2 launch 도 자식에게 SIGINT 를 한 번 더 전달하므로,
    # 컨텍스트를 닫은 뒤에는 SIGINT 를 무시해 두 번째 신호가 인터프리터 종료(atexit) 를 KeyboardInterrupt 로 깨지 않게 한다.
    try:
        rclpy.init(args=args)
        node = LidarProjectionNode()
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
