"""TF + CameraInfo 로 KittiCalibration 을 만드는 헬퍼 (Phase 9, 팀 H).

하위 노드(projection, fusion, visualization)는 캘리브레이션 텍스트 파일을 직접 읽지 않는다.
ROS 방식대로 (1) `/kitti/camera_info` 의 P 행렬과 (2) TF 트리의 `camera_02 ← velodyne` 변환에서 얻는다.
플레이어가 발행한 정적 TF 는 latched(transient_local) 라서 늦게 켜진 노드도 받을 수 있다.

lookup_transform(target_frame='camera_02', source_frame='velodyne') 이 돌려주는 변환은
"source(velodyne) 좌표의 점을 target(camera_02) 좌표로 옮기는" T_camera_02←velodyne 이고,
이것이 바로 calib 파일의 T_velo_to_rect 와 같아야 한다 (docs/learning_notes/phase9_ros2.md 에서 검증).
"""
from __future__ import annotations

import rclpy.time
from rclpy.node import Node
from sensor_msgs.msg import CameraInfo
from tf2_ros import LookupException, ConnectivityException, ExtrapolationException
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener

from perceptrack3d.geometry.calibration import KittiCalibration

from perceptrack3d_ros import msg_utils as mu


class TfCalibration:
    """정적 캘리브레이션이므로 한 번 성공하면 결과를 캐시한다."""

    def __init__(self, node: Node, camera_frame: str = "camera_02", lidar_frame: str = "velodyne"):
        self.node = node
        self.camera_frame = camera_frame
        self.lidar_frame = lidar_frame
        self.buffer = Buffer()
        self.listener = TransformListener(self.buffer, node, spin_thread=False)
        self._calib: KittiCalibration | None = None

    def get(self, cam_info: CameraInfo) -> KittiCalibration | None:
        """TF 가 아직 없으면 None (호출 쪽에서 그 프레임을 건너뛴다)."""
        if self._calib is not None:
            return self._calib
        try:
            tf_msg = self.buffer.lookup_transform(self.camera_frame, self.lidar_frame, rclpy.time.Time())
        except (LookupException, ConnectivityException, ExtrapolationException) as e:
            self.node.get_logger().warning(f"TF {self.camera_frame}←{self.lidar_frame} 대기 중: {e}", throttle_duration_sec=2.0)
            return None
        T_velo_to_cam = mu.matrix_from_transform_msg(tf_msg)
        self._calib = mu.calibration_from_ros(cam_info, T_velo_to_cam)
        t = T_velo_to_cam[:3, 3]
        self.node.get_logger().info(
            f"캘리브레이션 확보: P from CameraInfo ({cam_info.width}x{cam_info.height}), "
            f"T_{self.camera_frame}←{self.lidar_frame} translation=({t[0]:.4f}, {t[1]:.4f}, {t[2]:.4f})")
        return self._calib
