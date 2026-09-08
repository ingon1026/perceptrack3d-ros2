"""kitti_player_node — KITTI 동기화 시퀀스를 ROS2 센서 토픽으로 재생한다 (Phase 9, 팀 H).

발행 (모두 같은 stamp, stamp = frame_id / rate_hz 초)
    /kitti/image_raw        sensor_msgs/Image        frame_id = camera_02  (bgr8, 375x1242)
    /kitti/camera_info      sensor_msgs/CameraInfo   frame_id = camera_02  (K, P = P_rect_02, R = R_rect_00)
    /kitti/velodyne_points  sensor_msgs/PointCloud2  frame_id = velodyne   (x, y, z, intensity float32)
    /kitti/frame_id         std_msgs/Int32           (사람이 echo 로 보기 위한 프레임 번호)

정적 TF (StaticTransformBroadcaster, latched)
    base_link → velodyne   : 항등 회전, z = +1.73 m (KITTI setup 페이지의 Velodyne 높이)
    velodyne  → camera_02  : T_velo_cam = inv(T_velo_to_rect)
        tf2 규약: header.frame_id(부모) = velodyne, child_frame_id(자식) = camera_02 일 때
        transform 은 "자식(camera_02) 좌표의 점을 부모(velodyne) 좌표로 옮기는" T_velodyne←camera_02 이다.
        calib 파일의 T_velo_to_rect 는 반대 방향(velodyne 점 → rect 카메라 좌표)이므로 역행렬을 넣는다.
        translation 은 곧 "camera_02 원점의 velodyne 좌표" (≈ x +0.27 m 앞, z −0.07 m 아래).
        검증: `ros2 run tf2_ros tf2_echo camera_02 velodyne` 의 translation 이 calib 의 T_velo_to_rect[:3, 3] 과 같아야 한다.

파라미터
    config_path (str)   configs/kitti.yaml 경로
    rate_hz (float)     재생 주기 (기본 10 = KITTI 원래 주기)
    loop (bool)         끝나면 처음부터 반복
    start_frame (int)   시작 프레임, end_frame (int) 끝 프레임 (포함, -1 = 마지막)
    start_delay_s (float) 첫 프레임 발행 전 대기 시간 (하위 노드 준비용)
"""
from __future__ import annotations

import signal

import numpy as np
import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from sensor_msgs.msg import CameraInfo, Image, PointCloud2
from std_msgs.msg import Header, Int32
from tf2_ros.static_transform_broadcaster import StaticTransformBroadcaster

from perceptrack3d.config import DEFAULT_CONFIG, load_config
from perceptrack3d.data.kitti_loader import KittiDataset
from perceptrack3d.geometry.calibration import KittiCalibration
from perceptrack3d.geometry.transforms import invert_rigid

from perceptrack3d_ros import msg_utils as mu

LIDAR_FRAME = "velodyne"
CAMERA_FRAME = "camera_02"
BASE_FRAME = "base_link"
VELODYNE_HEIGHT_M = 1.73     # https://www.cvlibs.net/datasets/kitti/setup.php


class KittiPlayerNode(Node):
    def __init__(self) -> None:
        super().__init__("kitti_player_node")
        self.declare_parameter("config_path", str(DEFAULT_CONFIG))
        self.declare_parameter("rate_hz", 10.0)
        self.declare_parameter("loop", False)
        self.declare_parameter("start_frame", 0)
        self.declare_parameter("end_frame", -1)
        self.declare_parameter("start_delay_s", 5.0)

        cfg = load_config(self.get_parameter("config_path").value)
        self.rate_hz = float(self.get_parameter("rate_hz").value)
        self.loop = bool(self.get_parameter("loop").value)
        self.start_delay_s = float(self.get_parameter("start_delay_s").value)
        self.dataset = KittiDataset(cfg)
        self.calib = KittiCalibration.from_dir(cfg["dataset"]["calib_dir"])

        ids = self.dataset.frame_ids()
        start = int(self.get_parameter("start_frame").value)
        end = int(self.get_parameter("end_frame").value)
        end = ids[-1] if end < 0 else end
        self.frame_ids = [f for f in ids if start <= f <= end]
        if not self.frame_ids:
            raise ValueError(f"재생할 프레임이 없습니다: start={start}, end={end}, 데이터셋 {ids[0]}..{ids[-1]}")
        self.cursor = 0

        self.pub_image = self.create_publisher(Image, "/kitti/image_raw", 10)
        self.pub_info = self.create_publisher(CameraInfo, "/kitti/camera_info", 10)
        self.pub_points = self.create_publisher(PointCloud2, "/kitti/velodyne_points", 10)
        self.pub_frame = self.create_publisher(Int32, "/kitti/frame_id", 10)

        self._publish_static_tf()
        # 하위 노드(YOLO 로드 ~3 s, open3d import) 가 구독을 시작할 때까지 기다린 뒤 재생 타이머를 켠다.
        self.timer = None
        self.delay_timer = self.create_timer(max(self.start_delay_s, 0.01), self._start_playback)
        self.get_logger().info(
            f"KITTI {cfg['dataset']['date']}_drive_{cfg['dataset']['drive']}: 프레임 {self.frame_ids[0]}..{self.frame_ids[-1]} "
            f"({len(self.frame_ids)} 개) 를 {self.rate_hz} Hz 로 재생, loop={self.loop}, {self.start_delay_s} s 뒤 시작")

    def _start_playback(self) -> None:
        self.delay_timer.cancel()
        self.timer = self.create_timer(1.0 / self.rate_hz, self._tick)

    def _publish_static_tf(self) -> None:
        stamp = self.get_clock().now().to_msg()
        T_base_velo = np.eye(4)
        T_base_velo[2, 3] = VELODYNE_HEIGHT_M
        T_velo_cam = invert_rigid(self.calib.T_velo_to_rect)      # camera_02 좌표 → velodyne 좌표
        self.static_tf = StaticTransformBroadcaster(self)
        self.static_tf.sendTransform([
            mu.transform_msg_from_matrix(T_base_velo, BASE_FRAME, LIDAR_FRAME, stamp),
            mu.transform_msg_from_matrix(T_velo_cam, LIDAR_FRAME, CAMERA_FRAME, stamp),
        ])
        t = T_velo_cam[:3, 3]
        self.get_logger().info(
            f"정적 TF 발행: {BASE_FRAME}→{LIDAR_FRAME} (z={VELODYNE_HEIGHT_M} m), "
            f"{LIDAR_FRAME}→{CAMERA_FRAME} translation=({t[0]:.4f}, {t[1]:.4f}, {t[2]:.4f}) m")

    def _tick(self) -> None:
        if self.cursor >= len(self.frame_ids):
            if self.loop:
                self.cursor = 0
                self.get_logger().info("시퀀스 끝 → 처음부터 반복 (stamp 도 0 부터 다시 시작)")
            else:
                self.get_logger().info("시퀀스 끝. 재생을 멈춥니다 (노드는 TF 를 위해 살아 있음).")
                self.timer.cancel()
                return
        frame_id = self.frame_ids[self.cursor]
        self.cursor += 1

        frame = self.dataset.load_frame(frame_id)
        stamp = mu.stamp_from_frame(frame_id, self.rate_hz)
        cam_header = Header(stamp=stamp, frame_id=CAMERA_FRAME)
        lidar_header = Header(stamp=stamp, frame_id=LIDAR_FRAME)

        # 순서: frame_id → camera_info → points → image (구독 쪽 동기화는 stamp 로 하므로 순서는 중요하지 않다)
        self.pub_frame.publish(Int32(data=int(frame_id)))
        self.pub_info.publish(mu.camera_info_from_calib(self.calib, cam_header))
        self.pub_points.publish(mu.points_to_msg(frame["points"], lidar_header))
        self.pub_image.publish(mu.image_to_msg(frame["image"], cam_header))
        if frame_id % 50 == 0:
            self.get_logger().info(f"frame {frame_id}: stamp={mu.stamp_to_str(stamp)}, points={len(frame['points'])}")


def main(args=None) -> None:
    # 종료 규약: Ctrl+C(SIGINT) 를 받으면 rclpy 시그널 핸들러가 컨텍스트를 닫고 spin 이 ExternalShutdownException 을 던진다.
    # 터미널 Ctrl+C 는 프로세스 그룹 전체에 SIGINT 를 보내고 ros2 launch 도 자식에게 SIGINT 를 한 번 더 전달하므로,
    # 컨텍스트를 닫은 뒤에는 SIGINT 를 무시해 두 번째 신호가 인터프리터 종료(atexit) 를 KeyboardInterrupt 로 깨지 않게 한다.
    try:
        rclpy.init(args=args)
        node = KittiPlayerNode()
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
