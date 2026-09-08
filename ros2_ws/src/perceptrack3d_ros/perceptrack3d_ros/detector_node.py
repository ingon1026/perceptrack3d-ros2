"""detector_node — image_02 프레임에 YOLO 2D 검출을 돌린다 (Phase 9, 팀 H).

구독  /kitti/image_raw               sensor_msgs/Image
발행  /perceptrack3d/detections_2d   vision_msgs/Detection2DArray (header = 입력 header 그대로: 같은 stamp, frame_id=camera_02)
        detections[i].bbox.center.position (박스 중심 px), size_x/size_y (px)
        detections[i].results[0].hypothesis.class_id = COCO 이름, .score = confidence

파라미터
    config_path        configs/kitti.yaml
    rate_hz            stamp → frame_id 복원용 (플레이어와 같은 값)
    use_offline_json   True 면 YOLO 대신 outputs/phase4/detections.json 에서 frame_id 로 조회 (CPU 부하 없는 데모)
    offline_json_path  "" 이면 {outputs.dir}/phase4/detections.json
    torch_threads      torch.set_num_threads 값 (기본 4; 다른 노드와의 CPU 경합 완화)
"""
from __future__ import annotations

import signal
import time
from pathlib import Path

import rclpy
from rclpy.executors import ExternalShutdownException
import torch
from rclpy.node import Node
from sensor_msgs.msg import Image
from vision_msgs.msg import Detection2DArray

from perceptrack3d.config import DEFAULT_CONFIG, load_config
from perceptrack3d.detection.detector import YoloDetector, load_detections_json

from perceptrack3d_ros import msg_utils as mu


class DetectorNode(Node):
    def __init__(self) -> None:
        super().__init__("detector_node")
        self.declare_parameter("config_path", str(DEFAULT_CONFIG))
        self.declare_parameter("rate_hz", 10.0)
        self.declare_parameter("use_offline_json", False)
        self.declare_parameter("offline_json_path", "")
        self.declare_parameter("torch_threads", 4)
        cfg = load_config(self.get_parameter("config_path").value)
        self.rate_hz = float(self.get_parameter("rate_hz").value)
        # 노드 6 개가 한 CPU 를 나눠 쓰므로 torch 가 코어 전부를 잡으면 OpenMP 스레드 경합으로 오히려 느려진다.
        torch.set_num_threads(int(self.get_parameter("torch_threads").value))

        self.offline: dict | None = None
        self.detector: YoloDetector | None = None
        if bool(self.get_parameter("use_offline_json").value):
            path = self.get_parameter("offline_json_path").value or str(Path(cfg["outputs"]["dir"]) / "phase4" / "detections.json")
            self.offline = load_detections_json(path)
            self.get_logger().info(f"오프라인 검출 사용: {path} ({len(self.offline)} 프레임)")
        else:
            t0 = time.perf_counter()
            self.detector = YoloDetector(cfg)          # 모델 로드 + warm-up
            self.get_logger().info(f"YOLO 로드+warm-up {time.perf_counter() - t0:.1f} s: {self.detector.model_path.name}, "
                                   f"device={self.detector.device}, imgsz={self.detector.imgsz}")

        self.pub = self.create_publisher(Detection2DArray, "/perceptrack3d/detections_2d", 10)
        self.sub = self.create_subscription(Image, "/kitti/image_raw", self._on_image, 10)
        self._n, self._ms_sum = 0, 0.0

    def _on_image(self, msg: Image) -> None:
        frame_id = mu.frame_from_stamp(msg.header.stamp, self.rate_hz)
        if self.offline is not None:
            dets = self.offline.get(frame_id, [])
            elapsed_ms = 0.0
        else:
            image = mu.msg_to_image(msg)
            dets, elapsed_ms = self.detector.detect_timed(image, frame_id)
        self.pub.publish(mu.detections_to_msg(dets, msg.header))

        self._n += 1
        self._ms_sum += elapsed_ms
        if self._n % 50 == 0:
            self.get_logger().info(f"frame {frame_id}: {len(dets)} 검출, 최근 50 프레임 평균 추론 {self._ms_sum / 50:.1f} ms")
            self._ms_sum = 0.0


def main(args=None) -> None:
    # 종료 규약: Ctrl+C(SIGINT) 를 받으면 rclpy 시그널 핸들러가 컨텍스트를 닫고 spin 이 ExternalShutdownException 을 던진다.
    # 터미널 Ctrl+C 는 프로세스 그룹 전체에 SIGINT 를 보내고 ros2 launch 도 자식에게 SIGINT 를 한 번 더 전달하므로,
    # 컨텍스트를 닫은 뒤에는 SIGINT 를 무시해 두 번째 신호가 인터프리터 종료(atexit) 를 KeyboardInterrupt 로 깨지 않게 한다.
    try:
        rclpy.init(args=args)
        node = DetectorNode()
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
