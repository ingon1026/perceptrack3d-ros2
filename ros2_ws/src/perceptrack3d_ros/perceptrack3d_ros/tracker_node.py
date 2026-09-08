"""tracker_node — 3D 객체를 칼만 필터로 추적한다 (Phase 9, 팀 H).

구독  /perceptrack3d/objects_3d       vision_msgs/Detection3DArray (velodyne 프레임)
발행  /perceptrack3d/tracks           vision_msgs/Detection3DArray  id = track_id (문자열), class_id = 클래스, 위치 = 칼만 상태
      /perceptrack3d/markers_tracks   visualization_msgs/MarkerArray  CUBE(트랙 id 색) + "ID n" TEXT + LINE_STRIP 궤적(history)

KalmanTracker.step() 은 프레임 순서대로 매 프레임 호출한다는 가정이라, 프레임이 뒤로 가면(loop 재생) 추적기를 새로 만든다.
프레임을 건너뛰면(하위 노드가 느려 drop) dt 가 실제보다 짧게 가정되므로 예측 거리가 짧아진다 — docs 의 알려진 제한.

파라미터: config_path, rate_hz, marker_lifetime_s, trail_length (궤적 표시 프레임 수)
"""
from __future__ import annotations

import signal

import numpy as np
import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from std_msgs.msg import Header
from vision_msgs.msg import Detection3DArray
from visualization_msgs.msg import MarkerArray

from perceptrack3d.config import DEFAULT_CONFIG, load_config
from perceptrack3d.tracking.kalman_tracker import KalmanTracker

from perceptrack3d_ros import msg_utils as mu


class TrackerNode(Node):
    def __init__(self) -> None:
        super().__init__("tracker_node")
        self.declare_parameter("config_path", str(DEFAULT_CONFIG))
        self.declare_parameter("rate_hz", 10.0)
        self.declare_parameter("marker_lifetime_s", 0.15)
        self.declare_parameter("trail_length", 30)
        self.cfg = load_config(self.get_parameter("config_path").value)
        self.rate_hz = float(self.get_parameter("rate_hz").value)
        self.lifetime = float(self.get_parameter("marker_lifetime_s").value)
        self.trail = int(self.get_parameter("trail_length").value)

        self.tracker = KalmanTracker(self.cfg)
        self.last_frame_id = -1
        self.pub_tracks = self.create_publisher(Detection3DArray, "/perceptrack3d/tracks", 10)
        self.pub_markers = self.create_publisher(MarkerArray, "/perceptrack3d/markers_tracks", 10)
        self.sub = self.create_subscription(Detection3DArray, "/perceptrack3d/objects_3d", self._on_objects, 10)
        t = self.cfg["tracking"]
        self.get_logger().info(f"KalmanTracker: gating={t['gating']}, max_misses={t['max_misses']}, min_hits={t['min_hits']}, dt={t['dt']}")

    def _on_objects(self, msg: Detection3DArray) -> None:
        frame_id = mu.frame_from_stamp(msg.header.stamp, self.rate_hz)
        if frame_id < self.last_frame_id:
            self.get_logger().info(f"프레임이 {self.last_frame_id} → {frame_id} 로 되돌아감 (loop). 추적기 초기화")
            self.tracker = KalmanTracker(self.cfg)
        elif self.last_frame_id >= 0 and frame_id != self.last_frame_id + 1:
            self.get_logger().warning(f"프레임 건너뜀 {self.last_frame_id} → {frame_id}: 등속 예측 dt 가 실제보다 짧음",
                                      throttle_duration_sec=5.0)
        self.last_frame_id = frame_id

        objects = mu.msg_to_objects(msg, frame_id)
        tracks = self.tracker.step(objects, frame_id)

        header = Header(stamp=msg.header.stamp, frame_id=msg.header.frame_id)
        self.pub_tracks.publish(mu.tracks_to_msg(tracks, header))
        self.pub_markers.publish(self._markers(tracks, header))
        if frame_id % 50 == 0:
            self.get_logger().info(f"frame {frame_id}: 관측 {len(objects)} → 확정 트랙 {len(tracks)} "
                                   f"(ids {[t.track_id for t in tracks]})")

    def _markers(self, tracks, header: Header) -> MarkerArray:
        arr = MarkerArray()
        for t in tracks:
            tid = int(t.track_id)
            center = np.asarray(t.state[:3], dtype=float)
            size = mu.size_or_default(t.size, t.class_name)
            arr.markers.append(mu.cube_marker(header, "track_box", tid, center, size, 0.0, mu.color_for_id(tid, 0.5), self.lifetime))
            text_pos = np.array([center[0], center[1], center[2] + size[2] / 2 + 0.6])
            arr.markers.append(mu.text_marker(header, "track_text", tid, text_pos, f"ID {tid}", mu.color_for_id(tid, 1.0), self.lifetime))
            hist = np.asarray([h[1:4] for h in t.history[-self.trail:]], dtype=float)
            if len(hist) >= 2:
                arr.markers.append(mu.line_strip_marker(header, "track_path", tid, hist, mu.color_for_id(tid, 0.9), self.lifetime))
        return arr


def main(args=None) -> None:
    # 종료 규약: Ctrl+C(SIGINT) 를 받으면 rclpy 시그널 핸들러가 컨텍스트를 닫고 spin 이 ExternalShutdownException 을 던진다.
    # 터미널 Ctrl+C 는 프로세스 그룹 전체에 SIGINT 를 보내고 ros2 launch 도 자식에게 SIGINT 를 한 번 더 전달하므로,
    # 컨텍스트를 닫은 뒤에는 SIGINT 를 무시해 두 번째 신호가 인터프리터 종료(atexit) 를 KeyboardInterrupt 로 깨지 않게 한다.
    try:
        rclpy.init(args=args)
        node = TrackerNode()
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
