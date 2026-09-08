"""Phase 9 headless 검증: 파이프라인이 떠 있는 동안 토픽을 구독해 발행률·stamp 일관성·값 범위를 표로 만든다.

실행 (파이프라인을 다른 터미널에서 띄운 뒤):
    source ros2_ws/install/setup.bash
    .venv/bin/python scripts/ros2_check_pipeline.py --seconds 20 [--rate-hz 10] [--out outputs/phase9/verification_topics.md]

검사 항목
- 토픽별 메시지 수와 평균 Hz (측정 구간 동안)
- 모든 토픽의 header.stamp 가 플레이어 stamp 집합(frame_id / rate_hz) 에 속하는지, 프레임별로 어느 토픽이 빠졌는지
- objects_3d / tracks 의 frame_id == "velodyne", 중심 좌표 범위(x 0~70 m, |y| ≤ 40 m, |z| ≤ 3 m) 검사
- /tf_static 의 velodyne→camera_02 변환이 calib 파일의 inv(T_velo_to_rect) 와 일치하는지
"""
from __future__ import annotations

import argparse
import struct
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import rclpy
from builtin_interfaces.msg import Time as TimeMsg
from rclpy.node import Node
from rclpy.qos import QoSProfile, DurabilityPolicy, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import CameraInfo, Image, PointCloud2
from tf2_msgs.msg import TFMessage
from vision_msgs.msg import Detection2DArray, Detection3DArray
from visualization_msgs.msg import MarkerArray

from perceptrack3d.config import load_config
from perceptrack3d.geometry.calibration import KittiCalibration
from perceptrack3d.geometry.transforms import invert_rigid

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ros2_ws" / "src" / "perceptrack3d_ros"))
from perceptrack3d_ros import msg_utils as mu  # noqa: E402

TOPICS = {
    "/kitti/image_raw": Image,
    "/kitti/camera_info": CameraInfo,
    "/kitti/velodyne_points": PointCloud2,
    "/perceptrack3d/detections_2d": Detection2DArray,
    "/perceptrack3d/projection_image": Image,
    "/perceptrack3d/objects_3d": Detection3DArray,
    "/perceptrack3d/markers_objects": MarkerArray,
    "/perceptrack3d/tracks": Detection3DArray,
    "/perceptrack3d/markers_tracks": MarkerArray,
    "/perceptrack3d/annotated_image": Image,
}
HEAVY_TOPICS = {"/kitti/image_raw", "/kitti/velodyne_points", "/perceptrack3d/projection_image", "/perceptrack3d/annotated_image"}


class Checker(Node):
    def __init__(self, rate_hz: float):
        super().__init__("phase9_checker")
        self.rate_hz = rate_hz
        self.count: dict[str, int] = defaultdict(int)
        self.first_t: dict[str, float] = {}
        self.last_t: dict[str, float] = {}
        self.stamps: dict[str, set] = defaultdict(set)
        self.frame_ids: dict[str, list] = defaultdict(list)
        self.samples: dict[str, object] = {}
        self.bad_frame: dict[str, int] = defaultdict(int)
        self.range_violations = 0
        self.n_objects = 0
        self.tf_static = None
        # 검사 스크립트 자신이 병목이 되지 않도록: depth 50, 그리고 대용량 토픽(이미지 1.4 MB ×3, 점군 2 MB) 은 raw=True 로
        # 직렬화된 CDR 바이트만 받아 header.stamp 를 struct 로 읽는다 (역직렬화 생략). 파이썬 구독자가 10 토픽을 전부
        # 역직렬화하면 프레임당 6 MB 를 처리하느라 30~50% 를 놓쳐 발행률이 과소 측정됐다.
        for topic, mtype in TOPICS.items():
            if topic in HEAVY_TOPICS:
                self.create_subscription(mtype, topic, lambda b, t=topic: self._on_raw(t, b), 50, raw=True)
            else:
                self.create_subscription(mtype, topic, lambda m, t=topic: self._on_msg(t, m), 50)
        latched = QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE,
                             durability=DurabilityPolicy.TRANSIENT_LOCAL, history=HistoryPolicy.KEEP_LAST)
        self.create_subscription(TFMessage, "/tf_static", self._on_tf, latched)

    def _on_tf(self, msg: TFMessage) -> None:
        self.tf_static = {(t.header.frame_id, t.child_frame_id): mu.matrix_from_transform_msg(t) for t in msg.transforms}

    def _on_raw(self, topic: str, buf: bytes) -> None:
        """Header 로 시작하는 메시지의 CDR 바이트: [4 B 캡슐화 헤더][int32 sec][uint32 nanosec][string frame_id]..."""
        sec, nsec = struct.unpack_from("<iI", buf, 4)
        self._record(topic, sec, nsec)

    def _record(self, topic: str, sec: int, nsec: int) -> None:
        now = time.perf_counter()
        self.count[topic] += 1
        self.first_t.setdefault(topic, now)
        self.last_t[topic] = now
        self.stamps[topic].add((sec, nsec))
        self.frame_ids[topic].append(mu.frame_from_stamp(TimeMsg(sec=sec, nanosec=nsec), self.rate_hz))

    def _on_msg(self, topic: str, msg) -> None:
        header = msg.header if hasattr(msg, "header") else (msg.markers[0].header if msg.markers else None)
        if header is None:
            self.count[topic] += 1
            return
        self._record(topic, header.stamp.sec, header.stamp.nanosec)
        if topic in ("/perceptrack3d/objects_3d", "/perceptrack3d/tracks"):
            if header.frame_id != "velodyne":
                self.bad_frame[topic] += 1
            for d in msg.detections:
                p = d.bbox.center.position
                self.n_objects += 1
                if not (0.0 <= p.x <= 70.0 and abs(p.y) <= 40.0 and abs(p.z) <= 3.0):
                    self.range_violations += 1
            if msg.detections and topic not in self.samples:
                self.samples[topic] = (header, msg.detections[:3])

    def report(self, seconds: float, calib: KittiCalibration) -> str:
        lines = [f"# Phase 9 headless 검증 (자동 생성, 측정 {seconds:.0f} s, rate_hz={self.rate_hz})", ""]
        lines += ["## 토픽 발행률", "",
                  f"측정 창 {seconds:.0f} s 동안 받은 메시지 수 / 창 길이 = Hz. 플레이어 {self.rate_hz} Hz 기준 기대치 {self.rate_hz * seconds:.0f} 개.", "",
                  "| 토픽 | 메시지 수 | Hz (수/창) | 고유 stamp 수 | 프레임 범위 |", "|---|---|---|---|---|"]
        for topic in TOPICS:
            n = self.count[topic]
            fids = self.frame_ids[topic]
            rng = f"{min(fids)}..{max(fids)}" if fids else "-"
            lines.append(f"| `{topic}` | {n} | {n / seconds:.2f} | {len(self.stamps[topic])} | {rng} |")
        lines.append("")

        # stamp 일관성: 플레이어 stamp 집합 밖의 stamp 가 있는가, 프레임별 누락 토픽
        player = self.stamps["/kitti/image_raw"]
        lines += ["## stamp 일관성", ""]
        for topic in TOPICS:
            extra = self.stamps[topic] - player
            lines.append(f"- `{topic}`: 플레이어 stamp 집합 밖의 stamp {len(extra)} 개")
        common = set.intersection(*(self.stamps[t] for t in TOPICS if self.stamps[t])) if any(self.stamps.values()) else set()
        lines.append(f"- 플레이어 프레임 {len(player)} 개 중 모든 토픽이 발행된 프레임: {len(common & player)} 개 "
                     f"(나머지는 하위 노드가 처리 시간 부족으로 건너뜀)")
        lines.append("")

        lines += ["## objects_3d / tracks 값 검사", "",
                  f"- header.frame_id != 'velodyne' 인 메시지: {sum(self.bad_frame.values())} 개",
                  f"- 3D 박스 {self.n_objects} 개 중 범위(x 0~70, |y| ≤ 40, |z| ≤ 3 m) 위반: {self.range_violations} 개", ""]
        for topic in ("/perceptrack3d/objects_3d", "/perceptrack3d/tracks"):
            if topic in self.samples:
                header, dets = self.samples[topic]
                lines.append(f"### `{topic}` 샘플 (stamp {mu.stamp_to_str(header.stamp)} = frame {mu.frame_from_stamp(header.stamp, self.rate_hz)}, frame_id={header.frame_id})")
                lines.append("")
                lines.append("| id | class | score | center x,y,z (m) | size l,w,h (m) |")
                lines.append("|---|---|---|---|---|")
                for d in dets:
                    p, s = d.bbox.center.position, d.bbox.size
                    h = d.results[0].hypothesis if d.results else None
                    lines.append(f"| {d.id} | {h.class_id if h else '-'} | {h.score if h else 0:.2f} | {p.x:.2f}, {p.y:.2f}, {p.z:.2f} | {s.x:.2f}, {s.y:.2f}, {s.z:.2f} |")
                lines.append("")

        lines += ["## /tf_static vs calib 파일", ""]
        if self.tf_static is None:
            lines.append("- /tf_static 수신 실패")
        else:
            T_expected = invert_rigid(calib.T_velo_to_rect)
            T_tf = self.tf_static.get(("velodyne", "camera_02"))
            if T_tf is None:
                lines.append(f"- velodyne→camera_02 없음. 받은 쌍: {list(self.tf_static)}")
            else:
                err = np.abs(T_tf - T_expected).max()
                lines.append(f"- TF velodyne→camera_02 translation = {np.round(T_tf[:3, 3], 4).tolist()} (camera_02 원점의 velodyne 좌표)")
                lines.append(f"- inv(T_velo_to_rect) 와 최대 절대 오차 {err:.2e} → {'일치' if err < 1e-6 else '불일치'}")
                lines.append(f"- 반대 방향 T_camera_02←velodyne translation = {np.round(invert_rigid(T_tf)[:3, 3], 4).tolist()} = calib T_velo_to_rect[:3, 3] {np.round(calib.T_velo_to_rect[:3, 3], 4).tolist()}")
            Tb = self.tf_static.get(("base_link", "velodyne"))
            if Tb is not None:
                lines.append(f"- TF base_link→velodyne translation = {np.round(Tb[:3, 3], 3).tolist()}")
        lines.append("")
        return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=float, default=20.0)
    ap.add_argument("--rate-hz", type=float, default=10.0)
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    cfg = load_config()
    calib = KittiCalibration.from_dir(cfg["dataset"]["calib_dir"])

    rclpy.init()
    node = Checker(args.rate_hz)
    # 구독 매칭(DDS discovery) 이 끝날 때까지 2 s 예열한 뒤 카운터를 비우고 측정 창을 시작한다.
    # spin_once 루프 대신 spin_until_future_complete 로 executor 를 계속 돌려 검사 스크립트가 병목이 되지 않게 한다.
    from rclpy.task import Future
    warm = Future()
    node.create_timer(2.0, lambda: warm.set_result(True))
    rclpy.spin_until_future_complete(node, warm)
    node.count.clear(); node.first_t.clear(); node.last_t.clear(); node.stamps.clear(); node.frame_ids.clear()
    node.samples.clear(); node.bad_frame.clear(); node.range_violations = 0; node.n_objects = 0
    done = Future()
    node.create_timer(args.seconds, lambda: done.set_result(True))
    rclpy.spin_until_future_complete(node, done)
    text = node.report(args.seconds, calib)
    node.destroy_node()
    rclpy.shutdown()
    print(text)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
