"""PercepTrack3D 전체 파이프라인 launch (Phase 9, 팀 H).

    ros2 launch perceptrack3d_ros perceptrack3d.launch.py [use_rviz:=false] [method:=raw] [use_offline_json:=true]
                                                          [rate_hz:=5.0] [loop:=true] [start_frame:=0] [end_frame:=29]
                                                          [omp_threads:=4]

노드 6 개를 띄우고 config/params.yaml 을 읽은 뒤, launch 인자로 받은 값으로 덮어쓴다.
config_path 기본값 = 저장소의 configs/kitti.yaml. `ros2 launch` 는 시스템 python 으로 돌기 때문에 perceptrack3d 를 import
할 수 없으므로, 이 파일의 실제 위치(--symlink-install 이면 저장소 안)에서 위로 올라가며 configs/kitti.yaml 을 찾는다.
"""
from __future__ import annotations

from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def _find_repo_config() -> str:
    """이 launch 파일에서 위로 올라가며 configs/kitti.yaml 을 찾는다 (symlink-install / 일반 install 모두 지원)."""
    here = Path(__file__).resolve()
    for parent in here.parents:
        candidate = parent / "configs" / "kitti.yaml"
        if candidate.is_file():
            return str(candidate)
    return "configs/kitti.yaml"     # 못 찾으면 상대경로 (노드가 FileNotFoundError 로 알려 준다)


def generate_launch_description() -> LaunchDescription:
    share = Path(get_package_share_directory("perceptrack3d_ros"))
    args = [
        DeclareLaunchArgument("config_path", default_value=_find_repo_config(), description="configs/kitti.yaml 경로"),
        DeclareLaunchArgument("params_file", default_value=str(share / "config" / "params.yaml")),
        DeclareLaunchArgument("use_rviz", default_value="true"),
        DeclareLaunchArgument("rviz_config", default_value=str(share / "rviz" / "perceptrack3d.rviz")),
        DeclareLaunchArgument("method", default_value="clustered", description="fusion: raw | clustered"),
        DeclareLaunchArgument("use_offline_json", default_value="false", description="detector: YOLO 대신 detections.json"),
        DeclareLaunchArgument("rate_hz", default_value="10.0"),
        DeclareLaunchArgument("loop", default_value="false"),
        DeclareLaunchArgument("start_frame", default_value="0"),
        DeclareLaunchArgument("end_frame", default_value="-1"),
        DeclareLaunchArgument("start_delay_s", default_value="5.0"),
        DeclareLaunchArgument("omp_threads", default_value="4",
                              description="노드별 OMP/BLAS 스레드 수. 6 개 프로세스가 코어를 전부 잡으면 스핀 대기 경합으로 5~10 배 느려진다"),
    ]
    # 인자 문자열을 노드가 declare 한 타입으로 강제한다. 감싸지 않으면 "rate_hz:=10" 은 int 로 해석되어
    # double 파라미터에 들어가며 rclpy 가 InvalidParameterTypeException 으로 노드를 죽인다.
    def arg(name: str, value_type):
        return ParameterValue(LaunchConfiguration(name), value_type=value_type)

    common = {"config_path": arg("config_path", str), "rate_hz": arg("rate_hz", float)}
    params_file = LaunchConfiguration("params_file")
    # torch / open3d / numpy(OpenBLAS) 가 각각 코어 수만큼 스레드를 만들지 않도록 프로세스마다 상한을 건다.
    thread_env = {name: LaunchConfiguration("omp_threads")
                  for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")}

    def node(executable: str, extra: dict | None = None) -> Node:
        return Node(
            package="perceptrack3d_ros", executable=executable, name=executable, output="screen",
            parameters=[params_file, {**common, **(extra or {})}],
            additional_env=thread_env,
        )

    nodes = [
        node("kitti_player_node", {
            "loop": arg("loop", bool),
            "start_frame": arg("start_frame", int),
            "end_frame": arg("end_frame", int),
            "start_delay_s": arg("start_delay_s", float),
        }),
        node("detector_node", {"use_offline_json": arg("use_offline_json", bool)}),
        node("lidar_projection_node"),
        node("fusion_node", {"method": arg("method", str)}),
        node("tracker_node"),
        node("visualization_node"),
        Node(
            package="rviz2", executable="rviz2", name="rviz2", output="log",
            arguments=["-d", LaunchConfiguration("rviz_config")],
            condition=IfCondition(LaunchConfiguration("use_rviz")),
        ),
    ]
    return LaunchDescription(args + nodes)
