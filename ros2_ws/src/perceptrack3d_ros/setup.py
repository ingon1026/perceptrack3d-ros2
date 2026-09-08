"""ament_python 패키지 설정. 노드 6개를 console_scripts 로 등록한다.

빌드는 반드시 저장소 .venv 를 활성화한 뒤 (.venv 의 colcon 으로) 실행해야
생성되는 실행 스크립트의 shebang 이 .venv/bin/python 을 가리킨다 (scripts/ros2_demo.sh 참고).
"""
from glob import glob

from setuptools import find_packages, setup

package_name = "perceptrack3d_ros"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        ("share/" + package_name + "/launch", glob("launch/*.launch.py")),
        ("share/" + package_name + "/config", glob("config/*.yaml")),
        ("share/" + package_name + "/rviz", glob("rviz/*.rviz")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="ingon",
    maintainer_email="k3i_ai2@k3i.co.kr",
    description="PercepTrack3D KITTI camera-LiDAR perception pipeline as ROS2 nodes",
    license="MIT",
    entry_points={
        "console_scripts": [
            "kitti_player_node = perceptrack3d_ros.kitti_player_node:main",
            "detector_node = perceptrack3d_ros.detector_node:main",
            "lidar_projection_node = perceptrack3d_ros.lidar_projection_node:main",
            "fusion_node = perceptrack3d_ros.fusion_node:main",
            "tracker_node = perceptrack3d_ros.tracker_node:main",
            "visualization_node = perceptrack3d_ros.visualization_node:main",
        ],
    },
)
