#!/usr/bin/env bash
# Phase 9: ROS2 환경 source → colcon build → launch 를 한 번에.
#   scripts/ros2_demo.sh                      # RViz 포함 전체 파이프라인
#   scripts/ros2_demo.sh use_rviz:=false      # headless
#   scripts/ros2_demo.sh method:=raw use_offline_json:=true rate_hz:=5.0 loop:=true
# 주의: 워크스페이스는 저장소 안의 ros2_ws/ 이다 (~/ros2_ws 가 아님).
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WS="$REPO/ros2_ws"

set +u
source /opt/ros/jazzy/setup.bash
[ -f "$HOME/ros2_jazzy/install/setup.bash" ] && source "$HOME/ros2_jazzy/install/setup.bash"
set -u
# RMW 기본값: CycloneDDS. Fast DDS 는 1.4~2 MB 메시지(이미지·점군) 에서 약 55 프레임마다 1 프레임을 잃었다 (outputs/phase9/rmw_compare_*.log).
export RMW_IMPLEMENTATION="${RMW_IMPLEMENTATION:-rmw_cyclonedds_cpp}"
echo "[perceptrack3d] RMW_IMPLEMENTATION=$RMW_IMPLEMENTATION (Fast DDS 는 대용량 토픽에서 약 2 % 프레임 손실, CycloneDDS 권장)"
# venv 를 활성화한 채 venv 의 colcon 으로 빌드해야 노드 스크립트 shebang 이 .venv/bin/python 을 가리킨다.
source "$REPO/.venv/bin/activate"
command -v colcon >/dev/null || python -m pip install --isolated --index-url https://pypi.org/simple colcon-common-extensions

(cd "$WS" && colcon build --symlink-install)
set +u; source "$WS/install/setup.bash"; set -u
exec ros2 launch perceptrack3d_ros perceptrack3d.launch.py "$@"
