#!/usr/bin/env bash
# Phase 9: 파이프라인을 headless 로 띄우고 프레임 0..N-1 (기본 30) 을 rosbag2 로 기록한다.
#   scripts/record_bag.sh [N_FRAMES] [RATE_HZ]
# 산출: outputs/phase9/bag_sample/ (.gitignore 대상), outputs/phase9/bag_info.txt (ros2 bag info 결과)
set -euo pipefail
N_FRAMES="${1:-30}"
RATE_HZ="${2:-10.0}"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_DIR="$REPO/outputs/phase9"
BAG="$OUT_DIR/bag_sample"
START_DELAY=6.0

set +u
source /opt/ros/jazzy/setup.bash
[ -f "$HOME/ros2_jazzy/install/setup.bash" ] && source "$HOME/ros2_jazzy/install/setup.bash"
source "$REPO/ros2_ws/install/setup.bash"
set -u
# RMW 기본값: CycloneDDS. Fast DDS 는 1.4~2 MB 메시지(이미지·점군) 에서 약 55 프레임마다 1 프레임을 잃었다 (outputs/phase9/rmw_compare_*.log).
export RMW_IMPLEMENTATION="${RMW_IMPLEMENTATION:-rmw_cyclonedds_cpp}"
echo "[perceptrack3d] RMW_IMPLEMENTATION=$RMW_IMPLEMENTATION (Fast DDS 는 대용량 토픽에서 약 2 % 프레임 손실, CycloneDDS 권장)"
mkdir -p "$OUT_DIR"
rm -rf "$BAG"

# set -m: 잡 컨트롤을 켜야 백그라운드 launch 와 그 자식 노드가 SIGINT 를 무시(SIG_IGN 상속)하지 않고 1 s 안에 종료된다.
set -m
ros2 launch perceptrack3d_ros perceptrack3d.launch.py use_rviz:=false loop:=false \
    start_frame:=0 end_frame:=$((N_FRAMES - 1)) rate_hz:="$RATE_HZ" start_delay_s:="$START_DELAY" &
LAUNCH_PID=$!
cleanup() {
    # 프로세스 그룹(launch + 노드) 전체에 SIGINT → 노드는 rclpy 핸들러로 정상 종료. launch 는 백그라운드 작업이라
    # SIGINT 를 무시하고 SIGTERM 에는 자식 정리 없이 죽으므로, 노드가 끝난 뒤 TERM 을 보낸다 (scripts/ros2_verify.sh 와 동일).
    kill -INT -- "-$LAUNCH_PID" 2>/dev/null || true
    sleep 2
    kill -TERM "$LAUNCH_PID" 2>/dev/null || true
    wait "$LAUNCH_PID" 2>/dev/null || true
    pkill -f "[p]erceptrack3d_ros/" 2>/dev/null || true
}
trap cleanup EXIT

# 재생(N/rate) + 시작 지연 + 하위 노드 처리 여유(4 s) 만큼 기록한 뒤 SIGINT 로 정상 종료
DURATION=$(python3 -c "print(int($N_FRAMES / $RATE_HZ + $START_DELAY + 4))")
timeout -s INT "$DURATION" ros2 bag record -o "$BAG" \
    /kitti/image_raw /kitti/camera_info /kitti/velodyne_points /kitti/frame_id /tf_static \
    /perceptrack3d/detections_2d /perceptrack3d/projection_image /perceptrack3d/objects_3d \
    /perceptrack3d/markers_objects /perceptrack3d/tracks /perceptrack3d/markers_tracks \
    /perceptrack3d/annotated_image || true

ros2 bag info "$BAG" | tee "$OUT_DIR/bag_info.txt"
echo "bag: $BAG"
