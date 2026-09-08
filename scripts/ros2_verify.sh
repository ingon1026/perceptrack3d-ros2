#!/usr/bin/env bash
# Phase 9 headless 검증을 한 번에: 파이프라인 launch → 토픽/stamp/값 검사 → tf2_echo → RViz 크래시 테스트 → 종료.
#   scripts/ros2_verify.sh [CHECK_SECONDS=30] [RATE_HZ=10.0]
# 산출: outputs/phase9/verification_topics.md, verification_tf.txt, verification_launch.log, verification_rviz.log
set -uo pipefail
CHECK_SECONDS="${1:-30}"
RATE_HZ="${2:-10.0}"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="$REPO/outputs/phase9"
mkdir -p "$OUT"

set +u
source /opt/ros/jazzy/setup.bash
[ -f "$HOME/ros2_jazzy/install/setup.bash" ] && source "$HOME/ros2_jazzy/install/setup.bash"
source "$REPO/ros2_ws/install/setup.bash"
set -u
# RMW 기본값: CycloneDDS. Fast DDS 는 1.4~2 MB 메시지(이미지·점군) 에서 약 55 프레임마다 1 프레임을 잃었다 (outputs/phase9/rmw_compare_*.log).
export RMW_IMPLEMENTATION="${RMW_IMPLEMENTATION:-rmw_cyclonedds_cpp}"
echo "[perceptrack3d] RMW_IMPLEMENTATION=$RMW_IMPLEMENTATION (Fast DDS 는 대용량 토픽에서 약 2 % 프레임 손실, CycloneDDS 권장)"

# set -m: 잡 컨트롤을 켜야 백그라운드 launch 와 그 자식 노드가 SIGINT 를 무시(SIG_IGN 상속)하지 않고 1 s 안에 종료된다.
set -m
ros2 launch perceptrack3d_ros perceptrack3d.launch.py use_rviz:=false loop:=true rate_hz:="$RATE_HZ" \
    > "$OUT/verification_launch.log" 2>&1 &
LAUNCH_PID=$!
cleanup() {
    # set -m 덕분에 launch 잡은 자기 PID 를 PGID 로 갖는 프로세스 그룹이다. 그룹 전체에 SIGINT 를 보내면
    # 노드들은 rclpy 핸들러로 1 s 안에 정상 종료한다. launch 자체는 non-interactive 셸의 백그라운드 작업이라
    # SIGINT 를 무시(SIG_IGN 상속)하고, SIGTERM 에는 자식 정리 없이 바로 죽으므로 노드 종료 후에 TERM 을 보낸다.
    kill -INT -- "-$LAUNCH_PID" 2>/dev/null || true
    sleep 2
    kill -TERM "$LAUNCH_PID" 2>/dev/null || true
    wait "$LAUNCH_PID" 2>/dev/null || true
    pkill -f "[p]erceptrack3d_ros/" 2>/dev/null || true   # 혹시 남은 노드. [p] 로 감싸 이 스크립트 자신의 명령줄과는 불일치
}
trap cleanup EXIT
echo "[verify] launch pid $LAUNCH_PID, 22 s 대기 (YOLO 로드 + start_delay)"
sleep 22

echo "[verify] 토픽 검사 ${CHECK_SECONDS} s"
OMP_NUM_THREADS=2 "$REPO/.venv/bin/python" "$REPO/scripts/ros2_check_pipeline.py" \
    --seconds "$CHECK_SECONDS" --rate-hz "$RATE_HZ" --out "$OUT/verification_topics.md"

{
    echo "# tf2_echo (정적 TF, 파이프라인 실행 중 측정)"
    for pair in "camera_02 velodyne" "velodyne camera_02" "base_link velodyne" "base_link camera_02"; do
        echo; echo "\$ ros2 run tf2_ros tf2_echo $pair"
        timeout 10 ros2 run tf2_ros tf2_echo $pair 2>&1 | grep -A2 "Translation" | head -3
    done
} | tee "$OUT/verification_tf.txt"

echo "[verify] rviz2 15 s 크래시 테스트 (DISPLAY=${DISPLAY:-없음})"
timeout 15 rviz2 -d "$REPO/ros2_ws/install/perceptrack3d_ros/share/perceptrack3d_ros/rviz/perceptrack3d.rviz" \
    > "$OUT/verification_rviz.log" 2>&1
RVIZ_EXIT=$?
echo "rviz2 exit code: $RVIZ_EXIT (124 = 15 s 동안 크래시 없이 실행됨)" | tee -a "$OUT/verification_rviz.log"
grep -iE "error|exception|segfault" "$OUT/verification_rviz.log" | head -5 || true

echo "[verify] 종료"
