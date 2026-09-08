# Phase 9 — ROS2 Jazzy 통합 학습 노트 (팀 H)

독립 실행형 Python 파이프라인(Phase 1~8) 을 **바꾸지 않고** ROS2 노드 6 개로 감쌌다. 파이프라인 로직은 전부 `src/perceptrack3d/` 에 있고,
`ros2_ws/src/perceptrack3d_ros/` 는 (1) 메시지 ↔ numpy 변환(`msg_utils.py`), (2) 노드 배선, (3) launch/RViz/파라미터만 담당한다.

## 1. 노드 / 토픽 / TF 다이어그램

```text
                    ┌──────────────────────┐
                    │  kitti_player_node   │  KITTI PNG + .bin 을 10 Hz 로 재생 (stamp = frame_id / rate_hz)
                    └──────────┬───────────┘
   /kitti/image_raw (Image, camera_02)       ─┬──────────────┬─────────────────────────┐
   /kitti/camera_info (CameraInfo, camera_02) ─┼──────┬───────┼──────────┐              │
   /kitti/velodyne_points (PointCloud2, velodyne) ┼──────┼───────┼──┐       │              │
   /kitti/frame_id (Int32)                     │      │       │  │       │              │
   /tf_static  base_link→velodyne, velodyne→camera_02 (StaticTransformBroadcaster)       │
                                               ▼      │       ▼  ▼       │              │
                                   ┌───────────────┐  │  ┌─────────────────────┐        │
                                   │ detector_node │  │  │lidar_projection_node│        │
                                   │ YOLOv8n (CPU) │  │  │ project + overlay   │        │
                                   └───────┬───────┘  │  └──────────┬──────────┘        │
        /perceptrack3d/detections_2d (Detection2DArray)              │ /perceptrack3d/projection_image (Image)
                                           │          │                                 │
                                           ▼          ▼                                 │
                                   ┌─────────────────────┐                              │
                                   │     fusion_node     │  camera_info + points + dets 동기화
                                   │ fuse_frame_clustered│  (ApproximateTimeSynchronizer)
                                   └──────────┬──────────┘
        /perceptrack3d/objects_3d (Detection3DArray, velodyne)   /perceptrack3d/markers_objects (MarkerArray)
                                              ▼
                                   ┌─────────────────────┐
                                   │    tracker_node     │  KalmanTracker.step(objects, frame_id)
                                   └──────────┬──────────┘
        /perceptrack3d/tracks (Detection3DArray, id = track_id)  /perceptrack3d/markers_tracks (박스 + "ID n" + 궤적)
                                              ▼
                                   ┌─────────────────────┐
                                   │ visualization_node  │  image + camera_info + detections_2d + tracks 4-way 동기화
                                   └──────────┬──────────┘
        /perceptrack3d/annotated_image (Image)  →  RViz2 (Image 패널)

TF 트리:  base_link ──(z +1.73 m)──▶ velodyne ──(inv(T_velo_to_rect))──▶ camera_02
```

## 2. 메시지 타입을 고른 이유

| 데이터 | 메시지 | 이유 |
|---|---|---|
| RGB | `sensor_msgs/Image` (bgr8, step = W·3) | 표준. cv_bridge 가 OpenCV 5 + numpy 2.5 에서 깨져 `img.tobytes()` 로 직접 채움 (0.1 ms) |
| LiDAR | `sensor_msgs/PointCloud2` (x, y, z, intensity float32, point_step 16) | KITTI .bin 의 메모리 배치와 동일해 `tobytes()` 한 번 (1 ms). RViz 가 intensity 로 색칠 |
| 캘리브레이션 | `sensor_msgs/CameraInfo` (K, R, P) + `/tf_static` | P = P_rect_02, R = R_rect_00, D = 0 (정류 이미지). 외부 파라미터는 TF 로 |
| 2D 검출 | `vision_msgs/Detection2DArray` | bbox(center, size) + hypothesis(class_id 문자열, score). COCO 이름을 그대로 class_id 에 |
| 3D 객체 / 트랙 | `vision_msgs/Detection3DArray` | BoundingBox3D(center Pose, size Vector3) 가 Object3D(center, size, yaw) 와 1:1. 트랙 id 는 `Detection3D.id` |
| 시각화 | `visualization_msgs/MarkerArray` | CUBE(반투명) + TEXT_VIEW_FACING + LINE_STRIP(궤적), lifetime 0.15 s |

커스텀 메시지는 만들지 않았다 (`ament_python` 패키지는 msg 생성이 안 되고, 표준 타입으로 충분했다).

## 3. 타임스탬프와 동기화

- **stamp = frame_id / rate_hz 초.** 10 Hz 면 프레임 37 → 3.7 s. 하위 노드는 `round(stamp × rate_hz)` 로 프레임 번호를 복원한다.
  프레임 번호를 실어 보낼 헤더 있는 표준 메시지가 없어서 택한 규약이다. loop 재생 시 stamp 가 0 으로 돌아가고 tracker 는 자동 초기화된다.
- **모든 노드는 입력 header 를 그대로 복사**한다. `visualization_node` 가 4 개 입력의 stamp 가 정확히 같은지 매 프레임 검사해 로그로 남긴다
  (검증 실행에서 250/250 프레임 일치, `outputs/phase9/verification_launch.log`).
- **`message_filters.ApproximateTimeSynchronizer(queue_size=10, slop=0.02)`**: 같은 프레임은 stamp 가 완전히 같으므로 slop 0.02 s 는 여유값이다.
  fusion 은 (camera_info, velodyne_points, detections_2d) 3-way, visualization 은 (image, camera_info, detections_2d, tracks) 4-way.
  queue_size 10 = 1 s 분량이라 detector(20 ms) → fusion(15 ms) 지연은 충분히 흡수한다.

## 4. TF 방향 규약과 검증

tf2 의 `TransformStamped(header.frame_id = 부모, child_frame_id = 자식)` 에서 `transform` 은 **"자식 좌표의 점을 부모 좌표로 옮기는" 변환 T_parent←child** 이다.
calib 파일의 `T_velo_to_rect` 는 velodyne 점 → rect(camera_02) 좌표, 즉 T_camera_02←velodyne 이다. 그래서

- 부모 `velodyne`, 자식 `camera_02` 로 발행할 때는 **역행렬** `inv(T_velo_to_rect)` 를 넣는다. 이 translation 은 "camera_02 원점의 velodyne 좌표" = (0.273, −0.002, −0.072) m → 카메라가 Velodyne 보다 27 cm 앞, 7 cm 아래 (KITTI setup 페이지와 일치).
- `ros2 run tf2_ros tf2_echo A B` 는 `lookupTransform(A, B)` = "B 프레임 데이터를 A 프레임으로 옮기는 변환" 을 찍는다. 따라서

| 명령 | 출력 translation | 의미 |
|---|---|---|
| `tf2_echo camera_02 velodyne` | (−0.003, −0.075, −0.272) | = calib `T_velo_to_rect[:3, 3]` (velodyne 원점의 camera 좌표) |
| `tf2_echo velodyne camera_02` | (0.273, −0.002, −0.072) | = `inv(T_velo_to_rect)[:3, 3]` (camera 원점의 velodyne 좌표) |
| `tf2_echo base_link camera_02` | (0.273, −0.002, 1.658) | 카메라 높이 1.65 m (KITTI setup) |

`scripts/ros2_check_pipeline.py` 는 `/tf_static` 을 받아 행렬로 되돌린 뒤 `inv(T_velo_to_rect)` 와 비교한다 (최대 오차 6.7e-16).

하위 노드(projection, fusion, visualization) 는 파일을 읽지 않고 `CameraInfo.P` + `lookup_transform("camera_02", "velodyne")` 로
`KittiCalibration(T_velo_to_cam = TF, R_rect_00 = I, P_rect_02 = P)` 를 만든다 (`tf_calib.py`). camera_02 프레임을 정류 프레임으로 정의했으므로 R_rect 는 TF 안에 흡수된다.

## 5. QoS 결정

전 토픽 **RELIABLE, KEEP_LAST depth 10** (rclpy 기본값 `10`).
- 실제 센서 드라이버는 BEST_EFFORT(`qos_profile_sensor_data`) 가 관례지만, 이 노드는 파일 재생기라 "최신 것만" 이 아니라 "빠짐없이" 가 목표다 (tracker 의 dt 고정 가정).
- RELIABLE 발행자는 BEST_EFFORT 구독자(RViz, 외부 도구)와 호환되지만 반대는 연결이 안 된다. rosbag2 도 발행자 QoS 를 따라 기록한다.
- 단, RELIABLE + KEEP_LAST 도 이력(depth) 이 넘치면 버린다. 아래 RMW 비교 참고.

## 6. 실행 명령

```bash
# 빌드 + 전체 실행 (RViz 포함). 워크스페이스는 저장소 안 ros2_ws/ (~/ros2_ws 아님)
scripts/ros2_demo.sh
scripts/ros2_demo.sh use_rviz:=false method:=raw use_offline_json:=true rate_hz:=5.0 loop:=true

# 수동
source /opt/ros/jazzy/setup.bash; source ~/ros2_jazzy/install/setup.bash
source .venv/bin/activate && (cd ros2_ws && colcon build --symlink-install)   # .venv 의 colcon 이어야 함
source ros2_ws/install/setup.bash
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
ros2 launch perceptrack3d_ros perceptrack3d.launch.py

# headless 검증 / rosbag 기록
scripts/ros2_verify.sh 30 10.0        # outputs/phase9/verification_*.{md,txt,log}
scripts/record_bag.sh 30 10.0         # outputs/phase9/bag_sample/, bag_info.txt
ros2 bag play outputs/phase9/bag_sample   # 기록된 결과를 RViz 로 다시 보기
```

launch 인자: `use_rviz`, `method`(raw|clustered), `use_offline_json`, `rate_hz`, `loop`, `start_frame`, `end_frame`, `start_delay_s`, `omp_threads`, `config_path`.

RViz2 에서 볼 것 (Fixed Frame = velodyne): PointCloud2(intensity 색), TF 축 3 개, Objects 3D(클래스 색 박스 + "car 20.7m"), Tracks(ID 색 박스 + "ID n" + 궤적), 이미지 패널 2 개(annotated / projection).

## 7. 측정 결과 (i7-14700K 28 스레드 WSL2, CPU, 2011_09_26_drive_0015 297 프레임, 다른 ROS2 세션(FAST-LIO) 과 CPU 공유 중 측정)

단독 실행 (`.venv/bin/python`, 프레임 20~39 중앙값):

| 단계 | ms/프레임 |
|---|---|
| YOLOv8n imgsz 640 (torch 스레드 2/4/8/28) | 17.8 / 17.3 / 14.8 / 17.0 |
| fusion raw / clustered | 3.8 / 13.9 |
| projection / 오버레이 그리기(cv2.circle ×19k) | 2.6 / 26.2 |
| PointCloud2 패킹+직렬화+역직렬화+언패킹 / Image 동일 | 2.5 / 0.6 |

파이프라인 안 (노드 6 개 동시, 10 Hz, 노드 로그 "최근 50 프레임 평균"):

| 조건 | detector | fusion(clustered) | projection | 놓친 프레임 / 297 |
|---|---|---|---|---|
| 스레드 제한 없음 (torch 14, OMP 28 per process) | 120~184 | 131~142 | 76~85 | 다수 (5 Hz 로 낮춰도 42~62 / 60~80) |
| `omp_threads=4`, `torch_threads=4`, Fast DDS | 17~20 | 14~20 | 33~38 | 5 (9, 63, 120, 175, 242 근처, 1 프레임씩) |
| 위 + **CycloneDDS** | 19~20 | 13~17 | 40~47 | **0** |

교훈: (1) 프로세스마다 코어 전부를 스레드로 잡으면 OpenMP 스핀 대기 경합으로 5~10 배 느려진다. (2) Fast DDS 는 1.4~2 MB 메시지에서 주기적으로 1 프레임을 잃었고 CycloneDDS 는 잃지 않았다 (`outputs/phase9/rmw_compare_*.log`).

## 8. 함정 모음

- **venv + ROS2 PYTHONPATH**: 셸이 `~/ros2_jazzy/install/*/site-packages` 를 PYTHONPATH 에 넣어 주므로 `.venv/bin/python` 에서 rclpy 가 import 된다. 반대로 pytest 에는 launch_* 플러그인이 새어 들어와 `-p no:launch_*` 가 필요했다 (pyproject.toml).
- **colcon 의 python**: `/usr/bin/colcon` 은 `sys.executable`(시스템 python) 로 setup.py 를 돌려 노드 스크립트 shebang 이 `/usr/bin/python3` 이 된다 → numpy 2.5/torch 없음. `.venv/bin/pip install colcon-common-extensions` 후 venv 안에서 빌드하면 shebang 이 `.venv/bin/python`.
- **numpy 2 + cv_bridge**: import 는 되지만 `cv2_to_imgmsg` 가 `KeyError: 16` (OpenCV 5 타입 테이블 불일치). 5 줄 헬퍼로 대체.
- **저장소 `ros2_ws/` vs `~/ros2_ws`**: 이름이 같다. 항상 절대경로. 셸이 `~/ros2_ws/install/setup.bash` 를 자동 source 하지만 우리 패키지는 `ros2_ws/install/setup.bash` 를 따로 source 해야 보인다.
- **params.yaml 우선순위**: launch 의 dict 파라미터는 `/**` 로 기록되고 rcl 은 "정확한 노드 이름 > 와일드카드" 라 노드별 섹션이 있으면 `loop:=true` 가 무시된다. yaml 을 전부 `/**` 로.
- **`ros2 launch` 는 시스템 python**: launch 파일에서 `perceptrack3d` 를 import 할 수 없다. 저장소 루트는 launch 파일 위치에서 위로 올라가며 `configs/kitti.yaml` 을 찾는다.
- **launch 인자의 타입**: `start_delay_s:=6` 은 YAML 로 int 가 되어 double 파라미터에 들어가면 `InvalidParameterTypeException` 으로 노드가 죽는다. launch 에서 `ParameterValue(LaunchConfiguration(...), value_type=float)` 로 감싼다.
- **`ros2 bag record` 도 구독자**: 1.4 MB 이미지 토픽은 기록기 쪽에서 30 개 중 1~2 개를 놓쳤다 (`bag_info.txt`). 작은 메시지(검출·객체·트랙)는 30/30.
- **백그라운드 작업과 SIGINT**: non-interactive 셸에서 `cmd &` 는 SIGINT 를 무시(SIG_IGN) 하고 자식 노드까지 물려받아 Ctrl+C 상당의 `kill -INT` 가 안 먹는다. 스크립트는 `set -m` + `kill -TERM`.
- **`pkill -f 패턴`** 은 그 패턴을 담은 자기 자신의 셸도 죽인다. `[p]erceptrack3d_ros/` 처럼 대괄호로 감싼다.
- **Ctrl+C 종료 규약**: 터미널 Ctrl+C 는 프로세스 그룹 전체(launch + 노드) 에 SIGINT 를 보내고, launch 는 자식에게 SIGINT 를 **한 번 더** 전달한다.
  첫 SIGINT 에 rclpy 핸들러가 컨텍스트를 닫고 `spin` 이 `ExternalShutdownException` 을 던지는데, 그 뒤 `destroy_node()` 를 부르면 닫힌 컨텍스트라 실패하고,
  두 번째 SIGINT 는 (rclpy 핸들러가 이미 제거된 뒤라) Python 기본 핸들러로 가서 atexit 도중 KeyboardInterrupt → exit code −2 가 된다.
  해결: `finally` 에서 `if rclpy.ok(): destroy_node()` + `rclpy.try_shutdown()` + `signal.signal(SIGINT, SIG_IGN)`. 6 개 노드 모두 "process has finished cleanly" 확인.
  Jazzy 문서의 `with rclpy.init(args=args):` 패턴은 이 소스 빌드(rclpy.init 이 None 반환)에서 동작하지 않는다.
- **loop 재생**: stamp 가 0 으로 되돌아가므로 ApproximateTimeSynchronizer 큐에 남은 옛 메시지와 섞일 수 있고 tracker 는 초기화된다. 평가용이 아니라 데모용.

## 9. 알려진 제한

- CPU 온라인 YOLO 로 10 Hz 는 유지되지만(20 ms), 20 Hz 이상은 projection 오버레이(40 ms) 가 먼저 막힌다. `use_offline_json:=true` 로 검출 부하를 없앨 수 있다.
- RViz2 는 WSLg(DISPLAY=:0) 에서 15 s 동안 크래시 없이 뜨는 것만 확인했다(`outputs/phase9/verification_rviz.log`). 스크린샷은 사용자가 직접.
- Fast DDS 기본 설정에서는 약 55 프레임마다 1 프레임을 잃는다 → 스크립트는 CycloneDDS 를 기본으로 하되 `RMW_IMPLEMENTATION` 이 이미 설정돼 있으면 존중한다.
  **이 머신의 셸 프로필은 `RMW_IMPLEMENTATION=rmw_fastrtps_cpp` 를 명시**하므로 실제로는 Fast DDS 로 돈다. 손실 없이 보려면 `RMW_IMPLEMENTATION=rmw_cyclonedds_cpp scripts/ros2_demo.sh` 처럼 명시한다 (다른 워크스페이스에는 영향 없음).
- Detection3D 에는 속도 필드가 없어 칼만 상태의 속도는 토픽으로 나가지 않는다 (궤적 LINE_STRIP 으로만 보임).
- 검증 중 다른 ROS2 세션(FAST-LIO + rviz2 + ffmpeg) 이 같은 CPU 를 쓰고 있었으므로 절대 ms 는 재현 시 더 낮을 수 있다. 검증은 `ROS_DOMAIN_ID=7` 로 토픽을 격리했다.

## 10. 자기 점검 질문

1. `tf2_echo camera_02 velodyne` 의 translation 이 calib 파일의 T_velo_to_rect 와 같은 이유를 "부모/자식" 과 "lookupTransform(target, source)" 의 정의로 설명할 수 있는가?
2. fusion_node 는 왜 image_raw 대신 camera_info 를 구독해도 되는가? 만약 이미지 크기가 프레임마다 바뀐다면 무엇이 깨지는가?
3. RELIABLE + KEEP_LAST(10) 인데도 Fast DDS 에서 프레임이 빠진 이유는 무엇이고, tracker 의 어떤 가정이 이때 깨지는가?
