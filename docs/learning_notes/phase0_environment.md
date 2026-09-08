# Phase 0 — 데이터셋과 환경

## 확인한 환경 (2026-09-07)
| 항목 | 값 |
|---|---|
| OS | Ubuntu 24.04.4 LTS (WSL2) → ROS2 **Jazzy** |
| Python | 3.12.3 (`/usr/bin/python3`), venv = `.venv/` |
| CPU / RAM | 28 스레드, 31 GB |
| GPU | RTX 4070 Ti 12 GB (기본은 CPU 사용, `detection.device` 로 전환) |
| 디스크 | 331 GB 여유 |
| Git | 2.43 |
| ROS2 | Jazzy 설치됨 (`/opt/ros/jazzy` + `~/ros2_jazzy` 소스 빌드), rviz2/tf2_ros/cv_bridge/sensor_msgs_py 사용 가능 |
| C++ | g++ 13.3, cmake 3.28, Eigen3, PCL 1.14, pybind11 |

## 데이터 (`/home/ingon/datasets/KITTI/raw/2011_09_26/`)
- `calib_cam_to_cam.txt`, `calib_velo_to_cam.txt`, `calib_imu_to_velo.txt`
- `2011_09_26_drive_0015_sync/` — image_00~03, velodyne_points, oxts 각 **297 프레임**, `tracklet_labels.xml` (tracklet 36개)
- ZIP 원본은 그대로 두고 옆에 해제했다 (`unzip -n`).

## 함정
- `~/.config/pip/pip.conf` 에 NVIDIA 인덱스가 있어 pip 가 매번 재시도하며 느려진다 → `pip install --isolated --index-url https://pypi.org/simple` 사용.
- 셸이 ROS2 setup 을 자동 source 하므로 `PYTHONPATH` 에 ROS2 경로가 들어 있다. venv 안에서도 `import rclpy` 가 되는 이유이며, Phase 9 에서 이를 활용한다.
- 같은 이유로 pytest 가 ROS2 의 `launch_*` 플러그인을 자동 로드해 죽는다 → `pyproject.toml` 의 `addopts = "-p no:launch_testing -p no:launch_ros -p no:launch_pytest"` 로 끈다.
- Windows 에서 복사한 파일 옆에 생기는 `*:Zone.Identifier` 는 무시(.gitignore).

## 자가 점검 질문
1. image_02 와 velodyne_points 의 같은 번호 파일이 "동기화" 되었다는 것은 무엇을 뜻하는가?
2. venv 를 저장소 안에 두는 이유는? 왜 데이터셋은 저장소 밖에 두는가?
3. Ubuntu 24.04 에서 ROS2 Jazzy 를 써야 하는 이유는?
