# 학습 노트 색인

각 단계를 혼자 공부할 때의 권장 순서. 노트북을 먼저 실행해 보고, 막히면 해당 노트를 읽고, 마지막에 `src/perceptrack3d/` 의 모듈 코드를 읽는다.

| 단계 | 노트 | 노트북 | 모듈 | 테스트 |
|---|---|---|---|---|
| 0 | [phase0_environment.md](phase0_environment.md) | – | `config.py` | – |
| 1 | [phase1_data.md](phase1_data.md) | `01_kitti_data_check.ipynb` | `data/kitti_loader.py` | `test_kitti_loader.py` |
| 2 | [phase2_calibration.md](phase2_calibration.md) | `02_calibration_and_frames.ipynb` | `geometry/calibration.py`, `transforms.py` | `test_calibration.py` |
| 3 | [phase3_projection.md](phase3_projection.md) | `03_lidar_camera_projection.ipynb` | `geometry/projection.py` | `test_projection.py` |
| 4 | [phase4_detection.md](phase4_detection.md) | `04_yolo_detection.ipynb` | `detection/detector.py` | `test_detector.py` |
| 5 | [phase5_fusion.md](phase5_fusion.md) | `05_camera_lidar_fusion.ipynb` | `fusion/frustum_fusion.py` (`fuse_frame_raw`) | `test_fusion.py` |
| 6 | [phase6_localization.md](phase6_localization.md) | `05_camera_lidar_fusion.ipynb` (후반) | `fusion/point_filters.py`, `fuse_frame_clustered` | `test_fusion.py` |
| 7 | [phase7_tracking.md](phase7_tracking.md) | `06_multi_object_tracking.ipynb` | `tracking/kalman_tracker.py`, `association.py` | `test_tracking.py` |
| 8 | [phase8_ground_truth.md](phase8_ground_truth.md), [phase8_evaluation.md](phase8_evaluation.md) | `06_multi_object_tracking.ipynb` (파라미터 스윕) | `evaluation/*.py` | `test_tracklets.py`, `test_metrics.py`, `test_evaluate.py` |
| 9 | [phase9_ros2.md](phase9_ros2.md) | – | `ros2_ws/src/perceptrack3d_ros/` | `scripts/ros2_verify.sh` |
| 10 | [phase10_cpp.md](phase10_cpp.md) | – | `cpp/`, `geometry/projection_cpp.py` | `ctest`, `test_cpp_projection.py` |

**두 가지 노트북 세트.** `notebooks/`(완성본, 정답지)와 `notebooks_2nd/`(실무 방식 재구성: 검색어 → 출처 → 원문 발췌 → 적용 → 정답지와 검증). 공부는 `notebooks_2nd`로 하고, 막히면 정답지를 본다. 2차 노트북 번호는 01 로드, 02 캘리브, 03 투영, 04 검출, 05 융합, 06 개선, 07 추적, 08 평가.

각 단계에서 스스로 답할 수 있어야 하는 여섯 질문 (CLAUDE.md 5절):
1. 입력은 무엇이고 shape/type 은? 2. 출력은? 3. 어느 좌표계? 4. 핵심 수식/알고리즘은? 5. 실패 케이스는? 6. 어떻게 평가하나?

설계 결정의 이유는 [../decisions.md](../decisions.md), 참고 자료는 [../references.md](../references.md), 모듈 계약은 [../architecture.md](../architecture.md).
