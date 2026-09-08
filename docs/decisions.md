# 설계 결정 기록 (Decision log)

형식: 날짜 — 결정 — 이유 — 대안

- 2026-09-07 — **전체 파이프라인을 먼저 구현하고, 사용자는 이후 단계별 노트북으로 학습한다.** — 사용자가 최종 결과물의 모습을 먼저 보고 싶어 함. CLAUDE.md 의 "한 번에 한 단계" 규칙은 학습 세션에 적용하고, 이 빌드 세션은 예외로 한다. — 대안: 단계별 순차 진행 (느림).
- 2026-09-07 — **3D 결과의 기준 좌표계 = Velodyne.** — tracklet_labels.xml 이 Velodyne 좌표계이고 BEV·ROS(REP-103) 와 일치. — 대안: rectified camera 프레임 (KITTI object benchmark 방식). 평가 시 변환이 한 번 더 필요.
- 2026-09-07 — **Python 3.12 + 저장소 내부 .venv, torch 는 CPU 빌드.** — CLAUDE.md CPU-first. RTX 4070 Ti 가 있으나 재현성과 설치 크기를 위해 CPU 기본. `configs/kitti.yaml` 의 `detection.device` 로 GPU 전환 가능. — 대안: CUDA torch (2.5 GB).
- 2026-09-07 — **ROS2 는 이미 설치된 Jazzy(/opt/ros/jazzy + ~/ros2_jazzy 소스 빌드) 를 사용.** Ubuntu 24.04 → Jazzy. — 새로 설치하지 않음.
- 2026-09-07 — **Git 커밋은 사용자가 직접 수행.** — 사용자의 커밋 정책(AI 귀속 표시 없음)을 존중. `git init` 만 수행.
- 2026-09-07 — **(팀 C) 칼만 필터 프로세스 잡음 Q 는 단순 대각 모델 `diag(σp²·I3, σv²·I3)`.** — 설정 키 `process_noise_pos/vel` 두 개와 1:1 로 대응해 읽기 쉽다. — 대안: 이산 백색 가속(DWNA) 모델 (파라미터 σa 하나, dt 스케일링이 물리적으로 정확). Phase 7 개선 단계에서 비교 가능.
- 2026-09-07 — **(팀 C) `tracking.match_same_class` 기본값 false, `initial_vel_std` 5 → 10 m/s.** — YOLO 가 car/truck 을 프레임마다 바꾸면 클래스 제한 매칭은 ID 를 끊는다. 속도 prior 는 자차 기준 상대속도(마주 오는 차 ~25 m/s) 를 덮어야 2 번째 프레임에서 게이트 안에 들어온다. — 대안: 클래스 그룹(차량/보행자) 단위 매칭.

## 팀 B (Phase 4)

- 2026-09-07 — **검출 모델 = 사전학습 yolov8n (COCO), 학습하지 않음.** — 프로젝트 핵심은 융합·추적이고 CPU-first 이므로 가장 작은 모델로 기준선을 잡는다. CPU(i7-14700K) 에서 imgsz=640 기준 프레임당 중앙값 13.8 ms (72 FPS) 로 실시간 여유가 크다. 가중치는 `outputs/models/yolov8n.pt` 에 두어(`.gitignore` 의 `outputs/*`, `*.pt`) 저장소에 들어가지 않고, `save=False` 로 ultralytics 가 `runs/` 를 만들지 않게 했다. — 대안: yolov8s/m (정확도↑, CPU 시간 3~8배), KITTI 미세조정 (GPU·시간 필요, 나중에 정량 평가로 필요성 판단).
- 2026-09-07 — **COCO → KITTI 라벨 매핑은 1:1 이 아니므로 평가 시 아래 규칙으로 묶는다.**

  | COCO (검출 출력) | KITTI `objectType` | 비고 |
  |---|---|---|
  | car | Car, Van | Van 은 대부분 car 로 검출됨. car ↔ {Car, Van} 으로 묶어 평가 |
  | truck | Truck, Van(일부) | 배송 밴이 truck 으로 나오기도 함 (프레임 163, 168) |
  | bus | Truck/Van 의 오검 | 이 시퀀스에 Bus GT 없음. bus 검출 10건은 전부 큰 밴/트럭 (프레임 168, 169, 244) |
  | person | Pedestrian, Person_sitting | 자전거 탄 사람이 person 으로만 나올 수도 있음 |
  | bicycle + person | Cyclist | KITTI 는 한 객체, COCO 는 둘. 두 박스가 겹치면 Cyclist 로 합치는 규칙이 필요. 이 시퀀스에서 bicycle 검출은 1건뿐 → 사실상 Cyclist 는 검출 못 함 |
  | motorcycle | (없음) | 이 시퀀스에 없음 |
  | (없음) | Tram, Misc, DontCare | 평가 제외 |

- 2026-09-07 — **conf_threshold = 0.25 (ultralytics 기본값) 유지, iou_threshold = 0.45.** — GT 투영 박스와 IoU ≥ 0.4 로 매칭한 proxy 지표(임시 스크립트, Phase 8 정식 평가 아님)에서 conf 0.25→0.4 로 올리면 매칭률 74.7% → 62.3% 로 떨어지고 미매칭 검출은 21.7% → 11.7% 로 준다. 미매칭 검출의 상당수는 tracklet 에 라벨이 없는 주차 차량(프레임 8, 284)이라 실제 오검출은 더 적고, 오검출은 Phase 5 의 LiDAR 점 수 조건과 Phase 7 의 시간적 일관성으로 걸러낼 수 있으므로 **재현율 우선**으로 낮은 임계값을 택한다. Phase 8 에서 임계값 스윕으로 재검토. — 대안: 0.4 이상 (정밀도↑, 재현율↓).
- 2026-09-07 — **imgsz = 640 기준선 유지, 1280 은 측정된 개선 옵션으로 기록.** 1242×375 는 640 letterbox 시 640×192 로 축소(0.52배)되어 40 m 이상 차량이 10 px 대가 된다. 297 프레임 전체 측정(CPU, 다른 작업과 동시 실행 중이라 절대 ms 는 runtime.md 보다 약간 낮음):

  | imgsz | conf | 총 검출 | GT 매칭률 (proxy recall) | Car/Van | Cyclist | 미매칭 검출 | 중앙값 ms |
  |---|---|---|---|---|---|---|---|
  | 640 | 0.25 | 1325 | 74.7% | 79.6% | 3.4% | 21.7% | 11.2 |
  | 640 | 0.40 | 981 | 62.3% | 66.7% | 1.1% | 11.7% | 12.4 |
  | 1280 | 0.25 | 1894 | 87.0% | 90.4% | 33.7% | 36.2% | 33.6 |
  | 1280 | 0.40 | 1437 | 80.9% | 85.1% | 15.7% | 21.8% | 33.5 |

  1280 은 재현율 +12%p, Cyclist 10배 개선이지만 시간 3배·미매칭 검출 증가. 기준선은 공통 설정(640)을 따르고, Phase 8 에서 A/B/C 변형 비교 후 전환 여부를 결정한다. `configs/kitti.yaml` 의 `detection.imgsz` 한 줄로 바꿀 수 있다.
- 2026-09-07 — **NMS 는 ultralytics 기본(클래스별) 유지.** — 같은 물체에 truck+bus 처럼 클래스가 다른 박스가 겹쳐 남는 이중 검출(프레임 168)이 생기지만, 이를 없애는 `agnostic_nms=True` 는 붙어 있는 다른 클래스(사람 옆 자전거)까지 지울 수 있다. 융합 팀은 같은 3D 위치에 2개 객체가 올 수 있음을 전제할 것. — 대안: agnostic NMS, 또는 융합 후 3D 중복 제거.
- 2026-09-07 — **추론 결정성은 테스트로 보장, 시드 설정은 하지 않음.** — 사전학습 가중치 + CPU 추론은 난수를 쓰지 않아 같은 입력에 같은 출력이 나온다 (`tests/test_detector.py::test_detect_deterministic`). `torch.manual_seed` 는 학습/증강용.
- 2026-09-07 — **pytest 에 `-p no:launch_testing -p no:launch_ros -p no:launch_pytest` 추가 (pyproject.toml).** — 셸 PYTHONPATH 의 ROS2(~/ros2_jazzy) 경로가 venv 에 새어 들어와 pytest 9 와 호환되지 않는 launch_* 플러그인이 자동 로드되어 INTERNALERROR 가 났다. 모든 팀에 공통.
- 2026-09-07 — **캘리브레이션 회전 행렬(R, R_rect_00)은 파싱 시 SVD 로 가장 가까운 회전 행렬로 보정한다 (`geometry.transforms.nearest_rotation`).** — 텍스트 파일은 유효숫자 7자리라 `R Rᵀ − I ≈ 8e-8` 이고, 그대로 두면 `Rᵀ` 기반 역변환의 round-trip 오차가 70 m 에서 1.8e-6 m 로 허용치(1e-6)를 넘음. 보정 후 값 변화 1e-7, 투영 영향 < 0.001 px. — 대안: `np.linalg.inv` 사용 (강체 변환의 의미가 드러나지 않음), 허용치 완화.

## 팀 I (Phase 10)

- 2026-09-07 — **C++ 로 옮기는 컴포넌트 = LiDAR→이미지 투영 + ROI 마스크 + K개 박스 frustum 마스크 (`cpp/`, 정적 라이브러리 `pt3d_core` + pybind11 모듈 `pt3d_cpp`).** — `scripts/profile_projection.py`: 투영 1회 2.7 ms 는 프레임당 10% 지만, Phase 5/6 이 검출(평균 4.5개)마다 투영을 반복하면 15~17 ms 로 YOLO(13.8 ms)보다 커진다. cProfile 상 비용의 절반이 float64 승격·`hstack` 등 임시 배열 생성이라 단일 루프 C++ 이 확실히 이득인 패턴. YOLO/PNG 디코딩/DBSCAN 은 이미 C/C++ 커널이라 제외. — 대안: PointCloud2 변환(ROS2 단계에서 재평가), 클러스터링(Open3D 가 이미 C++).
- 2026-09-07 — **투영 행렬 `P` 는 float64 로 받아 double 로 계산한다 (지시서의 `Matrix<float,3,4>` 에서 변경).** — Python 이 float64 로 계산하므로 P 를 float32 로 반올림하면 u 에 ~1e-4 px 오차가 생겨 `u < W` 경계 판정이 뒤집힐 수 있다. double 계산 결과 20 프레임 uv 최대 차이 0.0, mask 완전 일치. 점은 float32 그대로(승격은 정확). — 대안: float32 전체 계산 (SIMD 2배 유리하지만 mask 완전 일치 보장 불가).
- 2026-09-07 — **(N,4) 배열을 자르지 않고 그대로 넘기고 C++ 이 `Eigen::Ref` + `OuterStride<>` 로 앞 3열만 읽는다 (zero-copy).** — `pts[:, :3]` 은 비연속 뷰라 pybind11 `c_style` 이 복사를 만든다. gtest 가 `ref.data() == 원본 포인터` 로 검증. — 대안: 래퍼에서 `np.ascontiguousarray(pts[:, :3])` (1.5 MB 복사, ~0.1 ms).
- 2026-09-07 — **frustum API 는 박스 K개를 한 번에 받아 투영을 1회만 한다 (`frustum_masks_cpp(points, boxes, ...) -> (K,N)`).** — 진짜 병목은 검출당 재투영. 벤치마크: 단독 `frustum_mask` 를 박스마다 호출하면 17.1 ms → C++ 0.67 ms (25배) 이지만, 순수 Python 도 투영 1회 후 uv 재사용하면 3.5 ms 가 되므로 **이득의 대부분은 알고리즘**이고 C++ 은 그 위에 5.3배. 팀 E 의 `frustum_fusion.py` 는 이미 프레임당 투영 1회 구조라 실제 적용 이득은 5.3배(≈2.9 ms/프레임). — 대안: 박스 1개 API (호출당 오버헤드 0.03 ms × K, 재투영 반복).
- 2026-09-07 — **확장 모듈 `.so` 는 `cpp/build/` 에 두고 `geometry/projection_cpp.py` 가 그 경로를 `sys.path` 에 넣어 로드한다.** — 패키지 디렉터리에 `.so` 를 복사하면 `.gitignore` 에 `*.so` 를 추가해야 하고 빌드 산출물이 소스 트리에 섞인다. `cpp/build/` 는 이미 무시 대상. 미빌드 환경에서는 빌드 명령이 적힌 `ImportError`, pytest 는 `importorskip` 으로 skip. — 대안: `pip install -e .` 에 C 확장 빌드 통합 (setup.py/scikit-build-core; 사용자 학습 부담 대비 이득 작음).
- 2026-09-07 — **Release 플래그를 `-O2 -DNDEBUG` 로 고정하고 `-march=native` 는 쓰지 않는다.** — 벤치마크 재현성과 다른 PC 에서의 동작 보장. 12만 점 0.37 ms 는 이미 예산 안. pybind11 이 확장 모듈에 기본으로 붙이는 `-flto`/`-fvisibility=hidden` 은 그대로 두고 benchmark.md 에 명시 (핫 루프가 있는 `pt3d_core` 는 LTO 없이 `-O2` 로 컴파일됨). — 대안: `-O3 -march=native` (AVX2/FMA, 측정 시 별도 옵션으로).
- 2026-09-07 — **스마트 포인터는 pybind11 경계(`moveToNumpy`, `unique_ptr` → capsule)에서만 쓴다. 코어 라이브러리는 `std::vector` 값 반환(RAII/NRVO)만으로 소유권이 명확해 스마트 포인터가 필요 없다.** — CLAUDE.md "소유권이 필요한 곳에서만". 코어에 `new`/`delete` 0줄. — 대안: `shared_ptr<ProjectionResult>` 반환 (불필요한 참조 카운트, 의미 불명확).
- 2026-09-07 — **C++ 단위 테스트는 시스템의 GoogleTest 1.14 (`libgtest-dev`) 를 사용, `ctest` 등록.** — 이미 설치되어 있고 실패 메시지가 assert 보다 명확. 없는 환경에서는 `find_package(GTest REQUIRED)` 가 설치 안내와 함께 실패한다. — 대안: assert 기반 main (의존성 0, 메시지 빈약).

## 팀 E (Phase 5, 6 + 파이프라인)

- 2026-09-07 — **대표 깊이 = 박스 안 깊이의 하위 30% 백분위(`method="nearest"`), 중심 = ±1.5 m 밴드의 좌표별 중앙값.** — 2D 박스 안에는 뒤쪽 배경 점이 섞이므로(frame 174 det1: 40%) 평균은 쓸 수 없고, "가까운 쪽이 물체" 가정으로 하위 백분위를 대표로 삼는다. 보간 백분위는 밴드가 빌 수 있어 실제 점 값을 쓴다. 스윕: p10 569 / **p30 621** / p50 655 매칭 — p50 이 조금 낫지만 오염에 약하므로 30 유지. — 대안: 최소 깊이(노이즈에 취약), 평균(배경에 끌림).
- 2026-09-07 — **클러스터 선택 규칙 기본값 = `largest` (점 수 최대), `nearest` + `cluster_min_ratio` 는 옵션.** — 297 프레임 스윕에서 largest 926 vs nearest 884 (ratio 0.3; ratio 0 이면 867) 매칭. 이 시퀀스는 물체 앞의 작은 덩어리(지면 잔여·기둥) 가 배경보다 흔하다. 반대 상황(배경이 더 큰 덩어리) 은 frame 174 det6 과 합성 테스트로 문서화. — 대안: 밀도 정규화 점수(count × depth²).
- 2026-09-07 — **DBSCAN eps 0.6 → 0.8, min_points 8 → 5.** — 8 이면 5~7 점짜리 먼 차가 전부 `no_cluster` (151 건), 0.6 은 60 m 에서 링 간격(0.42 m) 때문에 차가 쪼개짐. 스윕: (0.6, 8) 869 → (0.8, 5) 926 매칭, 오차 동일. — 대안: 거리에 따라 eps 를 키우는 적응형 (Phase 10 개선 후보).
- 2026-09-07 — **표면→중심 오프셋 `surface_offset` 추가 (Phase 6 선택 개선, 기본 on).** — LiDAR 는 보이는 면만 찍어 중앙값이 GT 중심보다 ~1.3 m 가깝다. 시선 방향으로 `max(0, half_depth[class] − extent/2)` 만큼 민다. ablation: 오프셋 없음 BEV 1.39 m / bias −1.24 → 0.63 m / −0.30. 클래스 prior(car 2.0 m 등) 는 이 시퀀스 GT 길이 평균에서 유도했고, car 2.2 가 bias 를 −0.11 로 더 줄이지만 과적합이라 채택 안 함. raw(Phase 5) 에는 적용하지 않아 기준선을 순수하게 유지. — 대안: 크기 prior 로 박스 완성(Frustum PointNets 류 학습 기반).
- 2026-09-07 — **AABB 기본, OBB(`use_obb`) 는 옵션.** — 멀리서 뒷면만 보이면 PCA 최대 분산이 폭 방향이라 yaw ≈ ±90° 가 나온다 (frame 174 det0). 이 시퀀스(대부분 35 m 이상) 에서는 크기·yaw 모두 신뢰할 수 없어 중심을 주 산출물로 둔다. — 대안: L-shape fitting.
- 2026-09-07 — **중복 제거 임계값 `dedup_distance_m` = 1.0 m, 클래스 무관.** — 같은 차에 truck+bus 처럼 박스 2개가 오면 3D 중심이 0.2~0.5 m 차이. 1 m 면 나란히 선 차(폭 1.8 + 간격) 는 합쳐지지 않고, person+bicycle 은 하나(Cyclist) 로 합쳐진다. 탈락한 쪽은 삭제하지 않고 `status=invalid, reason=duplicate_of_k` 로 남겨 분포를 볼 수 있게 했다.
- 2026-09-07 — **`Object3D` 에 `reason: str`, `truncated: bool` 필드 추가 (기본값 있음, `as_dict` 에 포함).** — 실패 원인 분포와 잘린 박스 분리 평가를 위해.
- 2026-09-07 — **트래커 파라미터는 팀 C 기본값 유지 (`measurement_noise` 0.5).** — 팀 C 는 실데이터에서 키워야 할 수 있다고 했지만, clustered 결과로 스윕한 결과 0.5 가 ID 스위치 최소(9) 이고 1.0 → 14, 1.5 → 19 로 늘었다 (재현율은 0.64 → 0.66 으로 미미). `max_misses` 5/8 은 재현율 +2~4%p 대신 정밀도 −8~18%p. 53 ID vs GT 36 tracklet (FOV 안), 단절 11. — 대안: 클래스별 R, 거리 의존 R (먼 물체일수록 오차 큼).
- 2026-09-07 — **지면 제거 RANSAC 은 numpy 구현(시드 고정) 이 기본, Open3D `segment_plane` 은 `ground.method: open3d` 옵션.** — Open3D 0.19 는 `open3d.utility.random.seed` 를 걸어도 병렬 실행 때문에 같은 프레임에서 inlier 수가 ±1000 점씩 달라져 결과 JSON 이 실행마다 미세하게 바뀌었다 (CLAUDE.md "시드 고정" 규칙 위배). numpy 버전은 가설 200개를 행렬곱으로 한 번에 평가하고 점수는 12 000 점 부분집합으로 매겨 프레임당 ~10 ms (Open3D 7~20 ms). 알고리즘이 코드에 드러나 학습 목적에도 맞다. — 대안: OMP 스레드 1개로 Open3D 실행 (환경 변수 의존).
- 2026-09-07 — **`runtime.json` 은 `{"meta", "stages", "summary", "frames": {frame_id: {stage: ms}}}` 형식.** — architecture.md 4절의 프레임별 dict 에 하드웨어·측정 방법 메타를 붙였다 (CLAUDE.md 11절 "성능 수치는 항상 하드웨어·측정 방법과 함께"). 시각화 시간은 제외.
- 2026-09-07 — **BEV 궤적 그림에는 관측으로 갱신된 위치만 (coasting 제외).** — 마주 오는 차(−26 m/s) 가 놓치면 예측만으로 4 프레임에 10 m 를 밀려 긴 직선 꼬리가 생겨 궤적 판독을 방해한다. `tracks.json` 에는 coasting 트랙도 그대로 들어 있다.

## 팀 H (Phase 9) — ROS2 Jazzy 통합

- 2026-09-07 — **워크스페이스는 저장소 안 `ros2_ws/`, 패키지는 `ament_python` `perceptrack3d_ros`, 빌드는 .venv 의 colcon 으로.** — 시스템 `/usr/bin/colcon` 은 `sys.executable`(/usr/bin/python3) 로 `setup.py` 를 돌려 노드 실행 스크립트 shebang 이 시스템 python 이 되고, 그러면 numpy 2.5·torch·ultralytics·perceptrack3d 를 못 찾는다. `.venv/bin/pip install colcon-common-extensions` 로 venv 안에서 빌드하면 shebang 이 `.venv/bin/python` 이 된다 (`scripts/ros2_demo.sh` 가 자동 처리). ROS2 파이썬 패키지(rclpy 등)는 셸 PYTHONPATH 로 venv 에 보인다. `~/ros2_ws` 는 사용자의 다른 프로젝트라 항상 절대경로로 구분. — 대안: 시스템 python 에 파이프라인 의존성 설치 (재현성·격리 훼손).
- 2026-09-07 — **타임스탬프 규약 `stamp = frame_id / rate_hz` (결정적).** — ament_python 패키지는 커스텀 메시지를 만들 수 없어 프레임 번호를 실어 보낼 헤더 있는 표준 메시지가 없다 (`/kitti/frame_id` 는 Int32 라 header 가 없어 동기화 불가). stamp 를 프레임 번호에서 계산하면 모든 하위 노드가 `round(stamp * rate_hz)` 로 프레임 번호를 복원할 수 있고, 헤더 복사 규약이 지켜지는지 검증하기도 쉽다. 단점: loop 재생 시 stamp 가 0 으로 되돌아가 tracker 가 초기화된다 (의도된 동작). — 대안: 벽시계 `now()` (프레임 번호 복원 불가), KITTI timestamps.txt (동일 문제).
- 2026-09-07 — **정적 TF 는 `velodyne`(부모) → `camera_02`(자식) 에 `inv(T_velo_to_rect)` 를 넣는다.** — tf2 의 TransformStamped.transform 은 "자식 좌표의 점을 부모 좌표로 옮기는" T_parent←child 다. calib 의 T_velo_to_rect 는 velodyne 점 → rect 카메라 좌표(반대 방향)이므로 역행렬이 필요하다. camera_02 TF 프레임은 정류(rect) 프레임으로 정의해 R_rect_00 을 TF 안에 흡수했다. 검증: `tf2_echo camera_02 velodyne` translation = (−0.003, −0.075, −0.272) = calib T_velo_to_rect[:3,3]; `tf2_echo velodyne camera_02` = (0.273, −0.002, −0.072) = 카메라 원점의 velodyne 좌표 (KITTI setup: 카메라가 Velodyne 보다 27 cm 앞, 7~8 cm 아래). `base_link → velodyne` 은 z = 1.73 m (setup 페이지) 항등 회전. — 대안: 부모/자식을 반대로 두고 T_velo_to_rect 그대로 (트리 루트가 카메라가 되어 base_link 연결이 어색).
- 2026-09-07 — **하위 노드는 캘리브레이션 파일을 읽지 않고 CameraInfo(P) + TF(camera_02←velodyne) 로 `KittiCalibration` 을 만든다 (`tf_calib.py`).** — ROS 방식의 단일 진실 공급원: 플레이어만 파일을 읽고, 나머지는 토픽/TF 로 받는다. `KittiCalibration(T_velo_to_cam=TF, R_rect_00=I, P_rect_02=CameraInfo.P)` 로 만들면 P_velo_to_img 가 파일 기반과 1e-9 안에서 같아 Phase 3/5/6 함수를 그대로 재사용한다. — 대안: 모든 노드가 파일을 읽음 (rosbag 재생·다른 센서로 바꿀 때 깨짐).
- 2026-09-07 — **QoS 는 전 토픽 RELIABLE, KEEP_LAST depth 10 (rclpy 기본).** — 실제 센서 드라이버는 BEST_EFFORT(sensor data QoS) 가 관례지만 이 노드는 파일 재생기라 손실이 의미가 없고, RELIABLE 발행자는 BEST_EFFORT 구독자(RViz 기본 설정 일부, 외부 도구)와도 호환되지만 그 반대는 연결이 안 된다. rosbag2 는 발행자 QoS 를 따라 기록. — 대안: 센서 토픽만 BEST_EFFORT (ApproximateTimeSynchronizer 구독자 QoS 를 맞춰야 하고 드롭이 tracker 의 고정 dt 가정을 깬다).
- 2026-09-07 — **메시지 타입: 커스텀 없이 `vision_msgs/Detection2DArray`, `Detection3DArray`, `visualization_msgs/MarkerArray`.** — Detection3D.bbox(center Pose + size Vector3) 가 Object3D(center, size, yaw) 와 1:1. class_name 은 `results[0].hypothesis.class_id`(문자열), confidence 는 `.score`, 트랙 id 는 `Detection3D.id`, 융합 메타는 `id = "method:n_points"`. status != "ok" 객체는 싣지 않는다 (구독자는 "있으면 유효" 로 단순화). COCO 숫자 id 는 이름에서 복원 (`msg_utils.COCO_CLASS_IDS`). 속도는 표준 필드가 없어 싣지 않음. — 대안: 커스텀 msg 패키지 (ament_cmake 필요, 팀 I 의 C++ 단계에서 필요해지면 추가).
- 2026-09-07 — **`config/params.yaml` 은 노드별 섹션 대신 전부 `/**` 아래.** — launch_ros 는 dict 파라미터를 `/**` 로 임시 yaml 에 쓰는데, rcl 은 정확한 노드 이름 섹션을 와일드카드보다 우선하므로 노드별 섹션이 있으면 `loop:=true` 같은 launch 인자가 무시된다 (실측: rate_hz 만 적용됨). 같은 `/**` 끼리는 나중 파일이 이긴다. — 대안: launch 에서 `ParameterFile` 만 쓰고 인자를 없앰.
- 2026-09-07 — **launch 가 노드마다 `OMP_NUM_THREADS/OPENBLAS_NUM_THREADS/MKL_NUM_THREADS = omp_threads(기본 4)` 를 걸고, detector 는 `torch.set_num_threads(4)`.** — 단독 실행에서는 YOLO 15 ms·clustered 융합 14 ms 인데, 6 개 프로세스가 각각 14~28 스레드를 만들자 파이프라인 안에서 YOLO 120~180 ms, 융합 130 ms 로 5~10 배 느려져 10 Hz 를 못 따라갔다 (OpenMP 스핀 대기 경합). 상한을 걸자 18~20 ms / 17~24 ms 로 회복. — 대안: MultiThreadedExecutor 하나에 노드를 합침 (프로세스 격리·ROS 관례 상실).
- 2026-09-07 — **cv_bridge 미사용, `sensor_msgs/Image` 를 numpy 로 직접 채움 (`msg_utils.image_to_msg/msg_to_image`).** — 설치된 cv_bridge 는 OpenCV 5 + numpy 2.5 조합에서 `cv2_to_imgmsg` 가 KeyError 로 실패. bgr8 은 `data = img.tobytes()`, `step = W*3` 으로 충분하고 변환 0.1 ms. — 대안: cv_bridge 재빌드 (sudo/apt 불가).
- 2026-09-07 — **fusion_node 는 image_raw 대신 camera_info 를 동기화 입력으로 쓴다.** — 융합 함수는 이미지 픽셀이 아니라 (H, W) 만 쓰므로 1.4 MB 이미지를 받을 이유가 없다. CameraInfo 의 width/height 가 같은 정보를 준다. — 대안: 지시서 원안(image_raw 구독).
- 2026-09-07 — **`ros2 topic hz`/`param get` 대신 rclpy 검증 스크립트 `scripts/ros2_check_pipeline.py` 로 발행률·stamp·값 범위·TF 를 한 번에 표로 만든다.** — CLI 도구는 짧은 timeout 안에서 출력이 flush 되지 않거나 daemon 상태에 따라 실패하는 일이 잦았고, 검증을 "명령 한 줄로 재현" 하려면 스크립트가 낫다 (`scripts/ros2_verify.sh`).
- 2026-09-07 — **`RoiParams` 멤버를 float → double 로 변경, 점을 double 로 승격해 비교.** — 팀 E 확인: `roi_filter` 는 앞 3열을 float64 로 캐스팅한 뒤 float64 임계값과 비교한다. 현재 설정값은 float32 로 정확해 결과 차이가 없지만, 0.1 같은 임계값에서는 float 비교가 경계 점을 잘못 포함한다 (gtest 케이스 추가). 속도 영향 없음(메모리 바운드). — 대안: float 유지 (NEP 50 가정, 기준 구현과 어긋남).
- 2026-09-07 — **C++ 컴포넌트는 `fuse_frame_*` 파이프라인에 연결하지 않고 옵션(`geometry/projection_cpp.py`)으로만 둔다.** — 리드 지시: Python 기준선은 비교 대상으로 순수하게 유지. 팀 E 확인: 붙일 때는 `keep_uv` 대신 `frustum_masks_cpp` 의 (K,N) 마스크에 `roi & non_ground` 를 AND 하면 되고 나머지 로직(DBSCAN, 오프셋, 중복 제거)은 그대로라 결과 JSON 이 동일해야 한다. Phase 10 비교 표의 대표 수치는 "Python 투영 1회 경로 3.5 ms vs C++ 0.7 ms". — 대안: 설정 플래그로 C++ 경로 전환 (기준선 오염 위험, ROS2 단계에서 재검토).
- 2026-09-07 — **데모/검증 스크립트의 RMW 기본값은 `rmw_cyclonedds_cpp` (환경변수 `RMW_IMPLEMENTATION` 이 있으면 그 값 유지).** — 같은 조건(10 Hz, clustered, YOLO 온라인, 297 프레임, 다른 ROS2 세션과 CPU 공유) 에서 Fast DDS(Jazzy 기본) 는 tracker 기준 5 프레임(9, 63, 120, 175, 242 근처 — 약 55 프레임 주기) 을 잃었고 CycloneDDS 는 0 프레임을 잃었다 (`outputs/phase9/rmw_compare_*.log`). 1.4 MB 이미지·2 MB 점군을 KEEP_LAST 10 RELIABLE 로 보낼 때 Fast DDS 의 단편화/흐름제어 특성으로 보인다. launch 파일에는 넣지 않았다 — RMW 는 프로세스 환경의 결정이고 사용자의 다른 워크스페이스(Fast DDS) 와 섞이지 않게 스크립트 단에서만 기본값을 준다. — 대안: Fast DDS XML 프로파일로 `max_message_size`/SHM 세그먼트 조정 (sudo 없이 가능하지만 검증 비용이 큼).

## 팀 G (Phase 8 후반) — 정량 평가

- 2026-09-07 — **평가 규칙은 팀 E `scripts/compare_fusion.py` 와 동일하게 두고 `evaluation/evaluate.py` 가 독립적으로 재계산한다.** — GT 필터(FOV 81.4°, x ≤ 70 m, 5 클래스), `status == "ok"` 예측, BEV 중심 거리 헝가리안 1:1. 결과 A/B 매칭 621/926, BEV mean 1.45/0.63 m 가 comparison.md 와 소수점까지 일치해 두 구현이 서로를 검증한다. — 대안: compare_fusion 을 import 해 재사용 (한쪽이 바뀌면 조용히 같이 바뀌어 교차검증이 안 됨).
- 2026-09-07 — **변형 C 의 예측에는 coasting 트랙(관측 없이 예측만 한 confirmed 트랙)을 기본 포함하고, 제외한 결과를 '참고' 행으로 함께 보고한다.** — tracks.json 의 출력 규약이 그렇고, 실제 소비자(ROS2 시각화·후속 제어)도 coasting 위치를 받는다. 제외 행이 정밀도 0.78 → 0.91 차이를 드러내 "왜 C 가 B 보다 정밀도가 낮은가" 를 수치로 설명한다. — 대안: 항상 제외 (추적의 비용을 숨김).
- 2026-09-07 — **매칭 임계값 2 m 와 4 m 두 개, IoU 는 AABB(yaw = 0) 대 GT 회전 박스, 보고 기준 IoU ≥ 0.5.** — 2 m 는 승용차 길이의 절반이자 nuScenes 검출 지표의 임계값 중 하나. 4 m 표는 A 의 표면 편향(−1.3 m)이 게이트 밖으로 밀어낸 '미검출' 을 드러낸다. KITTI Car 의 공식 3D/BEV 기준 0.7 은 크기 추정이 없는 우리 박스로는 거의 0 이라 0.5 (KITTI 의 Pedestrian/Cyclist 기준) 로 보고하고 한계를 명시. — 대안: IoU 매칭 (A 평가 불가, B/C 도 0.13 만 매칭되어 비교 불가).
- 2026-09-07 — **MT/PT/ML 은 지시서대로 '가장 오래 붙은 단일 track_id 의 커버 비율' 로 정의(≥ 0.8 / ≤ 0.2), 원문(CLEAR MOT, ID 무관) 정의의 커버는 `cov_any` 로 함께 저장.** — 단일 ID 기준이 ID 유지력까지 반영해 추적기 비교에 더 엄격하다. 둘의 차이(0.61 vs 0.63)가 작아 어느 쪽이든 결론은 같다. — 대안: 원문 정의만.
- 2026-09-07 — **MOTA 는 A/B/C 모두에 보고 (A/B 는 IDSW = 0).** — 재현율·정밀도를 한 수치로 묶어 변형 순위를 매기기 쉽고, C 만 IDSW 페널티를 받는 것이 정의상 공정하다. 표에 '검출 MOTA' 임을 명시. — 대안: C 에만 보고.
- 2026-09-07 — **거리 구간별 정밀도는 예측 자신의 BEV 거리로, 재현율·오차·IoU 는 GT 거리로 구간을 나눈다.** — 매칭 쌍은 GT 거리 기준이라 예측 기준 분모와 섞으면 TP > n_pred 가 될 수 있다. 구간별 MOTA 는 정의하지 않는다.
- 2026-09-07 — **집계는 pandas DataFrame(pairs / preds / gts 세 표) 위에서 한다.** — 프레임·구간·클래스·tracklet 별 groupby 가 한 줄이 되고, 같은 표를 CSV/플롯/간트차트가 공유한다. 핵심 매칭·오차·IoU 계산은 팀 D 의 numpy 함수 그대로. — 대안: 순수 numpy/dict (구간·클래스별 집계 코드가 세 벌).
- 2026-09-07 — **`scripts/run_pipeline.py` 의 `runtime_meta` 에 `load_avg_1_5_15` 한 줄 추가, 세 변형을 load 1.6~2.6 에서 재측정.** — 팀 E 측정치는 load 7~10 (사용자 실험 + ROS2 노드) 에서 cluster p95 가 85 ms 로 부풀어 있었다. 재측정 후 JSON 결과는 바이트 단위로 동일(결정적)하고 total median A 11.3 / B 19.4 / C 21.1 ms, p95 13 / 26 / 30 ms. runtime.md 는 변형별 측정 시각·load 를 표로 남긴다. — 대안: 팀 E 수치를 그대로 인용 (mean/p95 가 오해를 부름).
- 2026-09-07 — **플롯 색은 변형별 고정 (A 파랑 #2a78d6, B 주황 #eb6834, C 청록 #1baf7a), 그림 안 글자는 영어.** — 색각 이상에서도 구분되도록 검증된 팔레트의 앞 3 슬롯이고, 6장 모두 같은 색 = 같은 변형. 간트차트의 track_id 색은 tab20 순환이지만 ID 숫자를 막대 위에 직접 써서 색만으로 읽지 않게 했다. 기본 글꼴에 한글 글리프가 없어 플롯 텍스트는 영어, 설명은 마크다운 한국어.
- 2026-09-07 — **노트북 06 은 `KalmanTracker.step` 을 부르기 직전에 예측·비용 행렬·할당을 밖에서 재현해 기록한다 (`tracker.all_tracks` 를 deepcopy).** — step 은 내부 상태를 제자리에서 바꾸므로 "게이트 안/밖" 을 보려면 같은 계산을 한 번 더 해야 한다. 트래커 코드는 수정하지 않았다. — 대안: 트래커에 디버그 훅 추가 (팀 C 파일 수정).
