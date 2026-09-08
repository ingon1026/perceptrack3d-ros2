# Phase 10 학습 노트 — C++ 투영·ROI·frustum 컴포넌트와 성능 엔지니어링

목표: "맹목적으로 전부 C++ 로 다시 쓰지 않는다." 프로파일로 병목 하나를 고르고, 현대 C++ 로 옮긴 뒤,
Python 기준 구현과 **수치가 같은지** 먼저 확인하고, 그 다음에야 **얼마나 빨라졌는지** 정직하게 잰다.

관련 파일
- C++: `cpp/include/pt3d/projection.hpp`, `cpp/src/projection.cpp`, `cpp/src/bindings.cpp`, `cpp/tests/test_projection.cpp`, `cpp/CMakeLists.txt`
- Python: `src/perceptrack3d/geometry/projection_cpp.py` (래퍼), `tests/test_cpp_projection.py`
- 스크립트: `scripts/build_cpp.sh`, `scripts/profile_projection.py`, `scripts/benchmark_projection.py`
- 결과: `outputs/phase10/profile_python.txt`, `outputs/phase10/benchmark.md`

---

## 1. 왜 이 컴포넌트를 골랐나 (프로파일 근거)

`scripts/profile_projection.py` (프레임 10장, 중앙값):

| 단계 | ms |
|---|---|
| LiDAR .bin 로드 (`np.fromfile`) | 0.73 |
| 이미지 PNG 로드 (`cv2.imread`) | 8.86 |
| 투영 `project_velo_to_image` (numpy) | 2.71 |
| YOLO 2D 검출 (Phase 4 runtime.md) | 13.84 |

- 투영 1회는 프레임당 약 26 ms 중 10% 다. 그 자체로는 두 번째로 큰 순수 파이썬 단계일 뿐이다.
- 단독 함수 `point_filters.frustum_mask` 는 호출마다 `project_velo_to_image` 를 다시 부른다. 검출(평균 4.5개)마다 이 함수를 부르는 구조라면
  투영이 5.5 번 일어나 약 17 ms 로 검출(신경망)보다 커진다 (벤치마크 C3). **다만 팀 E 의 실제 `frustum_fusion.py` 는 이미 프레임당 투영을 1회만 하고
  검출별로는 `uv_in_box` 만 적용한다** (C2 패턴, 3.5 ms). 그래서 실제 파이프라인 기준 C++ 의 이득은 C2 대비 5.3배이고, C3 의 25배는 "검출마다 재투영하면 이렇게 된다" 는 경고용 수치다.
- cProfile 을 보면 투영 시간의 절반이 `to_homogeneous` (float32→float64 승격 복사 + `hstack` 으로 (N,4) 배열 생성) 에 있고,
  나머지는 행렬곱과 대여섯 개의 임시 불리언/실수 배열이다. 계산량(점당 곱셈 12번)이 아니라 **메모리 왕복**이 비용이다.
- 이런 "점마다 단순한 연산을 한 번씩, 그러나 큰 임시 배열을 여러 번 만드는" 패턴이 C++ 단일 루프로 옮겼을 때 이득이 확실한 전형이다.
  반대로 YOLO, DBSCAN, PNG 디코딩은 이미 C/C++ 커널이 돌고 있어 옮겨도 얻을 게 없다 (§7).

선택: **투영 + ROI 마스크 + K개 박스 frustum 마스크** 세 함수. 셋 다 "점 배열 한 바퀴" 구조라 같은 데이터 레이아웃을 공유한다.

## 2. 데이터 레이아웃과 zero-copy

Python 쪽 점 배열은 로더가 주는 `(N, 4) float32` C-연속(row-major) 배열이다: 메모리에 `x0 y0 z0 r0 x1 y1 z1 r1 ...` 순으로 놓인다.

1. **numpy → C++ (복사 없음)**. `bindings.cpp:19` 의
   `py::array_t<float, py::array::c_style | py::array::forcecast>` 는 "float32 이고 C-연속이면 그대로 쓰고, 아니면 변환해서 복사" 라는 뜻이다.
   로더 출력은 이미 조건을 만족하므로 포인터만 넘어온다.
2. **(N,4) 의 앞 3열만 보기**. `bindings.cpp:23–32` 에서 `Eigen::Map<const PointMatrix, 0, Eigen::OuterStride<>>(data, N, 3, OuterStride(4))` 로
   "행 간격(outer stride) 은 4 이지만 열은 3개만" 인 뷰를 만든다. `pts(i, 0)` 은 `data[i*4 + 0]` 을 읽는다.
   그래서 파이썬에서 `pts[:, :3]` 으로 **자르지 말아야** 한다 — 자르면 비연속 뷰가 되어 오히려 복사가 생긴다 (`projection_cpp.py:35–40`).
3. **C++ 함수 시그니처**. `projection.hpp:31` 의 `PointsRef = Eigen::Ref<const PointMatrix, 0, Eigen::OuterStride<>>` 가 그 뷰를 받는 타입이다.
   `OuterStride<>` 를 빼면 Eigen 은 stride 3 만 허용해 (N,4) 입력을 몰래 임시 복사한다. gtest `StridedN4InputIsZeroCopyAndEqualToN3`
   (`test_projection.cpp:92`) 가 `ref.data() == n4.data()` 로 복사가 없음을 실제로 검증한다.
4. **C++ → numpy (복사 없음)**. 결과 `std::vector` 를 힙으로 옮기고 `py::capsule` 이 소유하게 한 뒤 numpy 배열이 그 버퍼를 보게 한다
   (`bindings.cpp:40–46`). 배열이 사라질 때 capsule 소멸자가 vector 를 delete 한다.
5. mask 는 `uint8` 0/1 로 돌아오고 래퍼가 `.view(np.bool_)` 로 재해석한다 (`projection_cpp.py:52`). numpy bool 도 1 바이트라 복사가 없다.

## 3. RAII / const / 값·참조 의미 — 코드 어디에 있나

| 원칙 | 위치 | 설명 |
|---|---|---|
| RAII (자원 = 객체 수명) | `projection.hpp:37–42` `ProjectionResult` 의 `std::vector` 멤버 | 메모리를 벡터가 소유. 코어 라이브러리에는 `new`/`delete` 가 한 줄도 없다. 예외가 나도 자동 해제. |
| RAII + 스마트 포인터 | `bindings.cpp:41–44` | 벡터를 `std::make_unique` 로 힙에 옮기고, capsule 생성이 성공한 뒤에만 `release()` 로 소유권을 넘긴다. capsule 생성이 실패(예외)하면 `unique_ptr` 이 정리한다. **이 프로젝트에서 스마트 포인터가 실제로 필요한 유일한 곳.** 코어에서는 힙 객체를 함수 밖까지 공유할 일이 없어 쓰지 않았다 (필요 없어서 안 씀). |
| const-correctness (입력) | `projection.hpp:50, 64, 79` | 모든 입력은 `const PointsRef&`, `const ProjMatrix&`, `const RoiParams&`, `const std::vector<Box2D>&`. 함수가 입력을 바꾸지 않음을 컴파일러가 보증한다. |
| const 멤버 | `projection.hpp:55–61` `RoiParams` | 생성 후 바뀌지 않는 설정값. `RoiParams{0.f, 70.f, 40.f, -3.f, 3.f}` 집합 초기화. 실수로 `roi.x_max = ...` 하면 컴파일 오류. |
| const 지역 변수 | `projection.cpp:21–24, 90, 96` | `const double x = pts(i,0)`, `const Box2D& box = boxes[b]`. "이 값은 여기서 끝까지 안 바뀐다" 를 읽는 사람과 최적화기에 알린다. |
| const 멤버 함수 + noexcept | `projection.hpp:42`, `projection.cpp:18–19` | `numProjected() const noexcept`, `projectOne(...) noexcept`. 예외를 던지지 않는 순수 계산. |
| 값 의미 (반환) | `projection.cpp:53, 67, 101` | 결과는 값으로 반환한다 (`return result;`). C++17 은 지역 변수를 값으로 돌려줄 때 복사를 생략한다(NRVO). 호출자는 결과를 온전히 소유하고, 입력과 별개의 수명을 가진다. |
| 값 의미 (작은 구조체) | `projection.cpp:18–31` `Projected` | 점 하나의 결과(double 3개 + bool)는 값으로 주고받는 게 가장 싸고 명확하다. |
| 참조 의미 (뷰) | `projection.hpp:31`, `bindings.cpp:23, 89` | `PointsRef`/`Map` 은 소유하지 않는 뷰. 저장하지 않고 호출 동안만 쓴다. 뷰의 수명은 원본(numpy 배열)에 묶여 있다. |
| PIC / 플래그 고정 | `CMakeLists.txt:14, 23` | `-O2 -DNDEBUG` 고정(벤치마크 재현성), 정적 라이브러리를 `.so` 에 넣기 위한 `POSITION_INDEPENDENT_CODE`. |

## 4. 수치 일치 검증 — 어떻게 "같다" 를 증명했나

Python 기준 구현은 float64 로 계산한다. C++ 도 같은 결과를 내려면:

- `P` 를 float64 (`Eigen::Matrix<double,3,4>`, `projection.hpp:34`) 로 받고 점을 float32→double 로 정확히 승격해 double 로 계산한다.
  P 를 float32 로 받으면 원소 반올림 오차(~1e-7 상대)가 u 에서 ~1e-4 px 가 되어, 경계 `u < W` 판정이 1242 픽셀 경계 근처 점에서 뒤집힐 수 있다.
  실제 측정: 20 프레임 uv 최대 차이 **0.0**, mask 불일치 0.
- 규칙을 한 글자씩 맞췄다: `d > 0` (NaN 도 제외), `0 <= u < W`, `0 <= v < H` (하한 포함, 상한 제외, 반올림 없음).
- frustum 은 Python 이 float32 `uv` 에 박스 조건을 걸므로, C++ 도 u,v 를 float32 로 반올림한 값을 double 박스 좌표와 비교한다 (`projection.cpp:96`).
- ROI 는 팀 E `roi_filter` 가 점을 float64 로 캐스팅한 뒤 float64 임계값과 비교하므로 `RoiParams` 멤버를 double 로 두고 점을 double 로 승격해 비교한다.
  처음에는 "numpy 는 float32 배열과 파이썬 float 를 float32 로 비교한다(NEP 50)" 는 가정으로 float 멤버를 썼는데, 팀 E 확인 결과 캐스팅이 먼저였다.
  현재 설정값(0, 70, 40, ±3)은 float32 로 정확해 결과가 같지만, 0.1 처럼 float32 로 표현되지 않는 임계값이면 float 비교는 경계 점을 잘못 포함한다
  (gtest `Roi.ThresholdNotRepresentableInFloat32IsComparedInDouble`). 교훈: **기준 구현의 dtype 경로를 한 줄씩 확인하고 가정하지 말 것.**

검증 계층
1. gtest 10개 (`cpp/tests/test_projection.cpp`): 손계산 핀홀 점, 카메라 뒤/평면 위 제거, 경계 포함·제외, 빈 입력, (N,4) zero-copy, ROI 경계, float32 로 표현되지 않는 ROI 임계값(0.1), frustum 포함 경계·이미지 밖·카메라 뒤·박스 0개.
2. pytest 16개 (`tests/test_cpp_projection.py`): 프레임 0 실제 데이터 Python vs C++ (mask 완전 일치, uv atol 1e-3, depth rtol 1e-5), 합성 격자, (N,3)/(N,4) 동일, ValueError, ROI/frustum 은 numpy 참조 **및 팀 E `roi_filter`/`frustum_mask`** 와 완전 일치 (shrink 0.0, 0.1), 팀 E 의 경계 점 (20, 1.30, 0) 케이스, float32 로 표현되지 않는 ROI 임계값에서 팀 E 와 일치, shrink 범위 ValueError.
3. 벤치마크 스크립트가 20 프레임 전부에 대해 일치를 다시 센다 (불일치 0).

## 5. 벤치마크 결과와 해석

`outputs/phase10/benchmark.md` (i7-14700K, g++ 13.3 `-O2`, 프레임 20장, 평균 123,667 점, 검출 평균 4.55개/프레임, 단일 스레드). 두 번 실행했다 — 1차 20회 반복(load 30.8), 2차 30회 반복(load 26.7). 둘 다 **다른 팀 프로세스 부하 하에서 측정**이라 median 만 대표값으로 쓴다. 아래 표는 1차, 괄호는 2차:

| 시나리오 | Python median (ms) | C++ median (ms) | speedup |
|---|---|---|---|
| A. 투영 (전체 프레임) | 3.235 (3.715) | 0.375 (0.383) | 8.6x (9.7x) |
| B. ROI 마스크 | 1.133 (1.341) | 0.180 (0.186) | 6.3x (7.2x) |
| C1. frustum K박스, Python naive (박스마다 재투영) | 16.579 (15.394) | 0.671 (0.703) | 24.7x (21.9x) |
| C2. frustum K박스, Python once (투영 1회 + 박스별 비교) | 3.531 (4.363) | 0.671 (0.703) | 5.3x (6.2x) |
| C3. frustum K박스, 팀 E `frustum_mask` 박스마다 호출 | 17.113 (17.879) | 0.671 (0.703) | 25.5x (25.4x) |
| D. C++ 호출 고정 오버헤드 (점 10개) | - | 0.031 (0.037) | - |

해석 (정직하게)
- **개선 있음.** 단, 이유는 "C++ 가 산술이 빠르다" 가 아니다. 점당 곱셈 12번은 어느 언어든 마이크로초 단위다. 이득은 numpy 판이 매 호출마다
  float64 승격 복사(3 MB), `hstack`(4 MB), 행렬곱 결과(3 MB), `np.full` 두 개, 불리언 임시 대여섯 개 등 10 MB 이상을 쓰고 읽는 데 반해
  C++ 은 입력 1.5 MB 를 한 번 읽고 결과 0.35 MB 만 쓰기 때문이다. 즉 **메모리 트래픽과 할당 횟수**의 차이다.
- C1/C3 의 25배 중 대부분은 **알고리즘**(투영을 한 번만) 에서 온다. 순수 Python 으로도 투영을 한 번만 하면(C2) 17 → 3.5 ms 로 줄고,
  그 위에 C++ 이 5배를 더 얹는다. 팀 E 의 `frustum_fusion.py` 는 이미 C2 방식이므로, 실제 파이프라인에 적용할 때 기대할 수 있는 이득은 **5.3배(3.5 → 0.67 ms)** 다.
  25배는 "검출마다 `frustum_mask` 를 부르면 생기는 낭비" 를 보여 주는 대조군이다.
- D 의 0.03 ms 는 pybind11 인자 변환 + 결과 numpy 배열 3개 생성 비용이다. A 의 C++ 시간 0.375 ms 중 8% 로, 지금 규모(12만 점)에서는 무시할 수 있지만
  점 100개짜리 호출을 수천 번 하는 구조라면 이 오버헤드가 지배한다 (그런 구조는 C++ 로 옮겨도 이득이 없다).
- 두 실행의 speedup 이 A 8.6/9.7, B 6.3/7.2, C2 5.3/6.2, C3 25.5/25.4 로 일관된다 (실행 이력은 benchmark.md 끝의 표, 원본은 `benchmark_runs.csv`). 1차 실행에서 Python p95 (12.7 ms) 가 median 의 4배로 튀었는데 C++ 은 1.5배였다. 측정 중 다른 팀 에이전트가 같은 머신에서 작업 중이었고(load average 27~35, 논리 코어 28),
  numpy 판은 큰 임시 배열을 매번 할당·해제하므로 외부 부하와 할당 지터에 더 민감하다. 절대값보다 median 비를 보는 게 맞다.
- 프레임 예산으로 환산하면: 실제 융합 경로(C2 패턴, 투영 1회 + 박스별 비교 ≈ 3.5 ms) 를 C++ `frustum_masks_cpp` 로 바꾸면 약 2.9 ms 절약, ROI 까지 합쳐 약 3.8 ms/프레임.
  로드 9.6 + 검출 13.8 ms 는 그대로다. 파이프라인(로드+검출+투영/ROI/frustum ≈ 28 ms) 이 ≈ 24 ms 가 되는 정도이지, "10배 빨라진 파이프라인" 이 아니다.
- 안 한 것: `-march=native`(AVX2/FMA) 와 멀티스레드. 재현성(다른 PC 에서 같은 바이너리 동작) 을 우선했다. 12만 점 0.37 ms 는 이미 예산 안이라 필요 시에만 켠다.

## 6. 빌드와 사용

```bash
scripts/build_cpp.sh        # cmake(Release, venv python, pybind11 경로) → build → ctest
.venv/bin/python -m pytest -q tests/test_cpp_projection.py
.venv/bin/python scripts/profile_projection.py     # outputs/phase10/profile_python.txt
.venv/bin/python scripts/benchmark_projection.py   # outputs/phase10/benchmark.md
```

```python
from perceptrack3d.geometry.projection_cpp import project_velo_to_image_cpp, roi_mask_cpp, frustum_masks_cpp
uv, depth, mask = project_velo_to_image_cpp(points_velo, calib, calib.image_shape)   # (N,4) 그대로 넘긴다
masks = frustum_masks_cpp(points_velo, boxes_xyxy, calib, calib.image_shape, shrink=0.1)  # (K, N) bool
```
확장 모듈이 없으면 `import` 시 빌드 명령이 적힌 `ImportError` 가 난다 (`projection_cpp.py:27`). ROS2 C++ 노드는 `pt3d_core` 정적 라이브러리를 직접 링크하면 된다 (Python 의존 없음).

## 7. C++ 로 옮기면 안 되는 것 (이 프로젝트 기준)

| 후보 | 옮기지 않는 이유 |
|---|---|
| YOLO 추론 (13.8 ms) | 이미 torch 의 C++ 커널. 다시 쓰면 같은 속도에 버그만 는다. 빠르게 하려면 모델·해상도·ONNX/TensorRT 같은 다른 축. |
| PNG 디코딩 (8.9 ms) | `cv2.imread` 는 libpng(C). 언어가 아니라 포맷 문제 — 비압축 캐시나 프리페치가 답. |
| DBSCAN / RANSAC | scikit-learn·Open3D 내부가 C++. |
| 칼만 필터 / 헝가리안 | 6×6 행렬 × 객체 5개 = 마이크로초. scipy `linear_sum_assignment` 도 C. 옮겨도 잴 수 없는 이득. |
| 평가·시각화 | 오프라인. 정확성과 가독성이 우선. |
| 아직 튜닝 중인 로직 (임계값, surface_offset 등) | 사양이 흔들리는 코드는 Python 에 두고, 굳은 뒤 옮긴다. |
| 점 100개짜리 호출을 수천 번 하는 구조 | 호출당 0.03 ms 오버헤드가 지배. 옮기기 전에 배치(batch) 형태로 API 를 바꿔야 한다 — frustum 을 K 박스 한 번에 받게 한 이유. |

## 8. 자기 점검 질문

1. `(N,4)` 배열을 파이썬에서 `pts[:, :3]` 으로 잘라 넘기면 왜 오히려 복사가 생기는가? C++ 쪽은 어떤 방식으로 앞 3열만 읽는가?
2. `P` 를 float32 로 받으면 결과의 어느 부분이 Python 과 달라질 수 있으며, 왜 uv 오차 1e-4 px 가 mask 불일치로 이어질 수 있는가?
3. frustum 벤치마크에서 25배 중 어디까지가 "알고리즘" 이고 어디부터가 "언어" 인가? 순수 Python 으로 얻을 수 있는 최대 개선은 얼마인가?
