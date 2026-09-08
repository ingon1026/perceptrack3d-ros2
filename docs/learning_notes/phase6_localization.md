# Phase 6 학습 노트 — 3D 위치·박스 개선 (ROI, 지면 제거, 프러스텀, DBSCAN, 표면 오프셋, AABB/OBB)

코드: `src/perceptrack3d/fusion/point_filters.py`, `fusion/frustum_fusion.py::fuse_frame_clustered, surface_to_center_offset`
노트북: `notebooks/05_camera_lidar_fusion.ipynb` (6.1–6.6 절) / 비교: `outputs/phase6/comparison.md` (`scripts/compare_fusion.py`)

## 0. 여섯 가지 질문

| 질문 | 답 |
|---|---|
| 입력은? | Phase 5 와 같음 (`points_velo` (N,4), `Detection2D` 리스트, calib, image_shape, cfg) |
| 출력은? | `Object3D` 에 `size` (3,) [l, w, h] m, `yaw` (AABB 면 0.0, OBB 면 rad) 추가. `method = "clustered"` |
| 좌표계는? | 모든 필터·클러스터·박스는 **Velodyne** (x 전방, y 좌, z 상). 프러스텀 판정만 픽셀 |
| 핵심 알고리즘은? | ROI → RANSAC 지면 제거 → 프러스텀 → DBSCAN → 최대 클러스터 → 중앙값 + 시선 방향 오프셋 → AABB(또는 BEV PCA OBB) |
| 실패 사례는? | 클러스터 오선택(배경이 더 큰 덩어리), 5~7 점짜리 먼 물체가 no_cluster, 클래스 prior 불일치(Van→car), 크기 과소 추정 |
| 평가는? | 297 프레임: BEV 오차 mean 1.45 → **0.63 m**, 깊이 bias −1.29 → **−0.30 m**, 재현율 0.45 → **0.66**, 정밀도 0.53 → **0.84** |

## 1. 파이프라인과 점 수 변화 (frame 174)

```
전체 119 911 ─ROI→ 61 669 ─지면 제거→ 21 010 ─투영·축소 박스→ (검출별 13~97) ─DBSCAN→ 1~4 클러스터 + noise
```

### 1.1 ROI (`roi_filter`)
x ∈ [0, 70], |y| ≤ 40, z ∈ [−3, 3] m. 카메라 뒤(x < 0) 는 어차피 투영되지 않고, 70 m 밖은 점이 너무 성기다. RANSAC 입력을 절반으로 줄여 속도도 번다.

### 1.2 지면 제거 (`remove_ground`: numpy RANSAC 기본, Open3D `segment_plane` 옵션)
RANSAC: 점 3개를 무작위로 뽑아 평면을 만들고, 평면에서 `distance_threshold` (0.2 m) 안에 있는 점(inlier) 수를 센다. 200 번 반복해 inlier 가 가장 많은 평면이 지면.
```
평면: a·x + b·y + c·z + d = 0,  (a, b, c) 단위 법선 (c > 0 으로 통일)
점의 거리: |a·x + b·y + c·z + d|
원점 아래 지면 높이: z = −d / c  →  KITTI 는 −1.67 m (Velodyne 높이 1.73 m 와 일치)
```
- 안전장치: 법선 z 성분 < 0.8 (벽을 잡음) 이면 제거하지 않는다. 점 < 3 이면 그대로.
- 기본 `method: ransac` 은 **numpy 구현** (`ransac_plane`): 가설 200개를 (M,3)@(3,50) 행렬곱으로 한 번에 평가, 점수는 시드로 뽑은 12 000 점 부분집합, 최종 inlier 로 SVD 최소제곱 재적합. 시드 고정 → 완전히 결정적 (테스트로 확인). `method: open3d` 는 `segment_plane` 인데 0.19 는 시드를 걸어도 실행마다 inlier 가 ±1000 점 달라진다.
- 왜 필요한가: 박스 아래쪽에 걸리는 도로 점이 없으면 DBSCAN 이 물체와 도로를 한 덩어리로 잇지 않는다. 또 지면 점이 클러스터 중앙값을 아래로 끌지 않는다.
- 함정: 차 하부 0.2 m 는 지면으로 분류된다 (테스트 장면에서 상자 바닥을 지면 +0.35 m 로 둔 이유).

### 1.3 프러스텀 (`frustum_mask`, 내부적으로 `uv_in_box`)
Phase 5 와 같은 축소 박스 판정에 `ROI ∧ 비지면` mask 를 AND 한다. 전체 점을 한 번만 투영하고 검출마다 마스크만 바꾼다 (프레임당 3 ms).

### 1.4 DBSCAN (`cluster_points`, Open3D `cluster_dbscan`)
- `eps` = 0.8 m: 이웃 반경. HDL-64 의 수직 링 간격은 60 m 에서 약 0.42 m 이므로 0.6 은 먼 차를 두 조각으로 나눴다.
- `min_points` = 5: 핵심점이 되기 위한 이웃 수. 8 이면 5~7 점짜리 먼 차가 전부 `no_cluster` 였다 (151 → 118 건).
- 라벨 −1 = noise. 클러스터 0 개면 `status = invalid, reason = no_cluster`.

### 1.5 클러스터 선택 (`cluster_select`)
| 규칙 | 설명 | 297 프레임 매칭 (2 m) |
|---|---|---|
| `largest` (기본) | 점 수 최대 | **926** |
| `nearest` (`cluster_min_ratio` 0) | 깊이 중앙값 최소 | 867 (Open3D RANSAC 기준) |
| `nearest` (ratio 0.3) | 최대 클러스터의 30% 이상인 것 중 가장 가까운 것 | 884 |

이 시퀀스에서는 가까운 작은 덩어리(지면 잔여, 기둥, 풀) 가 물체보다 자주 나타나 `largest` 가 이긴다.
반대로 **배경이 물체보다 큰 덩어리** 인 경우 `largest` 는 틀린다: frame 174 det6 (37 m 차 → 51 m 배경, 오차 17 m), 합성 테스트 `test_dense_background_failure_and_nearest_rule`.
어느 쪽이 맞는지는 장면 통계에 달렸으므로 두 규칙을 설정으로 남겼다.

## 2. 표면 → 중심 오프셋 (`surface_offset`)

Phase 5 노트 4절의 표면 편향을 다룬다. 클러스터 중앙값 c 는 보이는 면 위에 있다.

```
u       = c_xy / |c_xy|                        센서(Velodyne 원점) → 물체 BEV 단위벡터
extent  = max(p·u) − min(p·u)                  클러스터 점을 u 에 투영한 퍼짐 = "이미 보이는 깊이"
push    = max(0, half_depth[class] − extent/2)
c_xy   += u · push                             (z 는 그대로)
```
- `half_depth_m`: car 2.0 (KITTI Car 길이 평균 ≈ 4 m), truck 3.0, bus 4.0, person 0.35, bicycle 0.85. 이 시퀀스 GT 에서 유도.
- 옆면까지 보이면(extent ≈ 4 m) 밀지 않는다 → 이미 중심 근처.
- ablation (297 프레임): 오프셋 없음 BEV 1.39 m / bias −1.24 → 오프셋 후 0.63 m / −0.30. **Phase 6 개선의 대부분이 이 한 줄**이다. 클러스터링만으로는 raw 와 오차가 같다 (배경 오염이 이 시퀀스에서 드물기 때문).
- 한계: (1) 클래스 prior 의존 — Van(길이 5~7 m) 이 car 로 검출되면 ~1 m 덜 민다 (남은 bias −0.30 의 주원인). (2) 잘린 박스는 시선 방향이 물체 중심을 지나지 않는다. (3) car half 2.2 로 올리면 bias −0.11 까지 가지만 이 시퀀스에 맞춘 값이라 채택하지 않았다.

## 3. 3D 박스: AABB 와 OBB

- `aabb_from_points`: min/max. `size = [l(x), w(y), h(z)]`, `yaw = 0`.
- `obb_from_points_bev`: BEV 점의 공분산 행렬 고유벡터 중 최대 고유값 방향이 yaw. 점을 `Rz(−yaw)` 로 돌려 min/max, 중심을 `Rz(+yaw)` 로 되돌린다. yaw ∈ [−π/2, π/2) (앞뒤 구분 불가). 합성 사각형에서 size ±0.08 m, yaw ±0.03 rad 복원 (테스트).
- **실데이터 한계**: 멀리서 뒷면만 보이면 최대 분산 방향이 **폭** 방향이라 yaw ≈ ±90° 가 나온다 (frame 174 det0: +80.7°). 점 10~30 개짜리 먼 차는 크기가 0.4 × 1.4 m 로 실제(4 × 1.8) 보다 훨씬 작다. 이 시퀀스에서 크기 추정은 신뢰할 수 없고, 주 산출물은 **중심**이다. `use_obb` 기본값은 false.
- `max_extent_m` (15 m): l 또는 w 가 이보다 크면 `invalid/oversize` (벽·울타리와 붙은 클러스터).

## 4. 수치로 본 개선 (`outputs/phase6/comparison.md` 요약, 매칭 2 m)

| 변형 | 매칭/GT | 재현율 | 정밀도 | BEV mean | BEV median | 깊이 bias |
|---|---|---|---|---|---|---|
| A raw | 621/1394 | 0.45 | 0.53 | 1.45 | 1.48 | −1.29 |
| B0 clustered, 오프셋 없음 | 627 | 0.45 | 0.57 | 1.39 | 1.42 | −1.24 |
| B clustered (최종) | 926 | 0.66 | 0.84 | 0.63 | 0.56 | −0.30 |

4 m 매칭으로 넓히면 raw 도 971 매칭 (재현율 0.70) 이 된다 — raw 의 문제는 "못 찾음" 이 아니라 "1.5 m 어긋남" 이다.
나빠지는 조건: 배경이 더 큰 덩어리일 때(largest 오선택), Van/Truck 을 car 로 검출할 때(prior 불일치), 잘린 박스.

## 5. 런타임 (i7-14700K, 297 프레임, 단일 프로세스, `outputs/phase6/runtime.json`)

아래 표는 다른 프로세스가 없을 때의 값이다. 현재 저장된 `runtime.json` 은 팀 H 의 ROS2 노드가 동시에 실행 중(load ≈ 10)일 때 측정돼 cluster 단계 p95 가 80 ms 까지 올라가 있다 (Open3D DBSCAN 이 멀티스레드라 경합에 민감). 평가 팀은 머신이 한가할 때 `run_pipeline.py` 를 다시 돌려 집계할 것.

| 단계 | mean ms | 비고 |
|---|---|---|
| load (png + bin) | 8.3 | 디스크 |
| filters (ROI + RANSAC) | ~10 | numpy RANSAC 200 가설 (Open3D 는 6.7 ms 이지만 비결정적) |
| projection | 2.8 | (N,4)@(4,3) 행렬곱 |
| cluster (프러스텀 + DBSCAN + 박스, 검출 전체) | 3.0 | 검출당 수십~수백 점만 DBSCAN |
| 합계 | ~25 | 목표 < 500 ms 대비 여유. 전체 297 프레임 약 7~8 s (다른 프로세스가 없을 때) |

## 6. 튜닝 가이드

| 키 | 기본 | 언제 바꾸나 |
|---|---|---|
| `ground.distance_threshold` | 0.2 | 경사로·요철이 많으면 0.3 (단, 차 하부 점도 더 잃음: 스윕에서 매칭 781 로 감소) |
| `cluster.eps` | 0.8 | 가까운 물체끼리 붙으면 0.6, 먼 물체가 쪼개지면 1.0 |
| `cluster.min_points` | 5 | `fusion.min_points` 와 같게 유지 |
| `cluster_select` | largest | 배경이 크게 들어오는 도심 장면이면 nearest + `cluster_min_ratio` 0.3~0.5 |
| `surface_offset.half_depth_m` | car 2.0 | 데이터셋 클래스별 길이 평균의 절반 |
| `use_obb` | false | 가까운(< 20 m) 물체가 많고 옆면이 보일 때만 |
