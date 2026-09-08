# Phase 5 학습 노트 — 카메라–LiDAR 융합 기준선 (2D 박스 + LiDAR 깊이 통계)

코드: `src/perceptrack3d/fusion/frustum_fusion.py::fuse_frame_raw`, `fusion/point_filters.py::shrink_box, uv_in_box, frustum_mask`
노트북: `notebooks/05_camera_lidar_fusion.ipynb` (5.1–5.4 절) / 테스트: `tests/test_fusion.py` / 실행: `scripts/run_pipeline.py --variant raw`

## 0. 여섯 가지 질문 (learning-first)

| 질문 | 답 |
|---|---|
| 입력은? | `points_velo` (N, 4) float32 Velodyne [x, y, z, reflectance]; `list[Detection2D]` (xyxy = image_02 픽셀); `KittiCalibration`; `image_shape` (H, W) |
| 출력은? | `list[Object3D]` — 검출과 같은 순서. `center` (3,) float64 **Velodyne** m 또는 None, `n_points`, `status`, `reason`, `truncated`. raw 는 `size = None` |
| 좌표계는? | 점 **선택**은 이미지 픽셀(투영 결과 uv), 깊이는 정류 카메라 z (m), **중심**은 Velodyne 으로 돌려 준다 (원본 점을 그대로 중앙값 내므로 변환 불필요) |
| 핵심 알고리즘은? | 축소 박스 안 점 → 깊이 하위 30% 값(대표 깊이) → ±1.5 m 밴드 → 좌표별 **중앙값** |
| 실패 사례는? | 빈 박스(멀고 작은 물체), 희소, 배경 오염(뒤 건물이 30% 넘을 때), 겹친 박스(같은 차 2개), 잘린 박스, **표면 편향**(중심이 GT 보다 ~1.3 m 가까움) |
| 평가는? | GT(tracklet) 중심과 BEV 거리 ≤ 2 m 매칭 → 중심 오차 mean 1.45 m, 깊이 bias −1.29 m, 재현율 0.45 (297 프레임) |

## 1. 왜 "박스 안 점" 인가

2D 검출은 픽셀 (u, v) 만 알고 깊이를 모른다. LiDAR 점은 3D 를 알지만 무엇인지 모른다.
Phase 3 에서 LiDAR 점을 이미지에 투영했으므로, **"박스 안에 떨어지는 점" = "그 물체(이거나 그 뒤 배경) 위의 점"** 이다.
3D 로 보면 카메라 중심에서 2D 박스 네 모서리로 뻗는 사각뿔(**프러스텀, frustum**) 안의 점이다. 판정은 3D 기하 대신 투영으로 한다.

```
points_velo (N,3) ─project_velo_to_image→ uv (M,2), depth (M,), mask (N,)
                                           └ uv_in_box(uv, shrink_box(xyxy, 0.1)) → 박스 안 점 (로컬 인덱스 sel)
                                           └ idx = flatnonzero(mask); 원본 점 = points_velo[idx[sel]]
```

`shrink_box`: 박스를 각 변 10% 안쪽으로 줄인다 (가로·세로 80%). 박스 가장자리에는 도로·건물·나무가 걸리기 쉽다.
스윕 결과: shrink 0 → 매칭 645, 0.1 → 621 (raw 는 큰 차이 없음), clustered 에서는 0.2 가 정밀도를 올리지만 재현율을 잃는다.

## 2. 대표 깊이: 평균 vs 중앙값 vs 하위 백분위

박스 안 점의 깊이 분포는 보통 **쌍봉**이다: 물체(앞) + 배경(뒤). frame 174 det1 은 112 점 중 40% 가 30~45 m 뒤편 점이다.

| 통계 | 값 (det1) | 성질 |
|---|---|---|
| mean | 29.9 m | 배경에 끌려감 |
| median | 27.5 m | 배경 < 50% 면 버티지만 40% 면 이미 벗어남 |
| p30 (하위 30%) | 24.3 m | 전경 우선. "박스 안에서 가까운 것이 물체" 라는 가정 |

기준선 규칙 (`configs/kitti.yaml` fusion):
```
d_rep  = percentile(depth[sel], depth_percentile=30, method="nearest")   # 실제 점 값 → 밴드가 비지 않음
used   = sel[|depth − d_rep| ≤ depth_band_m=1.5]
center = median(points_velo[idx[used]], axis=0)                            # 좌표별 중앙값 (Velodyne)
```
`method="nearest"` 가 중요하다: 보간된 백분위 값은 실제 점 사이에 떨어질 수 있어 (깊이 [10, 20, 30, ...] 에서 p30 = 22) 밴드 안에 점이 하나도 없는 버그가 있었다.

스윕 (297 프레임, 2 m 매칭): p10 → 569 매칭, **p30 → 621**, p50 → 655. p50 이 살짝 낫지만 배경이 많은 박스에서 더 취약하므로 30 을 유지하고, 근본 해결은 Phase 6 (클러스터·오프셋) 에 둔다.

## 3. status 와 실패 처리

| status | 조건 | center |
|---|---|---|
| `empty` | 축소 박스 안 점 0 | None |
| `sparse` | 점 < `min_points` (5) | None (`n_points`, `point_indices` 는 기록) |
| `invalid` | 중복 제거에서 탈락 (`reason = duplicate_of_k`) | None |
| `ok` | 그 외 | (3,) |

- **중복 제거**: YOLO 가 같은 차에 car + truck 처럼 박스 2개를 낼 수 있다 (팀 B). BEV 중심 거리 < `dedup_distance_m` (1 m) 이면 confidence 낮은 쪽을 invalid. person + bicycle 이 하나(KITTI Cyclist) 로 합쳐지는 부수 효과도 있다.
- **잘린 박스**: x1 ≤ 1 또는 x2 ≥ W−1 또는 y2 ≥ H−1 이면 `truncated = True`. 화면 밖으로 나가는 차는 박스 중심 ≠ 물체 중심.
- 검출 0개 → 빈 리스트. 점 shape 이 (N, 3|4) 가 아니면 `ValueError`.

## 4. 표면 편향 (surface bias) — Phase 5 의 가장 큰 오차

LiDAR 는 물체의 **보이는 면** 만 찍는다. 앞차라면 뒷면이다. 중앙값은 그 면 위에 있으므로 GT 중심(박스 기하 중심) 보다
차 길이의 절반 ≈ 2 m 가깝다. 전 프레임 깊이 bias −1.29 m, 대표 프레임 0/50/150 에서 −1.4 ~ −2.1 m.
이건 통계 방법을 바꿔서 없어지지 않는다 (점이 거기 없다). Phase 6 의 표면→중심 오프셋이 이를 다룬다.

## 5. 튜닝 가이드

| 키 | 기본 | 올리면 | 내리면 |
|---|---|---|---|
| `box_shrink` | 0.1 | 배경 감소, 작은 박스가 empty 됨 | 배경 증가 |
| `depth_percentile` | 30 | 배경 오염에 약해짐, 표면 편향 약간 감소 | 전경 앞의 노이즈(기둥·풀)에 약해짐 |
| `depth_band_m` | 1.5 | 큰 차량(버스) 옆면까지 포함 | 점 수 감소 |
| `min_points` | 5 | 먼 물체가 sparse 로 빠짐 (정밀도↑ 재현율↓) | 노이즈 중심 증가 |
| `dedup_distance_m` | 1.0 | 나란히 선 보행자가 합쳐질 수 있음 | 이중 검출이 남음 |

## 6. 함정 (pitfalls)

1. `uv[i]` 는 `points_velo[mask][i]` 다. 원본 인덱스가 필요하면 `idx = np.flatnonzero(mask)` 로 되돌린다. 시각화(`draw_fusion`) 도 이 대응이 필요해서 `mask` 를 받는다.
2. `depth` 는 Velodyne x 가 아니라 정류 카메라 z 다 (약 0.27 m 오프셋). 중심은 원본 Velodyne 점의 중앙값이므로 변환 오차가 없다.
3. 박스가 커도 점이 적을 수 있다 (하늘·먼 곳). `n_points` 를 항상 같이 보라.
4. GT 에 없는 주차 차량이 있다 (frame 50 det3, frame 174 det1). "미매칭 예측" 이 전부 오검출은 아니다.
