# Phase 3 학습 노트 — LiDAR 점을 이미지에 투영

노트북: `notebooks/03_lidar_camera_projection.ipynb` · 모듈: `geometry/projection.py`, `visualization/overlay.py` · 테스트: `tests/test_projection.py`

## 함수 계약

```python
uv, depth, mask = project_velo_to_image(points_velo, calib, image.shape[:2])   # image_shape = (H, W)
# uv (M, 2) float32 픽셀, depth (M,) float32 m (> 0), mask (N,) bool — points_velo[mask] ↔ uv 1:1
```

## 단계 (각 단계에서 점이 줄어든다, 프레임 0 기준)

| 단계 | 연산 | 남는 점 |
|---|---|---|
| 0 | 원본 LiDAR | 123,839 (100 %) |
| 1 | 동차화 `(N, 3) → (N, 4)` | — |
| 2 | `proj = pts_h @ P_velo_to_imgᵀ` → `[u·d, v·d, d]` | — |
| 3 | 카메라 앞 `d > 0` | ≈ 50 % |
| 4 | `(u, v) = (u·d / d, v·d / d)` | — |
| 5 | 경계 `0 ≤ u < W`, `0 ≤ v < H` | 19,192 (15.5 %) |

- 이미지 안에 들어오는 점이 15 % 뿐인 이유: 카메라 화각(수평 약 90°)이 LiDAR 360° 의 일부이고, 뒤쪽 절반은 `d ≤ 0`.
- `depth` 는 투영 동차 좌표의 세 번째 성분 = rect z + `P_rect_02[2, 3]` (≈ 2.7 mm) — 카메라 2 광학축 방향 거리.

## 정렬 확인 (육안)

- 도로면이 아래(노랑, ≈4 m)에서 위(보라, 70 m)로 매끄럽게 멀어짐. 차량 안쪽 점은 같은 색으로 뭉치고 윤곽에서 색이 뛴다. 하늘에는 점이 없다.
- 이미지 위쪽 약 125 px 는 비어 있음: Velodyne 상향 시야각(+2°) 한계.

## 실패 사례

1. **뒤쪽 점 미필터**: `d < 0` 인 점을 나누면 부호가 뒤집혀 이미지 안에 가짜 픽셀이 찍힌다. 반드시 **나누기 전에** `d > 0`.
2. **초근접점**: d 가 수 cm 이면 u, v 가 폭주. 경계 필터가 걸러 주지만 차체 반사점이 화면 아래에 찍힐 수 있다.
3. **경계**: `u < W` (0-based). `<=` 를 쓰면 `img[v, u]` 인덱스가 넘친다.
4. **센서 시차(parallax)**: LiDAR 가 카메라보다 0.27 m 뒤·8 cm 위에 있어, 물체 가장자리에서 카메라에는 가려진 배경 점이 물체 위에 겹쳐 투영된다 → Phase 5 전경/배경 오염의 근원.

## 색상 규약

`plasma_r`, 0~70 m 고정 범위 (가까움 = 밝은 노랑). 프레임이 달라도 같은 색 = 같은 거리. BGR 로 반환하므로 `cv2` 로 바로 그린다.

## 평가

- 테스트: 뒤쪽 점 mask False, uv 경계, depth > 0, 광축 위 점 (0, 0, 10)_rect → `P_rect_02 @ [0,0,10,1]` 과 일치, 합성 격자의 경계 판정이 수작업과 일치, `mask.sum() == len(uv)`.
- 증거: `outputs/phase3/frame{0,100,200}_projection_overlay.png`.
