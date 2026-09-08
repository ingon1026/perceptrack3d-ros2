# Phase 2 학습 노트 — 캘리브레이션과 좌표계

노트북: `notebooks/02_calibration_and_frames.ipynb` · 모듈: `geometry/calibration.py`, `geometry/transforms.py` · 테스트: `tests/test_calibration.py`

## 네 개의 좌표계

| 이름 | 원점 | 축 | 단위 |
|---|---|---|---|
| Velodyne `velo` | LiDAR 중심 | x 전방, y 좌, z 상 | m |
| Camera 0 `cam0` | 왼쪽 흑백 카메라 광학 중심 | x 우, y 아래, z 전방 | m |
| Rectified `rect` | cam0 와 같음 | cam0 를 `R_rect_00` 으로 ≈1° 회전 | m |
| Image 02 `img` | 이미지 좌상단 | u 우, v 아래 | px |

대략 `x_velo → z_cam`, `y_velo → −x_cam`, `z_velo → −y_cam`. Velodyne 원점은 카메라보다 **0.27 m 뒤, 8 cm 위** (`T` 값에서 읽힘).

## 파일 → 행렬

- `calib_velo_to_cam.txt`: `R (3×3)`, `T (3,)` → `p_cam0 = R p_velo + T`. 4×4 동차 행렬 `T_velo_to_cam = [[R, T], [0, 1]]` 로 보관.
- `calib_cam_to_cam.txt`: `R_rect_00 (3×3)` → 4×4 로 확장 (`[3,3] = 1`). `P_rect_02 (3×4) = [[f, 0, cx, tx], [0, f, cy, ty], [0, 0, 1, tz]]`. `S_rect_02 = (1242, 375)` = (W, H).
  - f = 721.5 px, (cx, cy) = (609.6, 172.9). `tx = f·b` 이며 b = +0.062 m 는 카메라 2 가 cam0 에서 옆으로 떨어진 기준선(baseline).
  - image_02 에 투영해도 **`R_rect_00`** 을 쓴다 (`R_rect_02` 아님). `K_02`, `D_02` 는 비정류 원본용이라 쓰지 않는다.

## 핵심 수식

- 동차 좌표: `[p'; 1] = T @ [p; 1]` — 회전과 이동을 한 번의 곱으로.
- 역변환: `T⁻¹ = [[Rᵀ, −Rᵀ t], [0, 1]]` (R 이 직교이므로 `R⁻¹ = Rᵀ`). `np.linalg.inv` 불필요.
- 투영 체인 (KITTI devkit): `y = P_rect_02 @ R_rect_00 @ T_velo_to_cam @ [x, y, z, 1]ᵀ`, `(u, v) = (y0/y2, y1/y2)`. **오른쪽부터** 적용. 세 행렬을 미리 곱한 것이 `P_velo_to_img (3×4)`.

## 함정

1. **텍스트 반올림**: 파일의 회전은 유효숫자 7자리라 `R Rᵀ − I ≈ 8e-8`. 그대로 `Rᵀ` 역변환을 쓰면 70 m 점의 round-trip 오차가 1.8e-6 m 로 허용치(1e-6)를 넘는다 → `nearest_rotation()` (SVD `U Vᵀ`, orthogonal Procrustes) 로 보정. 값 변화 1e-7, 투영 영향 < 0.001 px.
2. 곱 순서를 바꾸면 점이 엉뚱한 곳에 찍힌다. "먼저 velo→cam0, 그다음 정류, 마지막 투영".
3. `S_rect_02` 는 (W, H). `KittiCalibration.image_shape` 가 (H, W) 를 돌려준다.

## 평가

- 직교성 (`R Rᵀ = I`, det = 1), round-trip `rect_to_velo(velo_to_rect(p)) ≈ p` (1e-6), `invert_rigid(T) @ T ≈ I`.
- 부호 확인: velo (10, 0, 0) → rect z ≈ 9.73; velo +y → rect −x; velo +z → rect −y; velo 원점 → rect z ≈ −0.27.
