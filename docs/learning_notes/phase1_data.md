# Phase 1 학습 노트 — KITTI 데이터 로드와 시각화

노트북: `notebooks/01_kitti_data_check.ipynb` · 모듈: `src/perceptrack3d/data/kitti_loader.py`, `visualization/pointcloud.py` · 테스트: `tests/test_kitti_loader.py`

## 개념

| 데이터 | 파일 | 읽는 법 | 결과 |
|---|---|---|---|
| RGB 이미지 | `image_02/data/0000000000.png` | `cv2.imread` | `(375, 1242, 3) uint8`, **BGR** |
| LiDAR | `velodyne_points/data/0000000000.bin` | `np.fromfile(dtype=np.float32).reshape(-1, 4)` | `(N≈124k, 4) float32` = `[x, y, z, reflectance]` |

- **BGR**: OpenCV 는 채널을 B, G, R 순으로 둔다. matplotlib·PyTorch·YOLO 는 RGB 를 기대하므로 `cv2.cvtColor(img, cv2.COLOR_BGR2RGB)` 가 필요하다. `cv2.imwrite` 는 다시 BGR 을 기대한다.
- **shape 순서**: numpy 이미지는 `(H, W, C)` 이고 `img[v, u]` 로 접근한다 (행 = 세로 v 가 먼저).
- **.bin 구조**: 헤더 없이 float32 가 `x y z r x y z r …` 로 이어진다. 파일 크기는 16 바이트의 배수. `reshape(-1, 4)` 의 `-1` 은 "나머지 축은 자동 계산".
- **Velodyne 좌표계**: x 전방, y 좌, z 상 (오른손 좌표계), 단위 m. 원점은 센서 중심(지상 약 1.73 m) → 지면 점의 z ≈ −1.7 m.
- **reflectance**: 레이저 반사 강도, 0~1. 재질·입사각·거리에 따라 달라진다 (이 프로젝트에서는 아직 쓰지 않음).
- **동기화**: 같은 번호의 png 와 bin 은 같은 시각. LiDAR 의 `timestamps.txt` 는 스캐너가 정면을 향해 카메라를 트리거한 순간이며, 카메라 타임스탬프와 약 10 ms 안에서 일치한다. 프레임 간격 ≈ 100 ms (10 Hz).

## 시각화

- **BEV**(bird's-eye view): x-y 평면 산점도, 색 = 높이 z. x(전방)를 위로, y(좌)를 왼쪽으로 두기 위해 가로축을 −y 로 그린다.
- **3D scatter**: 12만 점을 전부 그리면 느리므로 seed 고정 부표본(60k)만 그린다 (재현성).
- **Open3D**: `PointCloud` 변환 → PLY 저장(외부 뷰어용) → `OffscreenRenderer`(EGL headless) 로 이미지 저장. WSL 에서 창을 띄우지 않는다.

## 함정

1. `np.fromfile` 은 끝의 **자투리 바이트를 조용히 버린다** → 크기 검사는 `path.stat().st_size % 16` 으로 읽기 전에 해야 한다 (테스트로 잡아낸 실제 버그).
2. BGR 배열을 `plt.imshow` 에 넣으면 하늘이 주황색이 된다.
3. `S_rect_02` 는 (width, height) 이고 numpy shape 은 (height, width) — 순서가 반대.
4. 이미지 폴더와 LiDAR 폴더의 파일 번호 집합이 다르면 동기화가 깨진 것이다 (여기서는 0~296, 297 개 일치).

## 평가

- 테스트: shape/dtype, 없는 파일 → `FileNotFoundError`, 깨진 PNG / 16 배수가 아닌 .bin / 빈 .bin → `ValueError`, png·bin id 집합 일치.
- 증거 이미지: `outputs/phase1/frame0_rgb.png`, `frame0_pointcloud_bev.png`, `frame0_pointcloud_3d.png`, `frame0_pointcloud_o3d.png`, `frame0.ply`.
