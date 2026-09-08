# PercepTrack3D 아키텍처 및 모듈 계약

모든 팀(에이전트)과 학습 노트북은 이 문서의 계약을 따른다. 바꾸려면 `docs/decisions.md` 에 기록한다.

## 1. 좌표계 (Coordinate frames)

| 이름 | 축 | 단위 | 어디서 쓰나 |
|---|---|---|---|
| Velodyne (`velo`) | x 전방, y 좌, z 상 | m | LiDAR 원본, **3D 결과·추적·평가의 기준 프레임**, tracklet GT |
| Camera 0 (`cam0`) | x 우, y 아래, z 전방 | m | `calib_velo_to_cam.txt` 의 R, T 로 velo→cam0 |
| Rectified cam (`rect`) | cam0 를 `R_rect_00` 로 회전 | m | 스테레오 정렬 후 프레임 |
| Image 02 픽셀 (`img`) | u 우, v 아래, 원점 좌상단 | px | `P_rect_02` (3×4) 로 rect→img |

투영 체인: `p_velo(4×1, 동차) → T_velo_to_cam(4×4) → R_rect_00(4×4) → P_rect_02(3×4) → [u·z, v·z, z] → (u, v) = (·/z)`

3D 결과를 Velodyne 프레임으로 통일하는 이유: (1) tracklet GT 가 Velodyne 프레임, (2) BEV(x-y 평면)가 직관적, (3) ROS REP-103 (x 전방, z 상) 과 일치.

## 2. 파일 소유권 (팀 분담)

| 모듈 | 파일 | 담당 |
|---|---|---|
| 설정 | `src/perceptrack3d/config.py`, `configs/kitti.yaml` | 공통 (완료) |
| 타입 | `src/perceptrack3d/types.py` | 공통 (완료) — 필드 추가는 허용, 삭제/이름 변경 금지 |
| 로더 | `data/kitti_loader.py` | 팀 A (Phase 1) |
| 캘리브레이션 | `geometry/calibration.py`, `geometry/transforms.py` | 팀 A (Phase 2) |
| 투영 | `geometry/projection.py` | 팀 A (Phase 3) |
| 2D 검출 | `detection/detector.py` | 팀 B (Phase 4) |
| 융합 | `fusion/frustum_fusion.py`, `fusion/point_filters.py` | 팀 E (Phase 5, 6) |
| 추적 | `tracking/kalman_tracker.py`, `tracking/association.py` | 팀 C (Phase 7) |
| GT 파서 | `evaluation/tracklets.py` | 팀 D (Phase 8 전반) |
| 평가/메트릭 | `evaluation/metrics.py`, `scripts/evaluate.py` | 팀 G (Phase 8 후반) |
| 파이프라인 실행 | `scripts/run_pipeline.py` | 팀 E/G |
| 시각화 | `visualization/*.py` | 각 팀이 필요한 함수 추가 |
| ROS2 | `ros2_ws/src/perceptrack3d_ros/` | 팀 H (Phase 9) |
| C++ | `cpp/` | 팀 I (Phase 10) |

## 3. 함수 계약 (시그니처는 고정, 내부는 자유)

### data/kitti_loader.py
```python
def load_image(path) -> np.ndarray            # (H, W, 3) uint8 BGR. 없거나 깨지면 FileNotFoundError / ValueError
def load_velodyne(path) -> np.ndarray         # (N, 4) float32 [x, y, z, reflectance]. 크기가 16의 배수 아니면 ValueError
class KittiDataset:
    def __init__(self, cfg: dict)             # load_config() 결과
    def __len__(self) -> int                  # 297
    def frame_ids(self) -> list[int]
    def image_path(self, frame_id: int) -> Path
    def velodyne_path(self, frame_id: int) -> Path
    def load_frame(self, frame_id: int) -> dict   # {"frame_id", "image", "points", "image_path", "velodyne_path"}
```

### geometry/calibration.py
```python
@dataclass
class KittiCalibration:
    T_velo_to_cam: np.ndarray   # (4, 4)
    R_rect_00: np.ndarray       # (4, 4)  (3×3 을 4×4 로 확장)
    P_rect_02: np.ndarray       # (3, 4)
    @classmethod
    def from_dir(cls, calib_dir) -> "KittiCalibration"
    def velo_to_rect(self, pts_velo: np.ndarray) -> np.ndarray     # (N,3)->(N,3)
    def rect_to_velo(self, pts_rect: np.ndarray) -> np.ndarray     # 역변환 (round-trip 테스트용)
    @property
    def P_velo_to_img(self) -> np.ndarray                          # (3,4) = P_rect_02 @ R_rect_00 @ T_velo_to_cam
```

### geometry/projection.py
```python
def project_velo_to_image(pts_velo: np.ndarray, calib, image_shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray, np.ndarray]
    # 반환: uv (M,2) float32, depth (M,) float32 (rect 프레임 z), mask (N,) bool — 카메라 앞 & 이미지 안에 있는 원본 점 인덱스
```

### detection/detector.py
```python
class YoloDetector:
    def __init__(self, cfg: dict)                     # cfg["detection"] 사용
    def detect(self, image_bgr: np.ndarray, frame_id: int) -> list[Detection2D]
    def detect_timed(self, image_bgr, frame_id) -> tuple[list[Detection2D], float]   # (검출, 추론 ms)
def save_detections_json(dets: list[Detection2D], path) / load_detections_json(path) -> dict[int, list[Detection2D]]
```

### fusion/frustum_fusion.py
```python
def fuse_frame_raw(points_velo, detections, calib, image_shape, cfg) -> list[Object3D]        # Phase 5 기준선 (method="raw")
def fuse_frame_clustered(points_velo, detections, calib, image_shape, cfg) -> list[Object3D]  # Phase 6 (method="clustered", size/yaw 포함)
```
### fusion/point_filters.py
```python
def roi_filter(points_velo, roi_cfg) -> np.ndarray(bool)
def remove_ground(points_velo, ground_cfg) -> tuple[np.ndarray(bool), np.ndarray(4,)]  # (비지면 mask, 평면 계수)
def frustum_mask(points_velo, xyxy, calib, image_shape, shrink) -> np.ndarray(bool)
def cluster_points(points_xyz, cluster_cfg) -> np.ndarray(int)       # 라벨, -1 = noise
def aabb_from_points(points_xyz) -> tuple[center(3,), size(3,)]
def obb_from_points_bev(points_xyz) -> tuple[center(3,), size(3,), yaw]  # BEV PCA 기반
```

### tracking/kalman_tracker.py
```python
class KalmanTracker:
    def __init__(self, cfg: dict)                                     # cfg["tracking"]
    def step(self, objects: list[Object3D], frame_id: int) -> list[Track]   # predict → associate → update → lifecycle. 확정 트랙만 반환
    @property
    def all_tracks(self) -> list[Track]
```
### tracking/association.py
```python
def build_cost_matrix(tracks, objects, gating: str, ...) -> np.ndarray     # (T, D), 게이트 밖은 np.inf
def hungarian_assign(cost, gate) -> tuple[list[tuple[int,int]], list[int], list[int]]  # (matches, unmatched_tracks, unmatched_objects)
```

### evaluation/tracklets.py
```python
def load_tracklets(xml_path) -> list[dict]                     # 원본 tracklet (objectType, h, w, l, first_frame, poses[...])
def gt_boxes_by_frame(xml_path) -> dict[int, list[GtBox3D]]    # 프레임별 GT (중심 = 바닥중심 + h/2)
```

### evaluation/metrics.py
```python
def match_by_center_distance(preds: list[Object3D|Track], gts: list[GtBox3D], max_dist: float) -> list[tuple[int,int,float]]
def center_error_stats(...) / depth_error_stats(...) / bev_iou(box_a, box_b) / precision_recall(...) / id_switches(...)
```

## 4. 출력 파일 규약 (`outputs/`)
```
outputs/
├── phase1/  frame0_rgb.png, frame0_pointcloud_bev.png, frame0_pointcloud_3d.png
├── phase3/  frame0_projection_overlay.png
├── phase4/  detections.json (전 프레임), examples/*.png, failures/*.png, runtime.md
├── phase5/  objects_raw.json, frame*_fusion.png
├── phase6/  objects_clustered.json, frame*_bev.png, comparison.md
├── phase7/  tracks.json, frame*_tracks.png, bev_trajectories.png
├── phase8/  results.csv, results.md, plots/*.png, runtime.md
└── phase10/ benchmark.md
```
JSON 은 `{"frame_id": [...as_dict()...]}` 형태의 프레임별 리스트.

## 5. 공통 규칙
- 독스트링·주석은 한국어. 식별자는 영어. 입력/출력 shape·dtype·좌표계·단위를 독스트링에 명시.
- 절대경로 금지: `load_config()` 로만 경로를 얻는다.
- 모듈 경계에서 shape 검증 (`assert pts.ndim == 2 and pts.shape[1] in (3, 4)` 수준).
- 결정적 로직(파서, 변환, 투영, 할당)은 `tests/test_*.py` 로 검증. 실행: `.venv/bin/python -m pytest -q`.
- 시각화는 `matplotlib` 로 파일 저장 (headless). Open3D 창 띄우기는 노트북 셀에서 선택적으로.
- 성능 수치는 항상 하드웨어·시퀀스·프레임 수·측정 방법과 함께 기록.
