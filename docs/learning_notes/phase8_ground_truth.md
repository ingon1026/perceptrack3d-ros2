# Phase 8 학습 노트 — 정답(GT) tracklet 과 평가 메트릭

담당 모듈: `src/perceptrack3d/evaluation/tracklets.py`, `evaluation/metrics.py`, `visualization/gt.py`, `scripts/check_tracklets.py`

## 1. tracklet_labels.xml 구조

KITTI raw 의 3D 라벨은 C++ boost serialization 이 만든 XML 이다 (`devkit/cpp/tracklets.h` 가 원본 정의).

```text
<boost_serialization>
  <tracklets>
    <count>36</count>               ← tracklet 수 (실제 <item> 수와 같아야 함, 아니면 ValueError)
    <item>                          ← tracklet 1개 = 한 물체가 여러 프레임에 걸쳐 존재
      <objectType>Car</objectType>  ← Car, Van, Truck, Pedestrian, Person (sitting), Cyclist, Tram, Misc
      <h> <w> <l>                   ← 박스 크기 (m). tracklet 전체에서 고정
      <first_frame>0</first_frame>  ← 처음 등장하는 프레임 번호
      <poses>
        <count>15</count>
        <item>                      ← pose 1개 = 한 프레임의 위치/자세. poses[i] 는 프레임 first_frame + i
          <tx> <ty> <tz>            ← 박스 **바닥면 중심**, Velodyne 좌표 (m)
          <rx> <ry> <rz>            ← 회전 (rad). rx = ry = 0, rz 만 사용 (yaw)
          <state>                   ← 0 UNSET, 1 INTERP(보간), 2 LABELED(사람이 직접 라벨)
          <occlusion>               ← -1 UNSET, 0 VISIBLE, 1 PARTLY, 2 FULLY
          <truncation>              ← -1 UNSET, 0 IN_IMAGE, 1 TRUNCATED, 2 OUT_IMAGE, 99 BEHIND_IMAGE
          <amt_*>                   ← Mechanical Turk 원본 라벨 (사용 안 함)
        </item> ...
      </poses>
      <finished>1</finished>
    </item> ...
  </tracklets>
</boost_serialization>
```

- **tracklet_id** = XML 에 나오는 순서 (0부터). 파일에 명시적 id 는 없다. 같은 물체는 하나의 tracklet 이므로 이 인덱스가 곧 GT 트랙 id 이다.
- 이 드라이브(0015): tracklet 36개 = Car 33, Van 1, Truck 1, Cyclist 1. 프레임 297개, 프레임당 GT 1~11개(평균 4.97).
- 파싱은 표준 라이브러리 `xml.etree.ElementTree` 만 쓴다: `root.find("tracklets").findall("item")` 은 **직계 자식**만 찾으므로 pose 의 `<item>` 과 섞이지 않는다.

## 2. 좌표 규약 (공식 devkit 으로 확인)

| 항목 | 규약 | 근거 |
|---|---|---|
| 기준 프레임 | Velodyne (x 전방, y 좌, z 상, m) | devkit `readme.txt`: "All tracklets are represented in Velodyne coordinates." |
| (tx, ty, tz) | 박스 **바닥면** 중심 | `run_demoTracklets.m`: `corners.z = [0,0,0,0,h,h,h,h]` 에 `t` 를 더함 → tz 는 바닥 높이 |
| (h, w, l) | l = 로컬 x(진행 방향), w = 로컬 y, h = z | 같은 스크립트: `corners.x = ±l/2` (front/back), `corners.y = ±w/2` (left/right) |
| rz | z 축 yaw, R = [[cos, −sin],[sin, cos]] (반시계 양수) | 같은 스크립트의 `R = [cos(rz) -sin(rz) 0; sin(rz) cos(rz) 0; 0 0 1]` |
| 프레임 번호 | `first_frame + pose_idx` | `tracklets.h` `getPose()`: `pose_idx = frame_number - first_frame` |

주의: `run_demoTracklets.m` 의 주석에는 "LOCAL OBJECT COORDINATE SYSTEM: x -> facing right, y -> facing forward" 라고 적혀 있지만 **실제 코드**는 x = ±l/2, y = ±w/2 로 Velodyne 과 같은 방향(x 전방)을 쓴다. 코드를 따랐다.

### 수치 교차검증 (`scripts/check_tracklets.py` 와 동일한 방법)
프레임 0..290 (10 간격) 의 GT 박스 151개 안에 들어가는 LiDAR 점 수 합계:

| 가설 | 점 수 |
|---|---|
| A. 바닥중심 + h/2, [l, w, h], yaw = rz (**채택**) | 30591 |
| B. tz 가 이미 기하 중심 | 23270 |
| C. tz 가 천장 | 3498 |
| D. l 과 w 를 바꿈 | 9864 |
| E. yaw = −rz | 30526 |

A 가 최대. E 가 A 와 비슷한 이유는 이 드라이브의 yaw 가 대부분 0 또는 ±π 근처라 부호가 구분되지 않기 때문이며, 부호는 devkit 코드(반시계 양수)를 따른다.

## 3. 바닥 중심 → 기하 중심 변환

융합 결과 `Object3D.center` 는 점들의 기하 중심이므로 GT 도 기하 중심으로 맞춘다.

```text
center = (tx, ty, tz + h/2)        size = (l, w, h)        yaw = wrap_to_pi(rz)
```

BEV 4 모서리 (`box_corners_bev`):

```text
corner_k = Rz(yaw) · [±l/2, ±w/2]ᵀ + (x, y)      순서: 앞좌, 앞우, 뒤우, 뒤좌
```

박스 안 점 판정 (`points_in_box_mask`): 점을 박스 로컬 프레임으로 되돌려 절댓값 비교.

```text
d = Rz(−yaw) · (p − center)      |d_x| ≤ l/2,  |d_y| ≤ w/2,  |d_z| ≤ h/2
```

## 4. 메트릭 정의 (`evaluation/metrics.py`)

모든 함수는 순수 함수이며 numpy 배열 또는 `Object3D`/`Track`/`GtBox3D` 리스트를 받는다.

| 함수 | 정의 |
|---|---|
| `match_by_center_distance(P, G, max_dist)` | 비용 `c_ij = ‖p_i[:2] − g_j[:2]‖₂` (BEV). 헝가리안(`scipy.optimize.linear_sum_assignment`)으로 총 비용 최소 1:1 할당 후 `c_ij > max_dist` 인 쌍 제거. 게이트 밖 쌍의 비용은 큰 유한값(1e6) — `inf` 는 행 전체가 `inf` 일 때 실패하기 때문. |
| `center_errors` | 매칭 쌍의 `‖p − g‖₂` (3D) 와 `‖p[:2] − g[:2]‖₂` (BEV) 의 mean / median / p95 |
| `depth_errors` | `e = p_x − g_x` (Velodyne x = 전방 깊이). `mean_abs, median_abs, p95_abs` 와 부호 있는 `bias = mean(e)` (양수 = 예측이 더 멀다) |
| `bev_iou(a, b)` | 회전 사각형 IoU = 교집합 / (A + B − 교집합). 교집합 넓이는 Sutherland–Hodgman 클리핑(`convex_intersection`) + 신발끈 공식(`polygon_area`). shapely 를 쓰지 않은 이유: 의존성 없이 재현 가능하고, 사각형은 볼록이므로 정확하며, 알고리즘이 코드로 드러나 학습에 좋다. |
| `precision_recall(TP, n_pred, n_gt)` | P = TP/n_pred, R = TP/n_gt, F1 = 2PR/(P+R). 분모 0 → 0 |
| `count_id_switches(assignments)` | 프레임별 `(track_id, gt_id)` 매칭 목록에서, 같은 GT 가 **직전 매칭 트랙**과 다른 트랙에 붙으면 +1 (CLEAR MOT IDSW) |
| `track_fragmentation(assignments)` | GT 의 매칭 프레임 열에 공백이 생길 때마다 +1 (예: [0,1,2,5,6] → 1) |
| `summarize_runtime({stage: [ms, ...]})` | 단계별 mean / median / p95 ms, fps = 1000 / mean_ms. **입력 단위는 ms** |

## 5. 평가할 때 주의할 점

1. **GT 는 카메라 시야 밖 물체도 포함한다.** Velodyne 은 360° 를 보지만 우리 파이프라인은 image_02 검출에서 시작하므로, 카메라 뒤·옆의 GT 를 놓친 것을 recall 감점으로 세면 불공정하다. 지금은 `gt_in_front_fov(gt_boxes, fov_deg=81.4)` 로 Velodyne 프레임 근사 필터를 쓴다.
   - 근거: `P_rect_02` 의 fx = 721.54 px, 이미지 폭 1242 px → 수평 FOV = 2·atan(621 / 721.54) ≈ 81.4°. 조건 `x > 0 and |y| < x·tan(40.7°)`.
   - 한계: Velodyne 원점과 카메라 원점 사이 약 0.27 m 오프셋과 박스 크기를 무시한다. 캘리브레이션 팀의 `project_velo_to_image` 가 준비되면 박스 모서리 투영으로 정확히 걸러야 한다.
2. **보간 pose**: `state == 1 (INTERP)` 은 사람이 직접 찍은 라벨이 아니라 키프레임 사이를 보간한 값이다. 지금은 모두 포함하되, 필요하면 `load_tracklets` 결과의 `state` 로 거른다.
3. **occlusion / truncation**: 먼 곳(45 m 이상)이나 가려진 물체는 LiDAR 점이 수 개뿐이라 (예: 프레임 150 의 id 15 는 2점) 융합이 실패하는 것이 정상이다. 결과 표에는 "가시(occlusion 0) & 이미지 안(truncation 0)" 부분집합을 따로 보고하는 것이 정직하다.
4. **클래스 이름 차이**: GT 는 Car/Van/Truck/Pedestrian/Cyclist, 검출은 COCO(car/truck/bus/person/bicycle/motorcycle). 중심 거리 매칭은 클래스 무관이므로, 클래스별 평가는 매핑 표를 명시하고 해야 한다.
5. **tracklet_id 는 GT 트랙 id** 이므로 ID 스위치 계산 시 `(track.track_id, gt.tracklet_id)` 쌍을 그대로 쓰면 된다.

## 6. 사용 예시

```python
from perceptrack3d.config import load_config
from perceptrack3d.evaluation.tracklets import gt_boxes_by_frame, gt_in_front_fov
from perceptrack3d.evaluation.metrics import match_by_center_distance, center_errors, depth_errors, bev_iou, precision_recall

cfg = load_config()
gts = gt_boxes_by_frame(cfg["dataset"]["tracklets"])                     # {frame_id: [GtBox3D]}
gt_f = gt_in_front_fov(gts[50])                                           # 카메라 시야 안 GT 만
preds = [o for o in objects_frame50 if o.status == "ok"]                  # center 가 있는 예측만
matches = match_by_center_distance(preds, gt_f, cfg["evaluation"]["match_distance_m"])
print(center_errors(matches, preds, gt_f), depth_errors(matches, preds, gt_f))
print(precision_recall(len(matches), len(preds), len(gt_f)))
ious = [bev_iou(preds[p], gt_f[g]) for p, g, _ in matches if preds[p].size is not None]
```

## 7. 자가 점검 질문

1. XML 의 tz 를 그대로 `Object3D.center.z` 와 비교하면 어떤 편향이 생기는가? (답: 예측이 GT 보다 h/2 만큼 높게 나온다)
2. `bev_iou` 에서 두 박스 모서리 순서(시계/반시계)가 달라도 결과가 같은 이유는?
3. 카메라 시야 밖 GT 를 recall 계산에 포함하면 어느 변수가 어떻게 왜곡되는가?
