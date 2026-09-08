# Phase 8 학습 노트 — 정량 평가 (변형 A / B / C)

담당 모듈: `src/perceptrack3d/evaluation/evaluate.py`, `scripts/evaluate.py`, `tests/test_evaluate.py`, `notebooks/06_multi_object_tracking.ipynb`
결과: `outputs/phase8/results.md`, `results.csv`, `results.json`, `runtime.md`, `plots/*.png`, `config_used.yaml`
재현: `.venv/bin/python scripts/evaluate.py --config configs/kitti.yaml --out outputs/phase8` (먼저 `scripts/run_pipeline.py --variant raw|clustered|tracked`)

## 0. 여섯 가지 질문 (learning-first)

| 질문 | 답 |
|---|---|
| 입력은? | 변형별 프레임 결과 `{frame_id: [Object3D]}` (A, B) 또는 `{frame_id: [Track.as_dict()]}` (C), GT `{frame_id: [GtBox3D]}`, `configs/kitti.yaml` |
| 출력은? | 변형 × 임계값 × 지표 표 (long CSV), 마크다운 표, 플롯 6장, 런타임 보고. 중간 표: `pairs`(매칭 쌍), `preds`, `gts` DataFrame |
| 좌표계는? | Velodyne (x 전방, y 좌, z 상, m). 매칭·오차는 BEV(x, y), 깊이 = x. GT 중심 = 바닥중심 + h/2 |
| 핵심 알고리즘은? | 프레임별 BEV 중심 거리 헝가리안 1:1 매칭 → TP/FP/FN → 정밀도·재현율·오차·IoU 집계, C 는 GT tracklet ↔ track_id 이력으로 IDSW/단절/MT-PT-ML |
| 실패 사례는? | 크기 없는 예측(A) 의 IoU 불가, coasting 예측을 FP 로 셈, 클래스 불일치(COCO↔KITTI), FOV 밖 GT, tracklet 없는 주차 차량 |
| 평가는? | 지표 자체가 산출물. 검증은 합성 3 프레임 손계산 테스트 + 팀 E `comparison.md` 와의 일치 (A/B 수치 완전 일치) |

## 1. 변형 정의

| 변형 | 결과 파일 | 예측으로 쓰는 것 | 크기(size) |
|---|---|---|---|
| A raw | `outputs/phase5/objects_raw.json` | `status == "ok"` 인 Object3D 의 `center` (박스 안 깊이 하위 30 % 밴드의 중앙값) | 없음 |
| B clustered | `outputs/phase6/objects_clustered.json` | `status == "ok"` 인 Object3D (ROI → 지면 제거 → 프러스텀 → DBSCAN → 표면 오프셋) | AABB (yaw = 0) |
| C tracked | `outputs/phase7/tracks.json` | `confirmed` 트랙의 칼만 상태 `center` (coasting 포함) | 마지막 관측 크기 |

## 2. 매칭 규칙

1. **GT 필터** (`select_gt`): 카메라 수평 FOV 81.4° 안 (`gt_in_front_fov`: x > 0, |y| < x·tan 40.7°), x ≤ 70 m (`fusion.roi.x_max`), 클래스 ∈ {Car, Van, Truck, Pedestrian, Cyclist}. 297 프레임에서 **1394 개**.
2. **예측 필터**: A/B 는 `status == "ok"` 이고 center 가 있는 것. C 는 `confirmed == True` (`include_coasting=False` 면 `misses == 0` 만).
3. **프레임별 헝가리안 1:1** (`match_by_center_distance`): 비용 = BEV 거리, 임계값 τ 를 넘는 쌍은 큰 상수로 막고, 결과 중 거리 ≤ τ 인 쌍만 채택. τ = 2 m (기본, `evaluation.match_distance_m`) 와 4 m.
4. TP = 매칭 쌍 수, FP = 미매칭 예측, FN = 미매칭 GT. 클래스는 보지 않는다.

nuScenes 검출 벤치마크도 IoU 대신 BEV 중심 거리 {0.5, 1, 2, 4} m 로 매칭한다. 우리 박스는 크기 추정이 약하므로 중심 거리 매칭이 더 공정하고, 2 m 는 승용차 길이의 절반이다.

## 3. 지표 정의

매칭 쌍 (p, g) 에 대해 p = 예측 중심, g = GT 중심 (Velodyne).

```text
정밀도 P = TP / (TP + FP)          재현율 R = TP / (TP + FN)          F1 = 2PR / (P + R)
MOTA    = 1 − (FN + FP + IDSW) / n_GT                     (CLEAR MOT; A/B 는 IDSW = 0 → '검출 MOTA')
BEV 오차 = ‖p[:2] − g[:2]‖₂     3D 오차 = ‖p − g‖₂          → mean / median / p95
깊이 오차 e = p_x − g_x          |e| 의 mean/median/p95,  bias = mean(e)  (음수 = GT 보다 가깝게 추정)
BEV IoU  = area(box_p ∩ box_g) / area(box_p ∪ box_g)      (Sutherland–Hodgman 클리핑, box_p 는 yaw = 0 AABB)
IoU ≥ 0.5 비율 = 매칭 쌍 중 IoU ≥ 0.5 인 비율             (KITTI Pedestrian/Cyclist 기준; Car 공식 기준 0.7 은 우리 박스로는 거의 0)
```

거리 구간 (GT 의 BEV 거리 √(x²+y²)): 0–20, 20–40, 40–70 m. 구간별 정밀도만 **예측 자신의 거리** 로 분모를 센다.

추적 지표 (C, GT tracklet_id ↔ track_id 이력):

```text
IDSW  : 같은 GT 가 직전에 매칭됐던 트랙과 다른 track_id 에 매칭되면 +1        (count_id_switches)
단절  : GT 의 매칭 프레임 사이에 빈 구간이 생길 때마다 +1                    (track_fragmentation)
cov_single(g) = max_id |{f : g 가 id 에 매칭}| / |{f : g 존재}|             (가장 오래 붙은 단일 ID 커버 비율)
MT = #{g : cov_single ≥ 0.8},  ML = #{g : cov_single ≤ 0.2},  PT = 나머지
```

CLEAR MOT 원문의 MT/ML 은 ID 와 무관하게 "추적된 프레임 비율" 로 정의한다 (`cov_any` 로 함께 보고). 지시서의 근사 정의(단일 ID)가 더 엄격하며 ID 유지력까지 본다.

## 4. 결과 (297 프레임, τ = 2 m) — `outputs/phase8/results.md` 요약

| 변형 | 재현율 | 정밀도 | F1 | MOTA | BEV median | 깊이 bias | IoU mean | IDSW / 단절 |
|---|---|---|---|---|---|---|---|---|
| A raw | 0.45 | 0.53 | 0.48 | 0.05 | 1.48 m | −1.29 m | n/a | – |
| B clustered | 0.66 | 0.84 | 0.74 | 0.53 | 0.56 m | −0.30 m | 0.23 | – |
| C tracked | 0.65 | 0.78 | 0.71 | 0.46 | 0.56 m | −0.33 m | 0.24 | 9 / 11 |
| C, coasting 제외 | 0.61 | 0.91 | 0.73 | 0.54 | 0.55 m | −0.34 m | 0.24 | – |

거리 구간 (재현율 A / B / C): 0–20 m 0.76 / 0.81 / 0.80, 20–40 m 0.35 / 0.86 / 0.87, 40–70 m 0.49 / 0.50 / 0.46.
클래스 재현율 (B): Car 0.74, Van 0.52, Truck 0.62, Cyclist 0.02, Pedestrian n/a (GT 0). C 추적: 53 ID vs GT 36 tracklet, MT/PT/ML = 13/19/4.

### 4.1 왜 B 가 A 보다 좋은가 — 표면 편향 오프셋

A 의 깊이 bias −1.29 m 는 "LiDAR 는 보이는 면만 찍는다" 는 물리에서 온다: 박스 안 점의 중앙값은 물체의 **앞면** 근처이고, GT 중심은 그보다 반 길이(승용차 ~2 m) 뒤에 있다. 오차 CDF(플롯 ①)에서 A 는 1.3~1.8 m 에 뭉쳐 있고, 4 m 로 넓히면 재현율이 0.45 → 0.70 으로 뛴다 — 즉 A 의 '미검출' 절반은 검출 실패가 아니라 **2 m 게이트 밖으로 밀린 체계적 편향** 이다. B 는 클래스별 `half_depth` prior 로 시선 방향으로 중심을 밀어 bias 를 −0.30 m 로 줄였고(팀 E ablation: 오프셋 없이는 A 와 같은 1.39 m), 지면 제거·DBSCAN 은 배경 점 오염을 줄여 p95 를 1.95 → 1.42 m 로 낮춘다. 20–40 m 구간에서 차이가 가장 크다 (0.35 → 0.86).

### 4.2 왜 C 는 B 보다 정밀도가 낮은가 — coasting 예측

C 는 관측이 없는 프레임에도 최대 `max_misses` = 3 프레임 동안 예측 위치를 낸다 (225 개, 그중 GT 2 m 안 57 개). 이들이 예측 수를 1107 → 1156 으로 늘리고 정밀도를 0.84 → 0.78 로 내린다. coasting 을 빼면 정밀도 0.91, MOTA 0.54 로 B 보다 좋다 — 즉 추적은 "관측된 위치를 더 정확하게 골라내는" 효과(FP 인 깜빡이 검출은 확정되지 못해 걸러짐) 와 "없는 곳에 예측을 놓는" 비용을 동시에 가진다. 재현율은 tentative 구간(새 물체마다 2 프레임) 손실과 coasting 이득이 상쇄해 B 와 비슷하다 (0.65 vs 0.66).

### 4.3 IoU 가 낮은 이유

B/C 의 IoU 평균 0.23, ≥ 0.5 비율 0.13. 중심 오차가 0.56 m 인데도 낮은 것은 **크기** 때문이다: 클러스터 AABB 는 보이는 면(폭 1.3 m, 길이 0.4~1 m)만 담아 GT (4.0 × 1.8 m) 의 1/4 넓이다. 0–20 m 에서는 옆면이 보여 0.45, 40 m 밖에서는 0.20. 크기 prior 로 박스를 완성하지 않는 한 IoU 는 중심 품질을 반영하지 못한다 → 중심 오차를 주 지표로 쓴다.

### 4.4 추적 실패 유형 (간트차트 ⑤)

- **ML 4개** (gt4, gt10, gt32, gt34 Cyclist): 관측이 거의 없음. gt34 는 검출기(640 px) 가 자전거를 못 봄. 추적 파라미터로 구제 불가.
- **IDSW 9** = gt3 ×3 (프레임 57–59, 트랙 15→23→24→23), gt7 ×2 (99, 101: 28→36→28), gt11 ×2 (157, 167), gt32 ×1 (181), gt25 ×1 (259). 모두 왼쪽 차선의 원거리 주차 차량: 40 m 밖에서 점 5~15개로 융합이 `sparse`/`no_cluster` 로 깜빡여 트랙이 죽고(4 프레임 연속 미검출) 가까워지면 새 ID 로 태어나거나, 한 차에서 클러스터가 둘로 갈라져 트랙 2개가 번갈아 붙는다. 노트북 06 §6.6 에 gt3 을 프레임 단위로 재현.
- 파라미터 스윕 (노트북 06 §6.7): `max_misses` 5 는 정밀도 −10 %p, `measurement_noise` 는 ±1 %p 이내로 둔감. IDSW 8~14 는 잡음 수준.

## 5. 런타임 (`runtime.md`)

i7-14700K, 단일 프로세스, load average 1.6~2.6 에서 재측정 (팀 E 측정치는 load 7~10 에서 p95 가 부풀어 있었음). median ms/프레임: load 7.3–7.8, projection 1.9–2.6, filters 7.0–7.9, cluster 1.4, tracking 0.25. 파이프라인 total median A 11.3 / B 19.4 / C 21.1 ms, YOLO 13.8 ms(팀 B 인용) 를 더한 end-to-end 25 / 33 / 35 ms → 40 / 30 / 29 FPS (10 Hz 센서 대비 여유 3배). C++ 투영 대체(팀 I)는 프레임당 ~1.5 ms (8 %) 절감 예상.

## 6. 무효이거나 주의가 필요한 비교 (results.md §5 와 동일)

1. A 의 IoU 는 계산 불가 (크기 없음).
2. B/C IoU 는 AABB vs 회전 GT, 크기 과소 → 중심 오차와 함께 읽는다.
3. C 초반·새 물체 등장 시 tentative 2 프레임은 구조적 미검출.
4. C 정밀도는 coasting 예측 포함 (참고 행에 제외 결과).
5. FOV 밖·70 m 밖 GT 는 제외, tracklet 없는 주차 차량 검출은 FP 로 셈 (정밀도 하한).
6. Cyclist ≈ 0 은 검출기 한계, Pedestrian GT 0 → n/a.
7. τ 는 오차의 상한 — 2 m 표의 오차 통계는 잘린 분포.
8. 클래스 무시 매칭.
9. 런타임은 공용 PC — load average 와 함께, median 만 대표값.

## 7. 다음 개선 후보 3개

1. **박스 완성 (size prior / L-shape fitting)**: 클래스별 GT 평균 크기로 AABB 를 시선 방향 뒤로 확장하거나 BEV L-shape 로 yaw·길이를 추정. 목표 IoU ≥ 0.5 비율 0.13 → 0.5 이상. 평가 그대로 재사용.
2. **검출 해상도 1280** (팀 B 측정: 매칭률 +12 %p, Cyclist 10배, 추론 3배): 40–70 m 재현율 0.50 과 Cyclist 0.02 의 병목은 융합이 아니라 검출. 변형 D 로 추가해 같은 표에 비교.
3. **원거리 깜빡임 대응**: 거리 의존 R(먼 물체일수록 큰 관측 잡음)과 2D 박스 IoU 를 비용에 추가, 또는 tentative 트랙도 관측 없이 1~2 프레임 유지. 목표 IDSW 9 → 5 이하, ML 4 → 2 (Cyclist 제외).

## 8. 자가 점검 질문

1. τ 를 2 m 에서 4 m 로 올리면 A 의 재현율이 0.45 → 0.70 으로 뛰는데 B 는 0.66 → 0.69 로 거의 안 변한다. 이 차이가 말해 주는 A 의 오차 성질은 무엇인가?
2. C 의 coasting 예측 225 개 중 57 개만 GT 2 m 안에 든다. 나머지 168 개는 무엇인가 (두 가지 이상)? 이 비율을 줄이려면 어떤 파라미터를 어느 방향으로 바꾸고 무엇을 잃는가?
3. MT/ML 을 단일 ID 기준으로 정의하면 어떤 트랙이 CLEAR MOT 원문 정의보다 불리하게 평가되는가? gt7 (28 → 36 → 28) 을 예로 설명하라.
