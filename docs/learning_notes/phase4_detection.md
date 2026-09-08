# Phase 4 학습 노트 — 사전학습 YOLO 로 2D 검출

## 한 줄 요약
COCO 로 학습된 yolov8n 을 그대로 써서 image_02 프레임마다 도로 객체의 2D 박스(픽셀 xyxy)·신뢰도·클래스를 얻는다. 이 박스가 Phase 5 에서 LiDAR 점을 고르는 창이 된다.

## 6가지 질문
1. **입력**: `(375, 1242, 3) uint8 BGR` (cv2.imread 그대로. ultralytics 문서가 ndarray 입력을 HWC/BGR/uint8 로 정의).
2. **출력**: `list[Detection2D]`. `xyxy float32 (4,)` = (x1, y1, x2, y2) 픽셀, `confidence ∈ [0,1]`, `class_name`(COCO), `class_id`. 신뢰도 내림차순.
3. **좌표계**: image_02 픽셀 프레임. 원점 좌상단, x 오른쪽, y 아래. 3D 와 비교하려면 Phase 3 투영으로 LiDAR 점을 이 프레임에 가져와야 한다.
4. **알고리즘**: (1) letterbox 로 긴 변을 imgsz(640)에 맞춰 축소 → 1242×375 가 640×192 로. (2) 신경망 순전파로 후보 박스·클래스 점수. (3) conf ≥ 0.25 필터, keep_classes 필터. (4) NMS: IoU ≥ 0.45 로 겹치는 같은 클래스 박스 중 최고 신뢰도만 남김. (5) 박스를 원본 픽셀로 되돌림.
5. **실패 사례** (`outputs/phase4/failures/`): 원거리 소형 객체(축소로 10 px 대), 주차 차량 열의 가림, Cyclist 미검출(COCO 에 없는 클래스), 클래스별 NMS 로 인한 truck+bus 이중 검출, 공사 표지판 person 오검출, 이미지 경계에서 잘린 객체(박스 중심 치우침).
6. **평가**: 지금은 GT 투영 박스와 IoU 0.4 매칭 proxy 로 74.7% (imgsz 640). 정식 평가는 Phase 8. 런타임은 `outputs/phase4/runtime.md` (i7-14700K, CPU, 297 프레임, 중앙값 13.8 ms).

## 기억할 것
- 사전학습 모델을 쓰는 이유: 검출은 이 프로젝트의 목표가 아니라 도구. 기준선을 빨리 만들고 정량 평가로 한계를 재는 것이 우선.
- warm-up: 첫 추론은 커널 초기화로 느리다. 시간을 재기 전에 더미 입력으로 미리 돌린다.
- 결정성: 추론에는 난수가 없다. 같은 이미지 → 같은 박스. 테스트로 확인.
- COCO ↔ KITTI 라벨은 1:1 이 아니다 (car ↔ Car+Van, person+bicycle ↔ Cyclist). 표는 `docs/decisions.md` 팀 B 절.
- imgsz 640→1280 은 재현율 +12%p 이지만 시간 3배. 설정 한 줄로 바꿀 수 있으니 Phase 8 에서 비교.

## 명령
```bash
.venv/bin/python scripts/run_detection.py          # 297 프레임 → outputs/phase4/detections.json, runtime.md
.venv/bin/python -m pytest -q tests/test_detector.py
```
