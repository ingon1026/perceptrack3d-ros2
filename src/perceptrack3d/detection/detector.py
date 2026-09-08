"""사전학습 YOLO(ultralytics) 로 image_02 프레임에서 2D 도로 객체를 검출한다 (Phase 4).

입력: (H, W, 3) uint8 **BGR** 이미지 (cv2.imread 결과 그대로). ultralytics 공식 문서는
      np.ndarray 입력을 "HWC, BGR, uint8 (0-255)" 로 정의하므로 색 변환 없이 넣는다.
출력: Detection2D 리스트. xyxy 는 image_02 픽셀 좌표 (x1, y1, x2, y2), float32 (4,), 원점 좌상단.
클래스: COCO 80 클래스 중 configs/kitti.yaml 의 detection.keep_classes 만 유지한다.
       COCO 와 KITTI 라벨은 1:1 이 아니다 (docs/decisions.md "팀 B" 절 참고).

결정성: 사전학습 가중치 + CPU 추론은 난수를 쓰지 않으므로 같은 이미지에 같은 박스가 나온다.
       (torch.manual_seed 는 학습/증강용이라 추론에는 불필요.) tests/test_detector.py 에서 확인한다.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
from ultralytics import YOLO

from perceptrack3d.config import REPO_ROOT
from perceptrack3d.types import Detection2D

# 가중치 기본 저장 위치. .gitignore 의 outputs/* 와 *.pt 에 모두 걸리므로 커밋되지 않는다.
DEFAULT_MODEL_DIR = REPO_ROOT / "outputs" / "models"


def resolve_model_path(model: str | Path) -> Path:
    """설정의 model 값을 실제 경로로 바꾼다.

    - 절대경로거나 이미 존재하는 파일이면 그대로 사용.
    - 'yolov8n.pt' 처럼 이름만 있으면 outputs/models/ 아래로 둔다. 파일이 없으면
      ultralytics 가 이 경로로 자동 다운로드한다 (저장소 루트에 내려받지 않기 위함).
    """
    p = Path(model)
    if p.is_absolute() or p.is_file():
        return p
    return DEFAULT_MODEL_DIR / p.name


def _check_image(image_bgr: np.ndarray) -> None:
    """모듈 경계 shape/dtype 검증."""
    if not isinstance(image_bgr, np.ndarray) or image_bgr.ndim != 3 or image_bgr.shape[2] != 3:
        raise ValueError(f"이미지는 (H, W, 3) 배열이어야 합니다. got shape={getattr(image_bgr, 'shape', None)}")
    if image_bgr.dtype != np.uint8:
        raise ValueError(f"이미지 dtype 은 uint8 이어야 합니다. got {image_bgr.dtype}")


class YoloDetector:
    """configs/kitti.yaml 의 detection 절로 설정되는 YOLO 2D 검출기.

    사용 예:
        det = YoloDetector(load_config())
        dets = det.detect(image_bgr, frame_id=0)
    """

    def __init__(self, cfg: dict):
        dc = cfg["detection"]
        self.model_path = resolve_model_path(dc["model"])
        self.model_path.parent.mkdir(parents=True, exist_ok=True)
        self.conf = float(dc["conf_threshold"])
        self.iou = float(dc["iou_threshold"])
        self.device = str(dc.get("device", "cpu"))
        self.imgsz = int(dc.get("imgsz", 640))
        self.keep_classes = list(dc["keep_classes"])

        self.model = YOLO(str(self.model_path))
        # model.names: {COCO id: 이름}. keep_classes 를 id 목록으로 바꿔 ultralytics 의 classes= 필터에 넘긴다.
        name_to_id = {name: cid for cid, name in self.model.names.items()}
        unknown = [c for c in self.keep_classes if c not in name_to_id]
        if unknown:
            raise ValueError(f"모델 클래스에 없는 keep_classes: {unknown}")
        self.class_ids = sorted(name_to_id[c] for c in self.keep_classes)

        # warm-up: 첫 추론은 커널 초기화 때문에 수 배 느리므로 더미 이미지로 미리 돌려 둔다.
        dummy = np.zeros((375, 1242, 3), dtype=np.uint8)
        for _ in range(2):
            self._predict(dummy)

    def _predict(self, image_bgr: np.ndarray):
        """ultralytics predict 호출. 반환은 Results 1개 (이미지 1장이므로)."""
        results = self.model.predict(
            image_bgr,
            imgsz=self.imgsz,
            conf=self.conf,
            iou=self.iou,
            device=self.device,
            classes=self.class_ids,   # keep_classes 이외는 NMS 전에 버려짐
            verbose=False,
            save=False,               # runs/ 디렉터리를 만들지 않는다
        )
        return results[0]

    def detect(self, image_bgr: np.ndarray, frame_id: int) -> list[Detection2D]:
        """이미지 1장 검출. 신뢰도 내림차순 Detection2D 리스트를 돌려준다."""
        _check_image(image_bgr)
        return self._to_detections(self._predict(image_bgr), image_bgr.shape, frame_id)

    def detect_timed(self, image_bgr: np.ndarray, frame_id: int) -> tuple[list[Detection2D], float]:
        """detect 와 같지만 (검출, 경과 ms) 를 돌려준다.

        측정 구간 = ultralytics predict 한 번 (letterbox 전처리 + 신경망 순전파 + NMS 후처리).
        Detection2D 변환 시간은 포함하지 않는다.
        """
        _check_image(image_bgr)
        t0 = time.perf_counter()
        res = self._predict(image_bgr)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        return self._to_detections(res, image_bgr.shape, frame_id), elapsed_ms

    def _to_detections(self, res, image_shape: tuple, frame_id: int) -> list[Detection2D]:
        h, w = image_shape[:2]
        boxes = res.boxes
        xyxy = boxes.xyxy.cpu().numpy().astype(np.float32)   # (N, 4) 원본 이미지 픽셀 단위
        conf = boxes.conf.cpu().numpy().astype(np.float32)   # (N,)
        cls = boxes.cls.cpu().numpy().astype(int)            # (N,) COCO id
        dets: list[Detection2D] = []
        for i in range(len(cls)):
            name = self.model.names[int(cls[i])]
            if name not in self.keep_classes:   # classes= 필터의 이중 확인
                continue
            box = xyxy[i].copy()
            box[[0, 2]] = np.clip(box[[0, 2]], 0, w)   # 경계 밖 좌표 방지 (ultralytics 도 clip 하지만 계약으로 보장)
            box[[1, 3]] = np.clip(box[[1, 3]], 0, h)
            if box[2] <= box[0] or box[3] <= box[1]:
                continue
            dets.append(Detection2D(
                frame_id=int(frame_id),
                class_name=name,
                class_id=int(cls[i]),
                confidence=float(conf[i]),
                xyxy=box,
            ))
        return dets


def save_detections_json(dets_by_frame: dict[int, list[Detection2D]], path: str | Path) -> None:
    """{"0": [as_dict()...], "1": [...]} 형식으로 저장 (docs/architecture.md 4절)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {str(fid): [d.as_dict() for d in dets_by_frame[fid]] for fid in sorted(dets_by_frame)}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=1)


def load_detections_json(path: str | Path) -> dict[int, list[Detection2D]]:
    """save_detections_json 의 역. 키는 int frame_id, xyxy 는 float32 (4,) 로 복원."""
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    out: dict[int, list[Detection2D]] = {}
    for fid_str, items in payload.items():
        fid = int(fid_str)
        out[fid] = [
            Detection2D(
                frame_id=int(d["frame_id"]),
                class_name=str(d["class_name"]),
                class_id=int(d["class_id"]),
                confidence=float(d["confidence"]),
                xyxy=np.asarray(d["xyxy"], dtype=np.float32),
            )
            for d in items
        ]
    return out
