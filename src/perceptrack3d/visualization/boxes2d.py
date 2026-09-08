"""2D 검출 박스 시각화 (Phase 4). 입력/출력 모두 (H, W, 3) uint8 BGR."""
from __future__ import annotations

import cv2
import numpy as np

from perceptrack3d.types import Detection2D

# 클래스별 색 (BGR). keep_classes 에 없는 이름은 회색.
CLASS_COLORS: dict[str, tuple[int, int, int]] = {
    "car": (0, 200, 0),
    "truck": (0, 140, 255),
    "bus": (255, 0, 255),
    "person": (0, 0, 255),
    "bicycle": (255, 200, 0),
    "motorcycle": (255, 0, 0),
}


def draw_detections(image_bgr: np.ndarray, dets: list[Detection2D], color_by_class: bool = True) -> np.ndarray:
    """박스와 'class conf' 텍스트를 그린 복사본을 돌려준다 (원본은 수정하지 않음)."""
    out = image_bgr.copy()
    for d in dets:
        x1, y1, x2, y2 = (int(round(v)) for v in d.xyxy)
        color = CLASS_COLORS.get(d.class_name, (160, 160, 160)) if color_by_class else (0, 255, 0)
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        label = f"{d.class_name} {d.confidence:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        ty = y1 - 4 if y1 - th - 6 >= 0 else y2 + th + 4   # 위쪽 공간이 없으면 박스 아래에
        cv2.rectangle(out, (x1, ty - th - 3), (x1 + tw + 2, ty + 2), color, -1)
        cv2.putText(out, label, (x1 + 1, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    return out
