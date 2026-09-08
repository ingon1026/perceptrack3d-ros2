"""Phase 4 검출기 테스트. 모델 로드는 느리므로 module 범위 fixture 로 1회만 만든다."""
import numpy as np
import pytest

from perceptrack3d.config import load_config
from perceptrack3d.data.kitti_loader import KittiDataset
from perceptrack3d.detection.detector import YoloDetector, load_detections_json, save_detections_json
from perceptrack3d.types import Detection2D


@pytest.fixture(scope="module")
def cfg():
    return load_config()


@pytest.fixture(scope="module")
def detector(cfg):
    return YoloDetector(cfg)


@pytest.fixture(scope="module")
def frame0(cfg):
    return KittiDataset(cfg).load_frame(0)["image"]


def test_detect_frame0_format(cfg, detector, frame0):
    dets = detector.detect(frame0, frame_id=0)
    h, w = frame0.shape[:2]
    keep = set(cfg["detection"]["keep_classes"])
    assert isinstance(dets, list) and len(dets) > 0
    for d in dets:
        assert isinstance(d, Detection2D)
        assert d.frame_id == 0
        assert d.xyxy.shape == (4,) and d.xyxy.dtype == np.float32
        x1, y1, x2, y2 = d.xyxy
        assert x1 < x2 and y1 < y2
        assert 0 <= x1 and x2 <= w and 0 <= y1 and y2 <= h
        assert 0.0 <= d.confidence <= 1.0
        assert d.confidence >= cfg["detection"]["conf_threshold"]
        assert d.class_name in keep
        assert detector.model.names[d.class_id] == d.class_name


def test_detect_deterministic(detector, frame0):
    a = detector.detect(frame0, frame_id=0)
    b = detector.detect(frame0, frame_id=0)
    assert [d.class_name for d in a] == [d.class_name for d in b]
    assert np.allclose(np.stack([d.xyxy for d in a]), np.stack([d.xyxy for d in b]))
    assert np.allclose([d.confidence for d in a], [d.confidence for d in b])


def test_json_roundtrip(detector, frame0, tmp_path):
    dets = detector.detect(frame0, frame_id=0)
    path = tmp_path / "dets.json"
    save_detections_json({0: dets, 1: []}, path)
    loaded = load_detections_json(path)
    assert set(loaded) == {0, 1} and loaded[1] == []
    assert len(loaded[0]) == len(dets)
    for d, l in zip(dets, loaded[0]):
        assert l.frame_id == d.frame_id and l.class_name == d.class_name and l.class_id == d.class_id
        assert l.xyxy.dtype == np.float32 and np.allclose(l.xyxy, d.xyxy)
        assert l.confidence == pytest.approx(d.confidence)


def test_rejects_bad_input(detector):
    with pytest.raises(ValueError):
        detector.detect(np.zeros((375, 1242), dtype=np.uint8), frame_id=0)   # 흑백 2D
    with pytest.raises(ValueError):
        detector.detect(np.zeros((375, 1242, 3), dtype=np.float32), frame_id=0)   # dtype 오류
