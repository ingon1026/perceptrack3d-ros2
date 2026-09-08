"""Phase 1: KITTI 로더 테스트."""
from pathlib import Path

import numpy as np
import pytest

from perceptrack3d.data.kitti_loader import KittiDataset, load_image, load_velodyne


@pytest.mark.usefixtures("require_dataset")
class TestWithDataset:
    def test_frame0_image(self, cfg):
        ds = KittiDataset(cfg)
        img = load_image(ds.image_path(0))
        assert img.dtype == np.uint8
        assert img.ndim == 3 and img.shape[2] == 3
        assert img.shape[:2] == (375, 1242)

    def test_frame0_velodyne(self, cfg):
        ds = KittiDataset(cfg)
        pts = load_velodyne(ds.velodyne_path(0))
        assert pts.dtype == np.float32
        assert pts.ndim == 2 and pts.shape[1] == 4
        assert pts.shape[0] > 10_000
        assert np.isfinite(pts).all()
        assert 0.0 <= pts[:, 3].min() and pts[:, 3].max() <= 1.0   # reflectance

    def test_dataset_len_and_load_frame(self, cfg):
        ds = KittiDataset(cfg)
        assert len(ds) == 297
        assert ds.frame_ids()[0] == 0 and ds.frame_ids()[-1] == 296
        fr = ds.load_frame(0)
        assert set(fr) == {"frame_id", "image", "points", "image_path", "velodyne_path"}
        assert fr["frame_id"] == 0
        assert fr["image_path"].name == "0000000000.png"
        assert fr["velodyne_path"].name == "0000000000.bin"

    def test_image_and_lidar_frame_ids_match(self, cfg):
        ds = KittiDataset(cfg)
        png_ids = {int(p.stem) for p in ds.image_dir.glob("*.png")}
        bin_ids = {int(p.stem) for p in ds.velo_dir.glob("*.bin")}
        assert png_ids == bin_ids
        assert png_ids == set(ds.frame_ids())


def test_missing_image_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_image(tmp_path / "nope.png")


def test_missing_velodyne_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_velodyne(tmp_path / "nope.bin")


def test_corrupt_image_raises(tmp_path):
    bad = tmp_path / "bad.png"
    bad.write_bytes(b"this is not a png")
    with pytest.raises(ValueError):
        load_image(bad)


def test_velodyne_size_not_multiple_of_16_raises(tmp_path):
    bad = tmp_path / "bad.bin"
    bad.write_bytes(b"\x00" * 17)          # 17 bytes: float32 4개(16 bytes)의 배수가 아님
    with pytest.raises(ValueError):
        load_velodyne(bad)


def test_velodyne_empty_raises(tmp_path):
    bad = tmp_path / "empty.bin"
    bad.write_bytes(b"")
    with pytest.raises(ValueError):
        load_velodyne(bad)


def test_velodyne_valid_synthetic(tmp_path):
    pts = np.array([[1, 2, 3, 0.5], [4, 5, 6, 0.25]], dtype=np.float32)
    p = tmp_path / "ok.bin"
    pts.tofile(p)
    out = load_velodyne(p)
    assert out.shape == (2, 4) and out.dtype == np.float32
    np.testing.assert_array_equal(out, pts)


def test_dataset_missing_dir_raises(cfg):
    bad = {"dataset": {**cfg["dataset"], "image_dir": str(Path(cfg["dataset"]["image_dir"]) / "nope")}}
    with pytest.raises(FileNotFoundError):
        KittiDataset(bad)
