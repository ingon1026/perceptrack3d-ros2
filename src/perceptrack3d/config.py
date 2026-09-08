"""설정 파일(configs/kitti.yaml) 로더. 모든 경로는 여기서만 조합한다."""
from __future__ import annotations

from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "configs" / "kitti.yaml"


def load_config(path: str | Path = DEFAULT_CONFIG) -> dict:
    """YAML 을 읽고 dataset 절에 파생 경로(calib_dir, drive_dir, image_dir, velo_dir, tracklets)를 추가한다."""
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    ds = cfg["dataset"]
    root = Path(ds["root"])
    date, drive = ds["date"], ds["drive"]
    drive_dir = root / date / f"{date}_drive_{drive}_sync"
    ds["calib_dir"] = str(root / date)
    ds["drive_dir"] = str(drive_dir)
    ds["image_dir"] = str(drive_dir / ds["camera"] / "data")
    ds["velo_dir"] = str(drive_dir / "velodyne_points" / "data")
    ds["tracklets"] = str(drive_dir / "tracklet_labels.xml")
    cfg["outputs"]["dir"] = str(REPO_ROOT / cfg["outputs"]["dir"])
    return cfg
