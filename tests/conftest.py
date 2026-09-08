"""공용 fixture. 데이터셋이 없는 환경(CI 등)에서는 데이터 의존 테스트를 skip 한다."""
from pathlib import Path

import pytest

from perceptrack3d.config import load_config


@pytest.fixture(scope="session")
def cfg() -> dict:
    return load_config()


@pytest.fixture(scope="session")
def dataset_available(cfg) -> bool:
    ds = cfg["dataset"]
    return Path(ds["image_dir"]).is_dir() and Path(ds["velo_dir"]).is_dir() and Path(ds["calib_dir"]).is_dir()


@pytest.fixture
def require_dataset(dataset_available):
    if not dataset_available:
        pytest.skip("KITTI 데이터셋이 configs/kitti.yaml 경로에 없습니다")
