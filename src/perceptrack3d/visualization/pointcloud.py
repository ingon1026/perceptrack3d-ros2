"""LiDAR 점군 시각화 (Phase 1). headless 환경이므로 모두 파일로 저장한다 (plt.show() 를 부르지 않는다).

입력 점은 Velodyne 프레임 (x 전방, y 좌, z 상, m), shape (N, 3) 또는 (N, 4).
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

DEFAULT_Z_RANGE = (-2.5, 2.5)   # 높이 색 범위 (m). 지면 ≈ -1.7 m


def _xyz(points: np.ndarray) -> np.ndarray:
    pts = np.asarray(points)
    if pts.ndim != 2 or pts.shape[1] not in (3, 4):
        raise ValueError(f"points 는 (N, 3) 또는 (N, 4) 이어야 합니다: {pts.shape}")
    return pts[:, :3].astype(np.float64)


def plot_bev(
    points: np.ndarray,
    path: str | Path,
    x_range: tuple[float, float] = (-20.0, 70.0),
    y_range: tuple[float, float] = (-40.0, 40.0),
    z_range: tuple[float, float] = DEFAULT_Z_RANGE,
    point_size: float = 0.15,
    title: str | None = None,
) -> Path:
    """BEV(bird's-eye view, 위에서 내려다본 x-y 평면) 산점도를 저장한다. 색 = 높이 z.

    그림에서 x(전방)는 위쪽, y(좌)는 왼쪽이 되도록 가로축을 -y 로 둔다 (Velodyne 의 오른손 좌표계를 지키기 위함).
    """
    xyz = _xyz(points)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 9))
    sc = ax.scatter(-xyz[:, 1], xyz[:, 0], c=xyz[:, 2], s=point_size, cmap="viridis",
                    vmin=z_range[0], vmax=z_range[1], linewidths=0, rasterized=True)
    ax.plot(0, 0, marker="^", color="black", markersize=8)   # 차량(센서) 위치
    ax.set_xlim(-y_range[1], -y_range[0])
    ax.set_ylim(*x_range)
    ax.set_aspect("equal")
    ax.set_xlabel("-y  (right +, m)")
    ax.set_ylabel("x  (forward, m)")
    ax.set_title(title or f"BEV: {len(xyz):,} points")
    ax.grid(True, alpha=0.2)
    cb = fig.colorbar(sc, ax=ax, fraction=0.03, pad=0.02)
    cb.set_label("height z (m)")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def plot_3d_scatter(
    points: np.ndarray,
    path: str | Path,
    max_points: int = 60_000,
    seed: int = 0,
    z_range: tuple[float, float] = DEFAULT_Z_RANGE,
    elev: float = 25.0,
    azim: float = -150.0,
    title: str | None = None,
) -> Path:
    """matplotlib 3D 산점도를 저장한다. 점이 많으면 seed 고정 무작위 부표본(max_points)을 쓴다."""
    xyz = _xyz(points)
    if len(xyz) > max_points:
        rng = np.random.default_rng(seed)
        xyz = xyz[rng.choice(len(xyz), size=max_points, replace=False)]
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection="3d")
    sc = ax.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], c=xyz[:, 2], s=0.3, cmap="viridis",
                    vmin=z_range[0], vmax=z_range[1], linewidths=0)
    ax.set_xlabel("x forward (m)")
    ax.set_ylabel("y left (m)")
    ax.set_zlabel("z up (m)")
    ax.set_xlim(-20, 70)
    ax.set_ylim(-40, 40)
    ax.set_zlim(-3, 5)
    ax.set_box_aspect((90, 80, 8))
    ax.view_init(elev=elev, azim=azim)
    ax.set_title(title or f"3D scatter: {len(xyz):,} / {len(_xyz(points)):,} points")
    cb = fig.colorbar(sc, ax=ax, shrink=0.5, pad=0.05)
    cb.set_label("height z (m)")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def to_open3d_pointcloud(points: np.ndarray):
    """(N, 3|4) → open3d.geometry.PointCloud. 4열이면 reflectance 를 회색 밝기로 넣는다."""
    import open3d as o3d  # 지연 import: 시각화가 필요 없는 코드가 open3d 에 묶이지 않게

    pts = np.asarray(points)
    xyz = _xyz(pts)
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(xyz))
    if pts.shape[1] == 4:
        r = np.clip(pts[:, 3].astype(np.float64), 0.0, 1.0)
        pcd.colors = o3d.utility.Vector3dVector(np.repeat(r[:, None], 3, axis=1))
    return pcd


def save_ply(points: np.ndarray, path: str | Path) -> Path:
    """점군을 PLY 로 저장한다 (MeshLab / CloudCompare / Open3D 로 열어볼 수 있음)."""
    import open3d as o3d

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not o3d.io.write_point_cloud(str(path), to_open3d_pointcloud(points)):
        raise IOError(f"PLY 저장 실패: {path}")
    return path
