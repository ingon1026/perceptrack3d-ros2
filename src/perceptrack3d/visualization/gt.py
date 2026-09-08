"""GT(tracklet) 박스를 LiDAR BEV 위에 그리는 시각화. headless (matplotlib Agg) 로 파일 저장만 한다.

BEV 축 배치: 세로축 = Velodyne x (전방, 위쪽), 가로축 = Velodyne y (좌측, **왼쪽**이 +y 가 되도록 축을 뒤집음).
즉 그림은 차량 뒤에서 내려다본 운전자 시점이다.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from perceptrack3d.evaluation.tracklets import box_corners_bev  # noqa: E402
from perceptrack3d.types import GtBox3D, Object3D  # noqa: E402


def _draw_box(ax, center, size, yaw, color: str, ls: str, lw: float, label: str | None = None) -> None:
    corners = box_corners_bev(center, size, yaw)
    closed = np.vstack([corners, corners[:1]])
    ax.plot(closed[:, 1], closed[:, 0], color=color, ls=ls, lw=lw)
    front_mid = (corners[0] + corners[1]) / 2      # 앞변 중점 → 진행 방향 표시
    ax.plot([center[1], front_mid[1]], [center[0], front_mid[0]], color=color, lw=lw)
    if label:
        ax.text(center[1], center[0], label, color=color, fontsize=7, ha="center", va="bottom", clip_on=True)


def plot_bev_gt(
    points_velo: np.ndarray,
    gt_boxes: list[GtBox3D],
    path: str | Path,
    preds: list[Object3D] | None = None,
    xlim: tuple[float, float] = (-10.0, 70.0),
    ylim: tuple[float, float] = (-30.0, 30.0),
    max_points: int = 60000,
    title: str | None = None,
) -> Path:
    """LiDAR BEV 위에 GT 박스(초록 실선)와 예측 박스(빨강 점선)를 그려 저장한다.

    Args:
        points_velo: (N, 3|4) Velodyne 점
        gt_boxes: GtBox3D 리스트 (center 는 기하 중심, size [l, w, h], yaw rad)
        path: 저장 경로 (png)
        preds: Object3D 리스트. center 가 None 이면 건너뛰고, size 가 없으면 중심만 x 로 표시
        xlim/ylim: 표시 범위 (m). xlim 은 전방(x), ylim 은 좌우(y)
        max_points: 점이 이보다 많으면 seed 0 으로 무작위 부분 추출 (그리기 속도)
    Returns: 저장된 파일 경로
    """
    pts = np.asarray(points_velo)
    assert pts.ndim == 2 and pts.shape[1] >= 3, f"points 는 (N, >=3) 이어야 합니다: {pts.shape}"
    if len(pts) > max_points:
        pts = pts[np.random.default_rng(0).choice(len(pts), max_points, replace=False)]

    fig, ax = plt.subplots(figsize=(10, 11))
    ax.scatter(pts[:, 1], pts[:, 0], s=0.3, c="0.45", linewidths=0, rasterized=True)

    for b in gt_boxes:
        _draw_box(ax, b.center, b.size, b.yaw, "tab:green", "-", 1.4, f"{b.tracklet_id}:{b.object_type}")
    for o in preds or []:
        if o.center is None:
            continue
        if o.size is None:
            ax.plot(o.center[1], o.center[0], "x", color="tab:red", ms=6)
        else:
            _draw_box(ax, o.center, o.size, 0.0 if o.yaw is None else o.yaw, "tab:red", "--", 1.2)

    ax.plot(0, 0, "^", color="tab:blue", ms=9)           # 자차(Velodyne 원점)
    ax.set_xlim(ylim[1], ylim[0])                        # +y(좌) 가 그림 왼쪽에 오도록 뒤집음
    ax.set_ylim(*xlim)
    ax.set_aspect("equal")
    ax.set_xlabel("y [m] (left  <-)")
    ax.set_ylabel("x [m] (forward)")
    ax.grid(True, lw=0.3, alpha=0.5)
    ax.set_title(title or f"GT boxes (green) on LiDAR BEV — {len(gt_boxes)} boxes")

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    return path
