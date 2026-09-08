"""프레임 1장의 트랙 시각화 (Phase 7 실데이터): 이미지에 트랙 ID 라벨 + BEV 를 나란히 저장한다. headless.

트랙 위치(Velodyne 3D) 를 calib.P_velo_to_img 로 이미지에 투영해 ID 를 쓴다. 크기가 있으면 AABB 8 모서리를 투영해 3D 박스 윤곽도 그린다.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from perceptrack3d.evaluation.tracklets import box_corners_bev  # noqa: E402
from perceptrack3d.geometry.calibration import KittiCalibration  # noqa: E402
from perceptrack3d.types import GtBox3D, Track  # noqa: E402


def _color(track_id: int) -> tuple[int, int, int]:
    rgba = plt.get_cmap("tab20")(track_id % 20)
    return (int(rgba[2] * 255), int(rgba[1] * 255), int(rgba[0] * 255))   # BGR


def _project(calib: KittiCalibration, pts: np.ndarray) -> np.ndarray:
    """(K, 3) Velodyne → (K, 3) [u, v, depth]. depth <= 0 인 행은 nan."""
    P = calib.P_velo_to_img
    h = np.hstack([pts, np.ones((len(pts), 1))]) @ P.T
    d = h[:, 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        u, v = h[:, 0] / d, h[:, 1] / d
    bad = d <= 0
    u[bad] = v[bad] = np.nan
    return np.stack([u, v, d], axis=1)


def draw_tracks_on_image(image_bgr: np.ndarray, tracks: list[Track], calib: KittiCalibration) -> np.ndarray:
    """확정 트랙의 중심을 투영해 원 + "ID n (d m)" 라벨, size 가 있으면 AABB 윤곽을 그린 복사본을 돌려준다.

    coasting(misses > 0) 트랙은 점선 느낌으로 얇게 그리고 라벨에 '?' 를 붙인다.
    """
    out = image_bgr.copy()
    H, W = out.shape[:2]
    for tr in tracks:
        c = np.asarray(tr.position, dtype=np.float64)
        u, v, d = _project(calib, c[None, :])[0]
        if not np.isfinite(u) or not (0 <= u < W and 0 <= v < H):
            continue
        color = _color(tr.track_id)
        thick = 2 if tr.misses == 0 else 1
        if tr.size is not None:
            l, w, h = (float(s) for s in tr.size)
            corners = np.array([[sx * l / 2, sy * w / 2, sz * h / 2] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)]) + c
            pc = _project(calib, corners)
            if np.all(np.isfinite(pc[:, 0])):
                edges = [(0, 1), (0, 2), (1, 3), (2, 3), (4, 5), (4, 6), (5, 7), (6, 7), (0, 4), (1, 5), (2, 6), (3, 7)]
                for a, b in edges:
                    cv2.line(out, (int(pc[a, 0]), int(pc[a, 1])), (int(pc[b, 0]), int(pc[b, 1])), color, thick)
        cv2.circle(out, (int(u), int(v)), 4, color, -1)
        label = f"ID {tr.track_id}{'' if tr.misses == 0 else '?'} {c[0]:.0f}m"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        x1, y1 = int(u) - tw // 2, int(v) - 8
        cv2.rectangle(out, (x1, y1 - th - 3), (x1 + tw + 2, y1 + 2), color, -1)
        cv2.putText(out, label, (x1 + 1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def plot_frame_tracks(
    image_bgr: np.ndarray,
    tracks: list[Track],
    points_velo: np.ndarray,
    calib: KittiCalibration,
    path: str | Path,
    gt: list[GtBox3D] | None = None,
    history: dict[int, np.ndarray] | None = None,
    frame_id: int | None = None,
    xlim: tuple[float, float] = (-5.0, 70.0),
    ylim: tuple[float, float] = (-30.0, 30.0),
) -> Path:
    """위: 트랙 ID 를 그린 이미지, 아래: BEV(점 + 트랙 위치/궤적 + GT 박스) 를 한 파일로 저장한다.

    history: {track_id: (K, 3)} 이 프레임까지의 궤적 (선택). gt: 이 프레임의 GT 박스 (초록).
    """
    img = draw_tracks_on_image(image_bgr, tracks, calib)
    pts = np.asarray(points_velo)
    if len(pts) > 60000:
        pts = pts[np.random.default_rng(0).choice(len(pts), 60000, replace=False)]

    fig = plt.figure(figsize=(12, 12))
    gs = fig.add_gridspec(2, 1, height_ratios=[1, 2.4])
    ax0 = fig.add_subplot(gs[0])
    ax0.imshow(img[:, :, ::-1])
    ax0.set_axis_off()
    ax0.set_title(f"frame {frame_id}: {len(tracks)} confirmed tracks" if frame_id is not None else "tracks")

    ax1 = fig.add_subplot(gs[1])
    ax1.scatter(pts[:, 1], pts[:, 0], s=0.3, c="0.6", linewidths=0, rasterized=True)
    for b in gt or []:
        corners = box_corners_bev(b.center, b.size, b.yaw)
        closed = np.vstack([corners, corners[:1]])
        ax1.plot(closed[:, 1], closed[:, 0], color="tab:green", lw=1.3)
        ax1.text(b.center[1], b.center[0], f"gt{b.tracklet_id}", color="tab:green", fontsize=7, ha="center", va="bottom")
    for tr in tracks:
        rgba = plt.get_cmap("tab20")(tr.track_id % 20)
        c = tr.position
        if history and tr.track_id in history:
            hst = np.asarray(history[tr.track_id])
            ax1.plot(hst[:, 1], hst[:, 0], "-", color=rgba, lw=1.2, alpha=0.8)
        ax1.plot(c[1], c[0], "o" if tr.misses == 0 else "s", color=rgba, ms=6)
        ax1.text(c[1], c[0], f" ID {tr.track_id}", color=rgba, fontsize=8, va="center")
        vel = tr.velocity
        ax1.arrow(c[1], c[0], vel[1] * 0.5, vel[0] * 0.5, color=rgba, head_width=0.5, lw=0.8)
    ax1.plot(0, 0, "^", color="tab:blue", ms=9)
    ax1.set_xlim(ylim[1], ylim[0])
    ax1.set_ylim(*xlim)
    ax1.set_aspect("equal")
    ax1.set_xlabel("y [m] (left <-)")
    ax1.set_ylabel("x [m] (forward)")
    ax1.grid(True, lw=0.3, alpha=0.5)
    ax1.set_title("BEV: tracks (color, arrow = velocity x0.5 s) vs GT (green)")
    fig.tight_layout()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=100)
    plt.close(fig)
    return path
