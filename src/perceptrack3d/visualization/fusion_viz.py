"""카메라–LiDAR 융합 결과 시각화 (Phase 5–6). headless: OpenCV 로 이미지 배열을 만들거나 matplotlib Agg 로 파일 저장.

- draw_fusion       : 이미지 위에 투영 점(깊이색) + 2D 박스 + 각 객체에 배정된 점(객체별 색) + 중심 깊이/상태 텍스트
- plot_bev_objects  : BEV(Velodyne x-y) 에 점 + 예측 박스(빨강 점선 AABB/OBB) + 배정 점(객체별 색) + GT 박스(초록 실선)
"""
from __future__ import annotations

from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from perceptrack3d.evaluation.tracklets import box_corners_bev  # noqa: E402
from perceptrack3d.types import GtBox3D, Object3D  # noqa: E402
from perceptrack3d.visualization.overlay import draw_projected_points  # noqa: E402

# 객체별 색 (BGR, OpenCV 용). tab10 계열 6색을 돌려 쓴다.
OBJECT_COLORS_BGR = [(180, 119, 31), (14, 127, 255), (44, 160, 44), (40, 39, 214), (189, 103, 148), (75, 86, 140)]


def _rgb(color_bgr: tuple[int, int, int]) -> tuple[float, float, float]:
    b, g, r = color_bgr
    return (r / 255.0, g / 255.0, b / 255.0)


def _status_text(o: Object3D) -> str:
    if o.status == "ok":
        return f"d={o.center[0]:.1f}m n={o.n_points}"
    return f"{o.status}({o.reason})" if o.reason else o.status


def draw_fusion(
    image_bgr: np.ndarray,
    uv: np.ndarray,
    depth: np.ndarray,
    objects: list[Object3D],
    mask: np.ndarray | None = None,
    draw_all_points: bool = True,
) -> np.ndarray:
    """융합 결과를 이미지 위에 그린 복사본 (H, W, 3) BGR 을 돌려준다.

    Args:
        image_bgr: (H, W, 3) uint8
        uv, depth: project_velo_to_image 결과 (M, 2), (M,)
        objects: fuse_frame_* 결과. point_indices (원본 점 인덱스) 가 있는 객체의 점을 객체 색으로 굵게 찍는다.
        mask: (N,) bool — 투영 함수가 준 원본 점 mask. 이것으로 원본 인덱스 → uv 행 번호를 찾는다.
              None 이면 uv 가 원본 점과 같은 길이(N) 라고 가정한다.
        draw_all_points: True 면 배경으로 모든 투영 점을 깊이색(작은 점) 으로 먼저 찍는다.
    """
    out = draw_projected_points(image_bgr, uv, depth, radius=1) if draw_all_points else image_bgr.copy()
    if mask is not None:
        row_of = np.full(len(mask), -1, dtype=np.int64)              # 원본 인덱스 → uv 행 (없으면 -1)
        row_of[np.flatnonzero(mask)] = np.arange(int(mask.sum()))
    else:
        row_of = np.arange(len(uv))
    H, W = out.shape[:2]
    for k, o in enumerate(objects):
        color = OBJECT_COLORS_BGR[k % len(OBJECT_COLORS_BGR)]
        x1, y1, x2, y2 = (int(round(float(v))) for v in o.xyxy)
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2 if o.status == "ok" else 1)
        if o.point_indices is not None and len(o.point_indices):
            rows = row_of[np.asarray(o.point_indices, dtype=np.int64)]
            rows = rows[rows >= 0]
            for u, v in uv[rows].astype(int):
                if 0 <= u < W and 0 <= v < H:
                    cv2.circle(out, (int(u), int(v)), 2, color, -1)
        label = f"{o.class_name} {_status_text(o)}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)
        ty = y1 - 4 if y1 - th - 6 >= 0 else y2 + th + 4
        cv2.rectangle(out, (x1, ty - th - 3), (x1 + tw + 2, ty + 2), color, -1)
        cv2.putText(out, label, (x1 + 1, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def _draw_bev_box(ax, center, size, yaw, color, ls, lw, label=None) -> None:
    corners = box_corners_bev(center, size, yaw)
    closed = np.vstack([corners, corners[:1]])
    ax.plot(closed[:, 1], closed[:, 0], color=color, ls=ls, lw=lw)
    if label:
        ax.text(center[1], center[0], label, color=color, fontsize=7, ha="center", va="bottom", clip_on=True)


def plot_bev_objects(
    points_velo: np.ndarray,
    objects: list[Object3D],
    path: str | Path,
    gt: list[GtBox3D] | None = None,
    xlim: tuple[float, float] = (-5.0, 70.0),
    ylim: tuple[float, float] = (-30.0, 30.0),
    max_points: int = 60000,
    title: str | None = None,
) -> Path:
    """BEV 에 LiDAR 점(회색), 객체별 배정 점(색), 예측 박스(빨강 점선), 예측 중심(x), GT 박스(초록) 를 그려 저장한다.

    좌표: Velodyne. 세로축 x(전방), 가로축 y(좌측, 그림 왼쪽이 +y). 예측 size 가 None 이면 중심만 표시.
    """
    pts = np.asarray(points_velo)
    assert pts.ndim == 2 and pts.shape[1] >= 3, f"points 는 (N, >=3) 이어야 합니다: {pts.shape}"
    sub = pts
    if len(pts) > max_points:
        sub = pts[np.random.default_rng(0).choice(len(pts), max_points, replace=False)]

    fig, ax = plt.subplots(figsize=(9, 10))
    ax.scatter(sub[:, 1], sub[:, 0], s=0.3, c="0.6", linewidths=0, rasterized=True)
    for b in gt or []:
        _draw_bev_box(ax, b.center, b.size, b.yaw, "tab:green", "-", 1.4, f"gt{b.tracklet_id}:{b.object_type}")
    for k, o in enumerate(objects):
        color = _rgb(OBJECT_COLORS_BGR[k % len(OBJECT_COLORS_BGR)])
        if o.point_indices is not None and len(o.point_indices):
            q = pts[np.asarray(o.point_indices, dtype=np.int64)]
            ax.scatter(q[:, 1], q[:, 0], s=4, color=color, linewidths=0)
        if o.center is None:
            continue
        ax.plot(o.center[1], o.center[0], "x", color="tab:red", ms=7, mew=1.5)
        if o.size is not None:
            _draw_bev_box(ax, o.center, o.size, 0.0 if o.yaw is None else o.yaw, "tab:red", "--", 1.2,
                          f"{o.class_name} {o.confidence:.2f}")
    ax.plot(0, 0, "^", color="tab:blue", ms=9)
    ax.set_xlim(ylim[1], ylim[0])
    ax.set_ylim(*xlim)
    ax.set_aspect("equal")
    ax.set_xlabel("y [m] (left <-)")
    ax.set_ylabel("x [m] (forward)")
    ax.grid(True, lw=0.3, alpha=0.5)
    n_ok = sum(o.status == "ok" for o in objects)
    ax.set_title(title or f"pred (red dashed, {n_ok} ok / {len(objects)} det) vs GT (green, {len(gt or [])})")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    return path
