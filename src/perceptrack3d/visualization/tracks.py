"""트랙 궤적 BEV(bird's-eye view) 시각화. headless (matplotlib Agg) 로 파일에 저장한다.

입력 좌표계: Velodyne (x 전방, y 좌, z 상, m). BEV 는 x 를 위쪽, y 를 왼쪽으로 그린다.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


def plot_bev_tracks(
    tracks_history: dict[int, np.ndarray],
    path: str | Path,
    gt: dict[int, np.ndarray] | None = None,
    title: str | None = None,
) -> Path:
    """트랙 궤적을 BEV 로 그려 path 에 저장한다.

    tracks_history : {track_id: (K, 3) 또는 (K, 2) 배열, 행 = [x, y(, z)] 시간순}
    gt             : 같은 형식의 정답 궤적 (선택, 회색 점선)
    반환: 저장한 파일 경로
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 8))
    cmap = plt.get_cmap("tab20")

    if gt:
        for k, (gid, traj) in enumerate(sorted(gt.items())):
            traj = np.asarray(traj, dtype=float)
            ax.plot(traj[:, 1], traj[:, 0], "--", color="0.6", lw=1.2,
                    label="GT" if k == 0 else None)
            ax.text(traj[0, 1], traj[0, 0], f"gt{gid}", color="0.4", fontsize=8, ha="right", va="top")

    for k, (tid, traj) in enumerate(sorted(tracks_history.items())):
        traj = np.asarray(traj, dtype=float)
        if traj.ndim != 2 or traj.shape[1] < 2:
            raise ValueError(f"트랙 {tid} 궤적 shape 이 (K, 2|3) 이 아닙니다: {traj.shape}")
        color = cmap(k % 20)
        ax.plot(traj[:, 1], traj[:, 0], "-", color=color, lw=1.8)
        ax.plot(traj[0, 1], traj[0, 0], "o", color=color, ms=4)             # 시작점
        ax.plot(traj[-1, 1], traj[-1, 0], "s", color=color, ms=5)            # 끝점
        ax.text(traj[-1, 1], traj[-1, 0], f" ID {tid}", color=color, fontsize=9, va="center")

    # 자차(ego) 위치: 원점, 전방(+x) 을 향한 삼각형
    ax.plot(0.0, 0.0, marker=(3, 0, 0), color="k", ms=12, label="ego")

    ax.set_xlabel("y [m]  (Velodyne, +left)")       # 기본 폰트에 한글이 없어 영어 라벨 사용
    ax.set_ylabel("x [m]  (Velodyne, forward)")
    ax.invert_xaxis()                     # +y(좌측) 가 화면 왼쪽에 오도록
    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right")
    ax.set_title(title or f"BEV tracks (n={len(tracks_history)})")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path
