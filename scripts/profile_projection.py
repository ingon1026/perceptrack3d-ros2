"""Phase 10 (1/2): Python 기준 투영 구현을 cProfile 로 프로파일하고, 프레임당 시간에서 차지하는 비중을 추정한다.

실행: .venv/bin/python scripts/profile_projection.py   → outputs/phase10/profile_python.txt
측정: 프레임 10장(0..9)을 메모리에 미리 올린 뒤 project_velo_to_image 만 프로파일. 로드 시간은 별도로 잰다.
"""
from __future__ import annotations

import cProfile
import io
import json
import platform
import pstats
import re
import statistics
import time
from pathlib import Path

import numpy as np

from perceptrack3d.config import load_config
from perceptrack3d.data.kitti_loader import KittiDataset, load_image, load_velodyne
from perceptrack3d.geometry.calibration import KittiCalibration
from perceptrack3d.geometry.projection import project_velo_to_image

N_FRAMES = 10


def detection_median_ms(outputs_dir: Path) -> float | None:
    """outputs/phase4/runtime.md 에서 '중앙값 xx.xx' 를 읽는다 (없으면 None)."""
    path = outputs_dir / "phase4" / "runtime.md"
    if not path.is_file():
        return None
    m = re.search(r"프레임당 추론 \(ms\):.*?중앙값 ([\d.]+)", path.read_text(encoding="utf-8"))
    return float(m.group(1)) if m else None


def detections_per_frame(outputs_dir: Path) -> float | None:
    path = outputs_dir / "phase4" / "detections.json"
    if not path.is_file():
        return None
    dets = json.loads(path.read_text())
    return sum(len(v) for v in dets.values()) / max(len(dets), 1)


def main() -> None:
    cfg = load_config()
    ds = KittiDataset(cfg)
    calib = KittiCalibration.from_dir(cfg["dataset"]["calib_dir"])
    shape = calib.image_shape
    out_dir = Path(cfg["outputs"]["dir"]) / "phase10"
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) 단계별 시간: LiDAR 로드, 이미지 로드, 투영 (프레임 10장, 각 1회, 중앙값)
    t_velo, t_img, t_proj, frames = [], [], [], []
    for i in range(N_FRAMES):
        t0 = time.perf_counter(); pts = load_velodyne(ds.velodyne_path(i)); t_velo.append((time.perf_counter() - t0) * 1e3)
        t0 = time.perf_counter(); load_image(ds.image_path(i)); t_img.append((time.perf_counter() - t0) * 1e3)
        frames.append(pts)
    for _ in range(3):  # warm-up
        project_velo_to_image(frames[0], calib, shape)
    for pts in frames:
        t0 = time.perf_counter(); project_velo_to_image(pts, calib, shape); t_proj.append((time.perf_counter() - t0) * 1e3)

    # 2) cProfile: 투영만 10장
    prof = cProfile.Profile()
    prof.enable()
    for pts in frames:
        project_velo_to_image(pts, calib, shape)
    prof.disable()
    buf = io.StringIO()
    stats = pstats.Stats(prof, stream=buf)
    stats.sort_stats("tottime").print_stats(10)
    stats.sort_stats("cumulative").print_stats(10)

    # 3) 프레임 시간 비중 추정
    med = lambda v: statistics.median(v)  # noqa: E731
    det_ms = detection_median_ms(Path(cfg["outputs"]["dir"]))
    k_mean = detections_per_frame(Path(cfg["outputs"]["dir"]))
    n_pts = int(np.mean([len(p) for p in frames]))
    lines = [
        "# Phase 10 — Python 기준 투영 구현 프로파일",
        "",
        f"- 환경: {platform.machine()}, Python {platform.python_version()}, numpy {np.__version__}",
        f"- 시퀀스: {cfg['dataset']['date']}_drive_{cfg['dataset']['drive']}_sync, 프레임 0..{N_FRAMES - 1}, 평균 {n_pts:,} 점/프레임",
        "- 측정: time.perf_counter(), 각 단계 프레임당 1회, warm-up 3회 제외, 중앙값 (ms)",
        "",
        "## 단계별 시간 (ms, 중앙값)",
        "",
        "| 단계 | ms |",
        "|---|---|",
        f"| LiDAR .bin 로드 (np.fromfile) | {med(t_velo):.3f} |",
        f"| 이미지 PNG 로드 (cv2.imread) | {med(t_img):.3f} |",
        f"| 투영 project_velo_to_image (Python/numpy) | {med(t_proj):.3f} |",
    ]
    if det_ms is not None:
        lines.append(f"| YOLO 2D 검출 (outputs/phase4/runtime.md 중앙값) | {det_ms:.2f} |")
    lines += ["", "## 비중 추정", ""]
    if det_ms is not None and k_mean is not None:
        base = med(t_velo) + med(t_img) + det_ms + med(t_proj)
        naive = med(t_proj) * k_mean  # 검출마다 frustum 을 위해 투영을 다시 하면 K 배
        lines += [
            f"- 로드+검출+투영 1회 합계 ≈ {base:.2f} ms 중 투영 1회 = {med(t_proj) / base * 100:.1f}%.",
            f"- Phase 5/6 에서 검출(프레임당 평균 {k_mean:.2f}개)마다 frustum 을 위해 투영을 반복하면 추가 ≈ {naive:.2f} ms "
            f"(투영 합계 {med(t_proj) + naive:.2f} ms, 위 합계 대비 {(med(t_proj) + naive) / (base + naive) * 100:.1f}%).",
            "- 즉 검출(신경망)을 제외하면 투영이 순수 파이썬/numpy 단계 중 가장 큰 단일 항목이고, 검출당 반복되는 frustum 이 그 비용을 키운다.",
        ]
    else:
        lines.append("- phase4 runtime.md / detections.json 이 없어 검출 시간 대비 비중은 계산하지 않았다.")
    lines += ["", "## cProfile (투영 10장, tottime 상위 10 → cumulative 상위 10)", "", "```text", buf.getvalue().rstrip(), "```", ""]
    (out_dir / "profile_python.txt").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
