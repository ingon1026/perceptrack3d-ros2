"""Phase 10 (2/2): Python(numpy) vs C++(pybind11) 벤치마크 → outputs/phase10/benchmark.md

실행: .venv/bin/python scripts/benchmark_projection.py
측정: 프레임 20장(0..19) × 반복 30회, 프레임마다 warm-up 3회 제외, time.perf_counter(). 구현들을 같은 반복 안에서
      번갈아 호출해 캐시/클럭 상태를 공평하게 한다. mean / median / p95 (ms) 와 speedup = median(Python) / median(C++).
      다른 프로세스(다른 팀 파이프라인·빌드)가 같은 머신에서 돌 수 있으므로 load average 를 기록하고 median 만 대표값으로 쓴다.
      실행마다 median 을 outputs/phase10/benchmark_runs.csv 에 누적해, 부하가 다른 여러 실행을 benchmark.md 에 나란히 보여 준다.
시나리오
  A. 투영 (전체 프레임)             Python project_velo_to_image  vs  C++ project_velo_to_image_cpp
  B. ROI 마스크                      Python numpy(팀 E roi_filter 가 있으면 그것)  vs  C++ roi_mask_cpp
  C. frustum, 프레임의 실제 검출 K개  Python naive(박스마다 재투영) / Python once(투영 1회 + 박스별 비교) / 팀 E frustum_mask(있으면) vs C++
  D. C++ 고정 호출 오버헤드 (점 10개)
"""
from __future__ import annotations

import csv
import json
import os
import platform
import subprocess
import time
from datetime import datetime
from pathlib import Path

import numpy as np

from perceptrack3d.config import load_config
from perceptrack3d.data.kitti_loader import KittiDataset
from perceptrack3d.geometry.calibration import KittiCalibration
from perceptrack3d.geometry.projection import project_velo_to_image
from perceptrack3d.geometry.projection_cpp import frustum_masks_cpp, project_velo_to_image_cpp, roi_mask_cpp

try:
    from perceptrack3d.fusion import point_filters as pf
except ImportError:
    pf = None

N_FRAMES, N_REPS, N_WARMUP = 20, 30, 3


# ---- Python 참조 구현 --------------------------------------------------------------------------------
def roi_numpy(pts: np.ndarray, roi: dict) -> np.ndarray:
    x, y, z = pts[:, 0], pts[:, 1], pts[:, 2]
    return (x >= roi["x_min"]) & (x <= roi["x_max"]) & (np.abs(y) <= roi["y_abs_max"]) & (z >= roi["z_min"]) & (z <= roi["z_max"])


def _box_test(uv: np.ndarray, box: np.ndarray) -> np.ndarray:
    x1, y1, x2, y2 = box
    return (uv[:, 0] >= x1) & (uv[:, 0] <= x2) & (uv[:, 1] >= y1) & (uv[:, 1] <= y2)


def frustum_py_naive(pts, boxes, calib, shape) -> np.ndarray:
    """박스마다 투영을 다시 한다 (검출당 frustum_mask 를 호출하는 파이프라인의 소박한 형태)."""
    out = np.zeros((len(boxes), len(pts)), dtype=bool)
    for k, box in enumerate(boxes):
        uv, _, mask = project_velo_to_image(pts, calib, shape)
        out[k, np.flatnonzero(mask)[_box_test(uv, box)]] = True
    return out


def frustum_py_once(pts, boxes, calib, shape) -> np.ndarray:
    """투영 1회 + 박스별 비교 (numpy 로 할 수 있는 최선)."""
    uv, _, mask = project_velo_to_image(pts, calib, shape)
    idx = np.flatnonzero(mask)
    out = np.zeros((len(boxes), len(pts)), dtype=bool)
    for k, box in enumerate(boxes):
        out[k, idx[_box_test(uv, box)]] = True
    return out


def frustum_team_e(pts, boxes, calib, shape) -> np.ndarray:
    return np.stack([pf.frustum_mask(pts, box, calib, shape, 0.0) for box in boxes]) if len(boxes) else np.zeros((0, len(pts)), bool)


# ---- 측정 --------------------------------------------------------------------------------------------
def time_call(fn) -> tuple[float, object]:
    t0 = time.perf_counter()
    out = fn()
    return (time.perf_counter() - t0) * 1e3, out


def stats(samples: list[float]) -> dict:
    a = np.asarray(samples)
    return {"mean": a.mean(), "median": float(np.median(a)), "p95": float(np.percentile(a, 95)), "n": len(a)}


def hardware_info() -> dict:
    def sh(cmd):
        try:
            return subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=5).stdout.strip()
        except Exception:  # noqa: BLE001
            return "?"
    model = sh("lscpu | grep 'Model name' | sed 's/.*: *//'")
    return {
        "cpu": model or platform.processor(),
        "logical_cores": os.cpu_count(),
        "physical_cores": sh("lscpu | grep '^Core(s) per socket' | sed 's/.*: *//'"),
        "gxx": sh("g++ --version | head -1"),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "os": sh("uname -sr"),
        "loadavg": "%.1f / %.1f / %.1f" % os.getloadavg(),  # 측정 직전 1/5/15분 load average (다른 작업의 간섭 여부 기록)
        "load1": os.getloadavg()[0],
    }


def main() -> None:
    cfg = load_config()
    ds = KittiDataset(cfg)
    calib = KittiCalibration.from_dir(cfg["dataset"]["calib_dir"])
    shape = calib.image_shape
    roi = cfg["fusion"]["roi"]
    out_dir = Path(cfg["outputs"]["dir"]) / "phase10"
    out_dir.mkdir(parents=True, exist_ok=True)

    det_path = Path(cfg["outputs"]["dir"]) / "phase4" / "detections.json"
    dets = json.loads(det_path.read_text()) if det_path.is_file() else {}
    frames, boxes_per_frame = [], []
    for i in range(N_FRAMES):
        frames.append(ds.load_frame(i)["points"])
        bx = np.array([d["xyxy"] for d in dets.get(str(i), [])], dtype=np.float64).reshape(-1, 4)
        if len(bx) == 0:  # 검출 결과가 없으면 합성 박스 5개
            bx = np.array([[200 + 180 * k, 150, 300 + 180 * k, 230] for k in range(5)], dtype=np.float64)
        boxes_per_frame.append(bx)

    roi_py = (lambda p: pf.roi_filter(p, roi)) if (pf is not None and hasattr(pf, "roi_filter")) else (lambda p: roi_numpy(p, roi))
    roi_py_name = "Python (팀 E roi_filter)" if (pf is not None and hasattr(pf, "roi_filter")) else "Python (numpy)"
    have_team_e_frustum = pf is not None and hasattr(pf, "frustum_mask")

    impls = {
        "A_py": lambda p, b: project_velo_to_image(p, calib, shape),
        "A_cpp": lambda p, b: project_velo_to_image_cpp(p, calib, shape),
        "B_py": lambda p, b: roi_py(p),
        "B_cpp": lambda p, b: roi_mask_cpp(p, roi),
        "C_py_naive": lambda p, b: frustum_py_naive(p, b, calib, shape),
        "C_py_once": lambda p, b: frustum_py_once(p, b, calib, shape),
        "C_cpp": lambda p, b: frustum_masks_cpp(p, b, calib, shape),
    }
    if have_team_e_frustum:
        impls["C_py_teamE"] = lambda p, b: frustum_team_e(p, b, calib, shape)
    small = frames[0][:10].copy()
    impls["D_cpp_overhead"] = lambda p, b: project_velo_to_image_cpp(small, calib, shape)

    samples: dict[str, list[float]] = {k: [] for k in impls}
    mismatches = {"A_mask": 0, "A_uv": 0, "A_depth": 0, "B": 0, "C_once_vs_cpp": 0, "C_naive_vs_cpp": 0, "C_teamE_vs_cpp": 0}
    for pts, boxes in zip(frames, boxes_per_frame):
        for _ in range(N_WARMUP):
            for fn in impls.values():
                fn(pts, boxes)
        for _ in range(N_REPS):
            for name, fn in impls.items():
                ms, _ = time_call(lambda fn=fn: fn(pts, boxes))
                samples[name].append(ms)
        # 수치 일치 (프레임당 1회)
        uv_p, d_p, m_p = impls["A_py"](pts, boxes)
        uv_c, d_c, m_c = impls["A_cpp"](pts, boxes)
        mismatches["A_mask"] += int(not np.array_equal(m_p, m_c))
        mismatches["A_uv"] += int(not np.allclose(uv_p, uv_c, atol=1e-3))
        mismatches["A_depth"] += int(not np.allclose(d_p, d_c, rtol=1e-5))
        mismatches["B"] += int(not np.array_equal(impls["B_py"](pts, boxes), impls["B_cpp"](pts, boxes)))
        c_cpp = impls["C_cpp"](pts, boxes)
        mismatches["C_once_vs_cpp"] += int(not np.array_equal(impls["C_py_once"](pts, boxes), c_cpp))
        mismatches["C_naive_vs_cpp"] += int(not np.array_equal(impls["C_py_naive"](pts, boxes), c_cpp))
        if have_team_e_frustum:
            mismatches["C_teamE_vs_cpp"] += int(not np.array_equal(impls["C_py_teamE"](pts, boxes), c_cpp))

    st = {k: stats(v) for k, v in samples.items()}
    hw = hardware_info()
    under_load = hw["load1"] > 0.5 * (hw["logical_cores"] or 1)

    # 실행 이력: median 만 CSV 에 누적한다 (부하가 다른 실행을 나란히 비교하기 위해)
    run_keys = ["A_py", "A_cpp", "B_py", "B_cpp", "C_py_naive", "C_py_once", "C_py_teamE", "C_cpp", "D_cpp_overhead"]
    csv_path = out_dir / "benchmark_runs.csv"
    new_row = {"time": datetime.now().strftime("%Y-%m-%d %H:%M"), "loadavg_1min": f"{hw['load1']:.1f}", "reps": N_REPS,
               **{k: (f"{st[k]['median']:.4f}" if k in st else "") for k in run_keys}}
    write_header = not csv_path.is_file()
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(new_row.keys()))
        if write_header:
            w.writeheader()
        w.writerow(new_row)
    with open(csv_path, newline="", encoding="utf-8") as f:
        history = list(csv.DictReader(f))
    n_pts = [len(p) for p in frames]
    k_list = [len(b) for b in boxes_per_frame]

    def row(label, py_key, cpp_key):
        p, c = st[py_key], st[cpp_key]
        return f"| {label} | {p['mean']:.3f} / {p['median']:.3f} / {p['p95']:.3f} | {c['mean']:.3f} / {c['median']:.3f} / {c['p95']:.3f} | {p['median'] / c['median']:.1f}x |"

    lines = [
        "# Phase 10 벤치마크 — Python(numpy) vs C++(pybind11)",
        "",
        "## 환경",
        f"- CPU: {hw['cpu']} (물리 코어 {hw['physical_cores']}, 논리 {hw['logical_cores']}), {hw['os']}",
        f"- 컴파일러: {hw['gxx']}, C++17, 플래그 `-O2 -DNDEBUG` (CMAKE_CXX_FLAGS_RELEASE 고정, -march=native 없음; 확장 모듈은 pybind11 기본 `-flto`, `-fvisibility=hidden` 추가)",
        f"- Python {hw['python']}, numpy {hw['numpy']}, pybind11 3.1.0, Eigen 3 (헤더 전용)",
        f"- 측정 시작 시 load average (1/5/15분): {hw['loadavg']}"
        + (" — **다른 프로세스 부하 하에서 측정** (1분 load 가 논리 코어의 절반 초과). mean/p95 는 외부 간섭이 섞여 있으므로 median 만 대표값으로 쓴다." if under_load else " — 한가한 머신"),
        f"- 시퀀스: {cfg['dataset']['date']}_drive_{cfg['dataset']['drive']}_sync, 프레임 0..{N_FRAMES - 1}, 점 수 평균 {np.mean(n_pts):,.0f} (최소 {min(n_pts):,}, 최대 {max(n_pts):,}), 이미지 {shape[1]}x{shape[0]}",
        f"- frustum 박스: outputs/phase4/detections.json 의 실제 검출, 프레임당 평균 {np.mean(k_list):.2f}개 (최소 {min(k_list)}, 최대 {max(k_list)})",
        f"- 측정: 프레임 {N_FRAMES}장 × 반복 {N_REPS}회 = {N_FRAMES * N_REPS} 샘플/구현, 프레임마다 warm-up {N_WARMUP}회 제외, `time.perf_counter()`, 구현을 같은 반복 안에서 번갈아 호출. 단일 스레드.",
        "- 입력은 로더의 (N,4) float32 C-연속 배열을 그대로 넘긴다 (C++ 은 stride 로 앞 3열만 읽음, 복사 없음).",
        "",
        "## 결과 (ms: mean / median / p95, speedup = median 비)",
        "",
        "| 시나리오 | Python | C++ | speedup |",
        "|---|---|---|---|",
        row("A. 투영 (전체 프레임)", "A_py", "A_cpp"),
        row(f"B. ROI 마스크 — {roi_py_name}", "B_py", "B_cpp"),
        row("C1. frustum K박스 — Python naive (박스마다 재투영)", "C_py_naive", "C_cpp"),
        row("C2. frustum K박스 — Python once (투영 1회 + 박스별 비교)", "C_py_once", "C_cpp"),
    ]
    if have_team_e_frustum:
        lines.append(row("C3. frustum K박스 — 팀 E frustum_mask 를 박스마다 호출", "C_py_teamE", "C_cpp"))
    d = st["D_cpp_overhead"]
    lines += [
        f"| D. C++ 호출 고정 오버헤드 (점 10개 투영) | - | {d['mean']:.4f} / {d['median']:.4f} / {d['p95']:.4f} | - |",
        "",
        "## 수치 일치 (프레임 20장, 각 1회 비교; 값 = 불일치 프레임 수)",
        "",
        f"- A 투영: mask 완전 일치 실패 {mismatches['A_mask']}, uv allclose(atol 1e-3) 실패 {mismatches['A_uv']}, depth allclose(rtol 1e-5) 실패 {mismatches['A_depth']}",
        f"- B ROI: 완전 일치 실패 {mismatches['B']}",
        f"- C frustum: Python once vs C++ 실패 {mismatches['C_once_vs_cpp']}, Python naive vs C++ 실패 {mismatches['C_naive_vs_cpp']}"
        + (f", 팀 E vs C++ 실패 {mismatches['C_teamE_vs_cpp']}" if have_team_e_frustum else " (팀 E frustum_mask 없음: 비교 생략)"),
        "",
    ]
    lines += [
        "## 실행 이력 (median ms; 부하가 다른 실행을 나란히 비교, 출처 benchmark_runs.csv)",
        "",
        "| 실행 시각 | load(1분) | 반복 | A py | A cpp | B py | B cpp | C1 py naive | C2 py once | C3 py 팀E | C cpp | D cpp 오버헤드 |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in history:
        lines.append("| " + " | ".join([r["time"], r["loadavg_1min"], r["reps"]] + [r.get(k, "") or "-" for k in run_keys]) + " |")
    lines.append("")
    text = "\n".join(lines)
    (out_dir / "benchmark.md").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
