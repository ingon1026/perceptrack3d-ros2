"""Phase 8 정량 평가를 한 명령으로 재현한다: 변형 A/B/C 를 tracklet GT 와 비교해 표·플롯·런타임 보고서를 만든다.

실행
    .venv/bin/python scripts/evaluate.py --config configs/kitti.yaml --out outputs/phase8 [--frames 0 297]
입력 (먼저 scripts/run_pipeline.py --variant raw / clustered / tracked 를 실행해 둘 것)
    outputs/phase5/objects_raw.json        변형 A: 2D 박스 + raw LiDAR 통계
    outputs/phase6/objects_clustered.json  변형 B: 2D 박스 + 필터/클러스터 LiDAR
    outputs/phase7/tracks.json             변형 C: B + 칼만 추적 (confirmed 트랙)
    outputs/phase{5,6,7}/runtime.json      단계별 ms, outputs/phase4/runtime.md (YOLO 추론 시간 인용)
산출 (--out 아래)
    results.csv        변형 × 임계값 × 지표 long 형식 (variant, max_dist_m, section, group, metric, value)
    results.json       evaluate_variant 결과 전체 (표 제외) + 평가 설정
    results.md         비교 표 (전체 / 거리 구간 / 클래스 / 추적) 와 무효 비교 목록
    runtime.md         하드웨어·측정 방법·단계별 ms·FPS
    config_used.yaml   평가 시점의 설정 파일 복사본
    plots/01..06_*.png 오차 CDF, 거리-오차, 재현율/정밀도, 프레임별 매칭 수, ID 커버 간트, 런타임
플롯 안 글자는 영어 (기본 글꼴에 한글 글리프가 없음), 설명은 이 파일과 results.md 의 한국어.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import platform
import re
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from perceptrack3d.config import DEFAULT_CONFIG, load_config  # noqa: E402
from perceptrack3d.evaluation.evaluate import (  # noqa: E402
    RANGE_BINS,
    evaluate_variant,
    fmt,
    results_to_long,
)
from perceptrack3d.evaluation.metrics import summarize_runtime  # noqa: E402
from perceptrack3d.evaluation.tracklets import gt_boxes_by_frame  # noqa: E402
from perceptrack3d.fusion.frustum_fusion import load_objects_json  # noqa: E402

# 변형 정의 (CLAUDE.md Phase 8): 이름, 결과 파일, 형식, 런타임 파일
VARIANTS = [
    ("A_raw", "phase5/objects_raw.json", "objects", "phase5/runtime.json"),
    ("B_clustered", "phase6/objects_clustered.json", "objects", "phase6/runtime.json"),
    ("C_tracked", "phase7/tracks.json", "tracks", "phase7/runtime.json"),
]
LABEL = {"A_raw": "A raw", "B_clustered": "B clustered", "C_tracked": "C tracked"}
# 변형별 고정 색 (dataviz 팔레트 앞 3 슬롯: 정상 시각·색각 이상 모두에서 구분되도록 검증된 순서)
COLOR = {"A_raw": "#2a78d6", "B_clustered": "#eb6834", "C_tracked": "#1baf7a"}
GT_COLOR = "#6b6a66"
STAGES = ["load", "projection", "filters", "cluster", "fusion", "tracking"]
THRESHOLDS = (2.0, 4.0)


# ----------------------------------------------------------------------------- 입력
def load_variant(out_root: Path, rel: str, kind: str) -> dict:
    path = out_root / rel
    if not path.is_file():
        sys.exit(f"결과 파일이 없습니다: {path}  (먼저 scripts/run_pipeline.py 를 실행하세요)")
    if kind == "objects":
        return load_objects_json(path)
    with open(path, "r", encoding="utf-8") as fp:
        return {int(k): v for k, v in json.load(fp).items()}


def load_runtime(path: Path) -> dict | None:
    if not path.is_file():
        return None
    with open(path, "r", encoding="utf-8") as fp:
        d = json.load(fp)
    d["_mtime"] = datetime.fromtimestamp(path.stat().st_mtime).strftime("%Y-%m-%d %H:%M")
    return d


def detection_runtime(phase4_md: Path) -> dict | None:
    """outputs/phase4/runtime.md 의 '프레임당 추론 (ms): 평균 a, 중앙값 b, p95 c' 줄을 읽는다 (팀 B 측정값 인용)."""
    if not phase4_md.is_file():
        return None
    m = re.search(r"프레임당 추론 \(ms\): 평균 ([\d.]+), 중앙값 ([\d.]+), p95 ([\d.]+)", phase4_md.read_text(encoding="utf-8"))
    if not m:
        return None
    return {"mean": float(m.group(1)), "median": float(m.group(2)), "p95": float(m.group(3)), "source": str(phase4_md)}


# ----------------------------------------------------------------------------- 플롯
def plot_error_cdf(results: dict, out: Path) -> Path:
    """① 변형별 BEV 중심 오차 CDF (매칭 임계값 2 m / 4 m 두 패널). 임계값이 오차의 상한이므로 두 패널을 같이 본다."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    for ax, thr in zip(axes, THRESHOLDS):
        for name, _, _, _ in VARIANTS:
            e = np.sort(results[(name, thr)]["pairs"]["dist_bev"].to_numpy())
            if e.size == 0:
                continue
            y = np.arange(1, e.size + 1) / e.size
            ax.plot(e, y, color=COLOR[name], lw=2, label=f"{LABEL[name]} (n={e.size}, median {np.median(e):.2f} m)")
        ax.set_xlim(0, thr)
        ax.set_ylim(0, 1)
        ax.set_xlabel("BEV center error [m]")
        ax.set_title(f"match threshold {thr:.0f} m")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="lower right", fontsize=8)
    axes[0].set_ylabel("fraction of matched pairs <= error")
    fig.suptitle("BEV center error CDF per variant (matched pairs only)")
    fig.tight_layout()
    path = out / "01_bev_error_cdf.png"
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def plot_range(results: dict, out: Path, thr: float) -> Path:
    """② 왼쪽: GT 거리 vs BEV 오차 산점 (변형별). 오른쪽: 거리 구간별 재현율 막대 (막대 위 = 구간 GT 수)."""
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(12, 4.4), gridspec_kw={"width_ratios": [1.5, 1]})
    for name, _, _, _ in VARIANTS:
        p = results[(name, thr)]["pairs"]
        ax0.scatter(p["gt_range"], p["dist_bev"], s=7, alpha=0.35, color=COLOR[name], linewidths=0, label=LABEL[name])
    for lo, hi in RANGE_BINS[1:]:
        ax0.axvline(lo, color="0.7", lw=0.8, ls="--")
    ax0.set_xlabel("GT BEV range from ego [m]")
    ax0.set_ylabel("BEV center error [m]")
    ax0.set_ylim(0, thr)
    ax0.set_title(f"error vs range (threshold {thr:.0f} m)")
    ax0.grid(True, alpha=0.3)
    ax0.legend(loc="upper left", fontsize=8, markerscale=2.5)

    labels = [f"{lo:.0f}-{hi:.0f} m" for lo, hi in RANGE_BINS]
    x = np.arange(len(labels))
    w = 0.26
    for k, (name, _, _, _) in enumerate(VARIANTS):
        br = results[(name, thr)]["by_range"]
        rec = [br[l.split(" ")[0]]["recall"] for l in labels]
        bars = ax1.bar(x + (k - 1) * w, rec, w, color=COLOR[name], label=LABEL[name])
        for b, l in zip(bars, labels):
            ax1.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.01, f"{b.get_height():.2f}", ha="center", va="bottom", fontsize=7)
    n_gt = [results[(VARIANTS[0][0], thr)]["by_range"][l.split(" ")[0]]["n_gt"] for l in labels]
    ax1.set_xticks(x, [f"{l}\n(GT n={n})" for l, n in zip(labels, n_gt)])
    ax1.set_ylim(0, 1.05)
    ax1.set_ylabel("recall")
    ax1.set_title(f"recall by GT range (threshold {thr:.0f} m)")
    ax1.grid(True, axis="y", alpha=0.3)
    ax1.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    path = out / "02_range_vs_error.png"
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def plot_precision_recall(results: dict, out: Path) -> Path:
    """③ 재현율·정밀도·F1 막대 (변형 × 임계값 2 m / 4 m)."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    metrics = [("recall", "Recall"), ("precision", "Precision"), ("f1", "F1")]
    x = np.arange(len(metrics))
    w = 0.26
    for ax, thr in zip(axes, THRESHOLDS):
        for k, (name, _, _, _) in enumerate(VARIANTS):
            o = results[(name, thr)]["overall"]
            vals = [o[m] for m, _ in metrics]
            bars = ax.bar(x + (k - 1) * w, vals, w, color=COLOR[name], label=LABEL[name])
            for b in bars:
                ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.01, f"{b.get_height():.2f}", ha="center", va="bottom", fontsize=8)
        ax.set_xticks(x, [t for _, t in metrics])
        ax.set_ylim(0, 1.08)
        ax.set_title(f"match threshold {thr:.0f} m")
        ax.grid(True, axis="y", alpha=0.3)
        ax.legend(loc="upper left", fontsize=8)
    n = results[(VARIANTS[0][0], THRESHOLDS[0])]["overall"]["n_gt"]
    fig.suptitle(f"Detection quality per variant (GT in camera FOV, x <= 70 m, n_gt={n})")
    fig.tight_layout()
    path = out / "03_precision_recall.png"
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def plot_per_frame(results: dict, out: Path, thr: float) -> Path:
    """④ 프레임별 매칭 수 시계열 (A/B/C) 과 GT 수 (회색 영역)."""
    fig, ax = plt.subplots(figsize=(12, 4))
    pf0 = results[(VARIANTS[0][0], thr)]["per_frame"]
    ax.fill_between(pf0["frame"], 0, pf0["n_gt"], color=GT_COLOR, alpha=0.18, step="mid", label="GT count (FOV, x<=70 m)")
    for name, _, _, _ in VARIANTS:
        pf = results[(name, thr)]["per_frame"]
        ax.plot(pf["frame"], pf["n_matched"], color=COLOR[name], lw=1.4, label=f"{LABEL[name]} matched")
    ax.set_xlabel("frame")
    ax.set_ylabel("count per frame")
    ax.set_title(f"Matched objects per frame (threshold {thr:.0f} m)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", fontsize=8, ncol=2)
    ax.set_xlim(pf0["frame"].min(), pf0["frame"].max())
    fig.tight_layout()
    path = out / "04_per_frame_matches.png"
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def plot_gantt(result: dict, out: Path) -> Path:
    """⑤ 변형 C: GT tracklet 별 프레임 구간(회색) 위에 매칭된 track_id 를 색·글자로 표시하는 간트차트.

    한 GT 행에 색이 바뀌면 ID 가 바뀐 것(ID 스위치 또는 단절 후 재생성), 회색만 남은 구간은 미검출(FN).
    """
    gts, tr = result["gts"], result["tracking"]
    cov = tr["coverage"]
    order = sorted(cov, key=lambda g: (cov[g]["first_frame"], g))
    cmap = plt.get_cmap("tab20")
    fig, ax = plt.subplots(figsize=(13, 0.28 * len(order) + 1.8))
    for row, gid in enumerate(order):
        g = gts[gts["gt_id"] == gid].sort_values("frame")
        frames = g["frame"].to_numpy()
        # 존재 구간 (연속 프레임을 한 덩어리로)
        for seg in _segments(frames):
            ax.barh(row, seg[1] - seg[0] + 1, left=seg[0] - 0.5, height=0.7, color="0.85", edgecolor="none")
        # 매칭 구간을 track_id 별로
        m = g[g["matched"]]
        for tid, gm in m.groupby("track_id"):
            for seg in _segments(gm["frame"].to_numpy()):
                ax.barh(row, seg[1] - seg[0] + 1, left=seg[0] - 0.5, height=0.7, color=cmap(int(tid) % 20), edgecolor="white", lw=0.4)
                if seg[1] - seg[0] >= 3:
                    ax.text((seg[0] + seg[1]) / 2, row, str(int(tid)), ha="center", va="center", fontsize=6, color="black")
        c = cov[gid]
        ax.text(-2, row, f"gt{gid} {c['gt_class']}", ha="right", va="center", fontsize=7)
    ax.set_yticks([])
    ax.set_ylim(-0.7, len(order) - 0.3)
    ax.invert_yaxis()
    ax.set_xlim(-24, gts["frame"].max() + 2)                 # 왼쪽 여백 = GT 라벨 자리
    ax.set_xlabel("frame")
    ax.set_title(f"C tracked: track ID coverage per GT tracklet (grey = GT present, colored = matched track ID; "
                 f"IDSW={tr['idsw']}, frag={tr['frag']}, MT/PT/ML={tr['mt']}/{tr['pt']}/{tr['ml']}, threshold {result['settings']['match_distance_m']:.0f} m)",
                 fontsize=9)
    ax.grid(True, axis="x", alpha=0.3)
    fig.tight_layout()
    path = out / "05_track_gantt.png"
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def _segments(frames: np.ndarray) -> list[tuple[int, int]]:
    """정렬된 정수 프레임 배열을 연속 구간 [(start, end), ...] 로 자른다."""
    frames = np.asarray(sorted(set(int(f) for f in frames)))
    if frames.size == 0:
        return []
    cuts = np.flatnonzero(np.diff(frames) > 1)
    starts = np.concatenate([[frames[0]], frames[cuts + 1]])
    ends = np.concatenate([frames[cuts], [frames[-1]]])
    return list(zip(starts.tolist(), ends.tolist()))


def plot_runtime(runtimes: dict, det: dict | None, out: Path) -> Path:
    """⑥ 왼쪽: 변형별 단계 median ms 누적 막대 (+ YOLO 검출은 별도 측정값 인용, 빗금). 오른쪽: 프레임별 total 박스플롯."""
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(12, 4.4), gridspec_kw={"width_ratios": [1.2, 1]})
    names = [n for n, _, _, _ in VARIANTS if runtimes.get(n)]
    stage_colors = {"load": "#2a78d6", "projection": "#eb6834", "filters": "#1baf7a", "cluster": "#eda100",
                    "fusion": "#e87ba4", "tracking": "#4a3aa7"}
    bottoms = np.zeros(len(names))
    for st in STAGES:
        vals = np.array([runtimes[n]["summary"].get(st, {}).get("median_ms", 0.0) for n in names])
        if not vals.any():
            continue
        ax0.bar(names, vals, 0.55, bottom=bottoms, color=stage_colors[st], label=st, edgecolor="white", lw=0.6)
        bottoms += vals
    if det:
        ax0.bar(names, [det["median"]] * len(names), 0.55, bottom=bottoms, color="none", edgecolor="0.3", hatch="///",
                label=f"YOLO detection (phase4, {det['median']:.1f} ms median, measured separately)")
        bottoms += det["median"]
    for x, b in zip(names, bottoms):
        ax0.text(x, b + 0.5, f"{b:.1f} ms\n({1000 / b:.0f} FPS)", ha="center", va="bottom", fontsize=8)
    ax0.set_xticks(range(len(names)), [LABEL[n] for n in names])
    ax0.set_ylabel("median ms per frame")
    ax0.set_ylim(0, bottoms.max() * 1.75 if len(bottoms) else 1)        # 위쪽 여백에 범례를 두어 막대 라벨과 겹치지 않게
    ax0.set_title("Stage medians (stacked) + detection")
    ax0.legend(fontsize=7, loc="upper left", ncol=2)
    ax0.grid(True, axis="y", alpha=0.3)

    data = [[r["total"] for r in runtimes[n]["frames"].values()] for n in names]
    bp = ax1.boxplot(data, tick_labels=[LABEL[n] for n in names], showfliers=True, patch_artist=True,
                     flierprops={"marker": ".", "markersize": 3, "alpha": 0.4})
    for patch, n in zip(bp["boxes"], names):
        patch.set_facecolor(COLOR[n])
        patch.set_alpha(0.6)
    ax1.set_ylabel("pipeline total ms per frame (excl. detection)")
    ax1.set_title("Per-frame total (box = IQR, whiskers = 1.5 IQR)")
    ax1.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    path = out / "06_runtime.png"
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


# ----------------------------------------------------------------------------- 보고서
def fmt_signed(v: float) -> str:
    """부호를 붙인 포맷 (bias 용). nan → n/a."""
    return "n/a" if v is None or math.isnan(v) else f"{v:+.2f}"


def row_overall(name: str, o: dict) -> str:
    return (f"| {LABEL[name]} | {o['n_gt']} | {o['n_pred']} | {o['n_matched']} | {fmt(o['recall'])} | {fmt(o['precision'])} | {fmt(o['f1'])} "
            f"| {fmt(o['mota'])} | {fmt(o['bev_mean'])} | {fmt(o['bev_median'])} | {fmt(o['bev_p95'])} | {fmt(o['d3_mean'])} "
            f"| {fmt(o['depth_abs_mean'])} | {fmt_signed(o['depth_bias'])} | {fmt(o['iou_mean'])} | {fmt(o['iou_ge05_frac'])} |")


OVERALL_HEADER = ("| 변형 | GT | 예측 | 매칭(TP) | 재현율 | 정밀도 | F1 | MOTA | BEV mean | BEV median | BEV p95 | 3D mean "
                  "| 깊이 \\|오차\\| mean | 깊이 bias | BEV IoU mean | IoU≥0.5 비율 |\n"
                  "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")


def write_results_md(results: dict, cfg: dict, settings: dict, path: Path, plots: dict[str, Path], extra_c: dict) -> None:
    ds = cfg["dataset"]
    seq = f"{ds['date']}_drive_{ds['drive']}_sync"
    fr = settings["frames"]
    n_frames = (fr[1] - fr[0]) if fr else len(results[(VARIANTS[0][0], THRESHOLDS[0])]["per_frame"])
    L = [f"# Phase 8 정량 평가 결과 — 변형 A / B / C", "",
         f"- 시퀀스 `{seq}`, 프레임 {n_frames}장{'' if not fr else f' ([{fr[0]}, {fr[1]}))'}. 생성: `scripts/evaluate.py` (설정 `config_used.yaml`).",
         f"- **GT**: `tracklet_labels.xml` 중 카메라 수평 FOV({settings['gt_fov_deg']}°) 안, x ≤ {settings['gt_x_max_m']:.0f} m, 클래스 {settings['classes']}. "
         "중심 = 바닥중심 + h/2 (기하 중심). 이 시퀀스에 Pedestrian GT 는 0개 → 표에 n/a.",
         "- **예측**: A/B 는 `status == \"ok\"` 인 3D 객체, C 는 `confirmed` 트랙 (관측 없이 예측만 한 coasting 트랙 포함).",
         "- **매칭**: 프레임마다 BEV(x, y) 중심 거리로 헝가리안 1:1, 임계값 2 m (기본, `evaluation.match_distance_m`) 와 4 m. TP = 매칭 쌍, FP = 미매칭 예측, FN = 미매칭 GT.",
         "- **지표**: 재현율 = TP/GT, 정밀도 = TP/예측, MOTA = 1 − (FN + FP + IDSW)/GT (A/B 는 IDSW = 0). 오차는 매칭 쌍만, 단위 m. "
         "깊이 bias = mean(pred_x − gt_x), 음수 = GT 보다 가깝게 추정. BEV IoU 는 size 가 있는 B/C 만 (yaw = 0 AABB vs GT 회전 박스).",
         "- 좌표계: Velodyne (x 전방, y 좌, z 상).", ""]

    L += ["## 1. 전체 집계", ""]
    for k, thr in enumerate(THRESHOLDS, 1):
        L += [f"### 1.{k} 매칭 임계값 {thr:.1f} m", "", OVERALL_HEADER]
        for name, _, _, _ in VARIANTS:
            L.append(row_overall(name, results[(name, thr)]["overall"]))
        if thr == THRESHOLDS[0]:
            L.append(row_overall("C_tracked", extra_c["overall"]).replace(LABEL["C_tracked"], "C tracked, coasting 제외 (참고)"))
        L.append("")

    thr = THRESHOLDS[0]
    L += [f"## 2. 거리 구간별 (GT 의 BEV 거리 기준, 임계값 {thr:.1f} m)", "",
          "정밀도는 **예측 자신의 거리** 로 구간을 나눠 센다 (매칭 쌍은 GT 거리 기준이라 경계에서 구간이 다를 수 있음). BEV 거리가 70 m 를 넘는 GT 는 어느 구간에도 안 들어간다.", "",
          "| 구간 | 변형 | GT | 재현율 | 예측 | 정밀도 | BEV median | BEV p95 | 깊이 bias | IoU mean | IoU≥0.5 | IoU n |", "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for lo, hi in RANGE_BINS:
        key = f"{lo:.0f}-{hi:.0f}"
        for name, _, _, _ in VARIANTS:
            d = results[(name, thr)]["by_range"][key]
            L.append(f"| {key} m | {LABEL[name]} | {d['n_gt']} | {fmt(d['recall'])} | {d['n_pred']} | {fmt(d['precision'])} | {fmt(d['bev_median'])} "
                     f"| {fmt(d['bev_p95'])} | {fmt_signed(d['depth_bias'])} | {fmt(d['iou_mean'])} | {fmt(d['iou_ge05_frac'])} | {d['iou_n']} |")
    L.append("")

    L += [f"## 3. 클래스별 재현율 (GT 클래스 기준, 임계값 {thr:.1f} m)", "",
          "예측 클래스(COCO: car/truck/bus/person/bicycle) 와 GT 클래스(KITTI: Car/Van/Truck/Pedestrian/Cyclist) 가 1:1 이 아니므로 클래스별 정밀도는 정의하지 않는다. "
          "매칭은 클래스를 보지 않는다 (위치만).", "",
          "| 클래스 | GT | " + " | ".join(f"{LABEL[n]} 재현율" for n, _, _, _ in VARIANTS) + " | " + " | ".join(f"{LABEL[n]} BEV median" for n, _, _, _ in VARIANTS) + " |",
          "|---|---|" + "---|" * (2 * len(VARIANTS))]
    for cls in settings["classes"]:
        ds_ = [results[(n, thr)]["by_class"][cls] for n, _, _, _ in VARIANTS]
        L.append(f"| {cls} | {ds_[0]['n_gt']} | " + " | ".join(fmt(d["recall"]) for d in ds_) + " | " + " | ".join(fmt(d["bev_median"]) for d in ds_) + " |")
    L.append("")

    L += ["## 4. 추적 지표 (변형 C)", "",
          "| 임계값 | IDSW | 단절(frag) | 트랙 ID 수 | GT tracklet 수 | MT | PT | ML | 단일 ID 커버 평균 | 임의 ID 커버 평균 | coasting 예측 | coasting 매칭 |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for thr_ in THRESHOLDS:
        t = results[("C_tracked", thr_)]["tracking"]
        L.append(f"| {thr_:.1f} m | {t['idsw']} | {t['frag']} | {t['n_track_ids']} | {t['n_gt_tracklets']} | {t['mt']} | {t['pt']} | {t['ml']} "
                 f"| {fmt(t['mean_cov_single'])} | {fmt(t['mean_cov_any'])} | {t['n_coasting_pred']} | {t['n_coasting_matched']} |")
    L += ["",
          "- IDSW: 같은 GT 가 직전에 매칭됐던 트랙과 다른 track_id 에 매칭되면 1 (CLEAR MOT). 단절: GT 의 매칭 프레임 사이에 빈 구간이 생길 때마다 1.",
          "- 커버: GT tracklet 이 평가 GT 로 존재한 프레임 중 매칭된 프레임 비율. '단일 ID' 는 가장 오래 붙은 track_id 하나만 센다. MT ≥ 0.8, ML ≤ 0.2 (단일 ID 기준).",
          "- coasting 예측 = 그 프레임에 관측 없이 칼만 예측만으로 남아 있는 confirmed 트랙 (misses > 0). 이 중 GT 와 2 m 안에 든 것이 'coasting 매칭'.", ""]

    t = results[("C_tracked", thr)]["tracking"]
    L += [f"### GT tracklet 별 커버 (C, 임계값 {thr:.1f} m)", "",
          "| GT | 클래스 | 프레임 | 존재 | 매칭 | ID 수 | 주 track_id | 단일 ID 커버 | 판정 |", "|---|---|---|---|---|---|---|---|---|"]
    for gid in sorted(t["coverage"], key=lambda g: t["coverage"][g]["first_frame"]):
        c = t["coverage"][gid]
        verdict = "MT" if c["cov_single"] >= settings["mt_ratio"] else ("ML" if c["cov_single"] <= settings["ml_ratio"] else "PT")
        L.append(f"| gt{gid} | {c['gt_class']} | {c['first_frame']}–{c['last_frame']} | {c['n_present']} | {c['n_matched']} | {c['n_ids']} "
                 f"| {c['best_track_id'] if c['best_track_id'] >= 0 else '-'} | {c['cov_single']:.2f} | {verdict} |")
    L.append("")

    L += ["## 5. 무효이거나 주의가 필요한 비교", "",
          "1. **A 의 BEV IoU 는 계산 불가** — raw 융합은 크기(size)를 내지 않는다 (n/a). 박스 품질은 B/C 만 비교한다.",
          "2. **B/C 의 IoU 는 AABB(yaw = 0) 대 GT 회전 박스** — 클러스터 크기가 보이는 면만 반영해 GT 길이(4 m 대)보다 훨씬 작다 (팀 E: 35 m 이상은 크기 신뢰 불가). "
          "IoU 는 '중심이 맞아도 낮게' 나오므로 중심 오차와 함께 읽어야 한다. 거리 구간별 IoU 가 그 근거.",
          "3. **C 의 초반 프레임 재현율은 구조적으로 낮다** — 새 트랙은 `min_hits` = 3 프레임 동안 tentative 라 예측이 없다. 프레임별 매칭 수 그림(④)의 시작 구간과, 새 물체가 등장할 때마다 2 프레임씩 빠지는 것이 이 효과다.",
          "4. **C 의 정밀도가 B 보다 낮은 이유** — coasting 트랙(관측 없이 최대 `max_misses` = 3 프레임 예측)이 예측으로 들어간다. 위 '참고' 행(coasting 제외)이 그 크기를 보여준다. "
          "coasting 은 잠깐 가려진 물체를 잇기 위한 기능이므로 FP 로만 보면 안 되고, 반대로 정말 사라진 물체의 유령 예측도 섞여 있다.",
          "5. **GT 필터 밖의 물체는 평가하지 않는다** — 카메라 FOV 밖·70 m 밖 GT 는 카메라 기반 파이프라인이 볼 수 없으므로 뺐다. 반대로 tracklet 이 없는 주차 차량(팀 B 관찰)은 YOLO 가 검출해도 FP 로 세어져 정밀도가 실제보다 낮게 나온다.",
          "6. **Cyclist 재현율 ~0 은 검출기 한계** — 640 px 추론에서 bicycle 검출이 1건뿐 (팀 B). 융합·추적의 문제가 아니다. Pedestrian GT 는 0개라 사람 클래스는 평가 불가.",
          "7. **매칭 임계값이 오차의 상한** — 2 m 표의 BEV 오차는 정의상 2 m 를 넘지 않는다. A 는 4 m 로 넓히면 재현율 0.45 → 0.70 으로 뛰지만 오차가 1.75 m 로 커진다: A 의 '미검출' 상당수는 실제로는 1.3 m 표면 편향 때문에 게이트 밖으로 밀린 것이다.",
          "8. **클래스를 무시한 매칭** — car 검출이 Van/Truck GT 에 붙어도 TP 다. 클래스 정확도는 별도 지표가 아니다.",
          "9. **런타임은 부하가 있는 공용 PC 에서 측정** — `runtime.md` 의 load average 와 함께 읽을 것. median 만 대표값으로 쓴다.",
          "", "## 6. 플롯", ""]
    for k, p in plots.items():
        L.append(f"- `{p.relative_to(path.parent)}` — {k}")
    L.append("")
    path.write_text("\n".join(L), encoding="utf-8")


def hardware_info() -> dict:
    info = {"cpu": "unknown", "physical_cores": None, "logical_cores": os.cpu_count(), "ram_gb": None,
            "os": platform.platform(), "python": platform.python_version()}
    try:
        out = subprocess.run(["lscpu"], capture_output=True, text=True, check=True).stdout
        for line in out.splitlines():
            if line.startswith("Model name"):
                info["cpu"] = line.split(":", 1)[1].strip()
            if line.startswith("Core(s) per socket"):
                info["physical_cores"] = int(line.split(":", 1)[1])
    except Exception:
        pass
    try:
        with open("/proc/meminfo") as fp:
            for line in fp:
                if line.startswith("MemTotal"):
                    info["ram_gb"] = round(int(line.split()[1]) / 1024 / 1024, 1)
    except Exception:
        pass
    for mod in ("numpy", "scipy", "open3d", "torch", "ultralytics", "pandas"):
        try:
            info[mod] = __import__(mod).__version__
        except Exception:
            info[mod] = "n/a"
    return info


def write_runtime_md(runtimes: dict, det: dict | None, cfg: dict, path: Path, plot: Path) -> dict:
    hw = hardware_info()
    load_now = os.getloadavg()
    ds = cfg["dataset"]
    seq = f"{ds['date']}_drive_{ds['drive']}_sync"
    names = [n for n, _, _, _ in VARIANTS if runtimes.get(n)]
    L = ["# Phase 8 런타임 보고", "",
         "## 환경", "",
         f"- CPU: {hw['cpu']} (물리 코어 {hw['physical_cores']}, 논리 {hw['logical_cores']}), RAM {hw['ram_gb']} GB, {hw['os']}",
         f"- Python {hw['python']}, numpy {hw['numpy']}, scipy {hw['scipy']}, open3d {hw['open3d']}, pandas {hw['pandas']}, torch {hw['torch']}, ultralytics {hw['ultralytics']}",
         f"- 시퀀스 `{seq}`, image_02 1242×375, LiDAR 프레임당 약 12.4만 점",
         f"- 평가 스크립트 실행 시 load average (1/5/15분): {load_now[0]:.2f} / {load_now[1]:.2f} / {load_now[2]:.2f}", ""]
    L += ["## 측정 방법", "",
          "- 파이프라인 단계(load/projection/filters/cluster/fusion/tracking/total): `scripts/run_pipeline.py` 가 프레임마다 `time.perf_counter()` 로 단계 경과를 재고 "
          "`outputs/phase{5,6,7}/runtime.json` 에 저장. 단일 프로세스, 캐시 워밍 후, 시각화 저장 시간 제외. total 은 load 부터 tracking 까지의 합.",
          "- YOLO 2D 검출은 파이프라인이 `outputs/phase4/detections.json` 을 읽으므로 여기서 다시 재지 않고 **팀 B 의 `outputs/phase4/runtime.md`** 수치를 인용한다 "
          "(yolov8n, imgsz 640, CPU, warm-up 2회 후 297 프레임, letterbox + 순전파 + NMS).",
          "- 이 PC 는 다른 프로세스(사용자 실험, ROS2 노드)와 공유되어 mean/p95 에 간섭이 섞인다. **median 을 대표값**으로 쓴다. 각 변형의 측정 시각과 그때의 load average 는 아래 표.", ""]
    L += ["## 측정 조건 (변형별)", "", "| 변형 | 측정 시각 | 프레임 수 | 벽시계 | load avg 1/5/15 (측정 시작 시) |", "|---|---|---|---|---|"]
    for n in names:
        m = runtimes[n]["meta"]
        la = m.get("load_avg_1_5_15")
        la_s = "기록 없음" if la is None else " / ".join(f"{v:.2f}" for v in la)
        L.append(f"| {LABEL[n]} | {runtimes[n]['_mtime']} | {m['n_frames']} | {m.get('wall_time_s', float('nan')):.1f} s | {la_s} |")
    L.append("")

    L += ["## 단계별 시간 (ms/프레임)", "", "| 변형 | 단계 | n | mean | median | p95 | FPS (median 기준) |", "|---|---|---|---|---|---|---|"]
    if det:
        L.append(f"| (공통) | detection (YOLO, phase4 인용) | 297 | {det['mean']:.2f} | {det['median']:.2f} | {det['p95']:.2f} | {1000 / det['median']:.1f} |")
    e2e = {}
    for n in names:
        s = runtimes[n]["summary"]
        for st in STAGES + ["total"]:
            if st in s:
                v = s[st]
                L.append(f"| {LABEL[n]} | {st} | {v['n']} | {v['mean_ms']:.2f} | {v['median_ms']:.2f} | {v['p95_ms']:.2f} | {1000 / v['median_ms']:.1f} |")
        tot = s["total"]["median_ms"]
        e2e[n] = tot + (det["median"] if det else 0.0)
    L.append("")
    L += ["## 변형별 end-to-end 추정 (median 합, 검출 포함)", "", "| 변형 | 파이프라인 median | + 검출 median | = end-to-end | FPS |", "|---|---|---|---|---|"]
    for n in names:
        tot = runtimes[n]["summary"]["total"]["median_ms"]
        L.append(f"| {LABEL[n]} | {tot:.1f} ms | {det['median'] if det else 0:.1f} ms | {e2e[n]:.1f} ms | {1000 / e2e[n]:.1f} |")
    L += ["", "end-to-end 는 각 단계 median 의 합이므로 실제 한 프로세스의 median 과는 다를 수 있다 (단계별 최악 프레임이 겹치지 않으면 낮게, 겹치면 높게).", ""]
    ref = "B_clustered" if "B_clustered" in runtimes else names[-1]
    proj = runtimes[ref]["summary"].get("projection", {}).get("median_ms", float("nan"))
    L += ["## C++ 대체 시 예상 (팀 I, `outputs/phase10/benchmark.md` 인용)", "",
          f"- 투영(projection) median: Python 3.72 ms → C++ 0.38 ms (9.7배, 20 프레임 × 30회, 부하 중 측정). {LABEL[ref]} 의 projection median {proj:.2f} ms 를 0.4 ms 로 바꾸면 "
          f"프레임당 약 {max(proj - 0.4, 0):.1f} ms 절감 — {LABEL[ref]} total median 의 {100 * max(proj - 0.4, 0) / runtimes[ref]['summary']['total']['median_ms']:.0f}% 수준. "
          "지면 제거(RANSAC)·DBSCAN·PNG 디코딩이 남은 큰 항목이며 이미 C/C++ 커널이라 Python 오버헤드 제거 효과는 제한적.", ""]
    L += ["## 플롯", "", f"- `{plot.relative_to(path.parent)}` — 단계별 median 누적 막대 + 프레임별 total 박스플롯", ""]
    path.write_text("\n".join(L), encoding="utf-8")
    return {"hardware": hw, "load_avg_at_eval": list(load_now), "end_to_end_median_ms": e2e}


def strip_frames(result: dict) -> dict:
    """JSON 저장용: DataFrame 을 뺀 결과."""
    return {k: v for k, v in result.items() if k not in ("per_frame", "pairs", "preds", "gts")}


# ----------------------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(DEFAULT_CONFIG))
    ap.add_argument("--out", default=None, help="출력 디렉터리 (기본 outputs/phase8)")
    ap.add_argument("--frames", nargs=2, type=int, metavar=("START", "END"), help="[START, END) 구간만 평가 (기본 설정 파일의 evaluation.frames)")
    args = ap.parse_args()

    cfg = load_config(args.config)
    out_root = Path(cfg["outputs"]["dir"])
    out = Path(args.out) if args.out else out_root / "phase8"
    plots_dir = out / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy(args.config, out / "config_used.yaml")
    frames = list(range(args.frames[0], args.frames[1])) if args.frames else None

    gts = gt_boxes_by_frame(cfg["dataset"]["tracklets"])
    preds = {name: load_variant(out_root, rel, kind) for name, rel, kind, _ in VARIANTS}

    results = {}
    for name, _, _, _ in VARIANTS:
        for thr in THRESHOLDS:
            results[(name, thr)] = evaluate_variant(preds[name], gts, cfg, name, max_dist=thr, frames=frames)
    extra_c = evaluate_variant(preds["C_tracked"], gts, cfg, "C_tracked_no_coasting", max_dist=THRESHOLDS[0], frames=frames, include_coasting=False)
    settings = results[(VARIANTS[0][0], THRESHOLDS[0])]["settings"]

    # 런타임
    runtimes = {}
    for name, _, _, rt in VARIANTS:
        d = load_runtime(out_root / rt)
        if d:
            stage_times = {st: [r[st] for r in d["frames"].values() if st in r] for st in STAGES + ["total"]}
            d["summary"] = summarize_runtime(stage_times)
            runtimes[name] = d
    det = detection_runtime(out_root / "phase4" / "runtime.md")

    # 플롯
    plots = {
        "BEV 오차 CDF (변형별, 2 m / 4 m)": plot_error_cdf(results, plots_dir),
        "GT 거리 vs 오차 산점 + 거리 구간별 재현율": plot_range(results, plots_dir, THRESHOLDS[0]),
        "재현율·정밀도·F1 막대 (2 m / 4 m)": plot_precision_recall(results, plots_dir),
        "프레임별 매칭 수 시계열 (A/B/C vs GT 수)": plot_per_frame(results, plots_dir, THRESHOLDS[0]),
        "C: GT tracklet 별 track ID 커버 간트차트": plot_gantt(results[("C_tracked", THRESHOLDS[0])], plots_dir),
    }
    if runtimes:
        plots["런타임 단계별 누적 막대 + total 박스플롯"] = plot_runtime(runtimes, det, plots_dir)

    # 표·CSV·JSON
    long = results_to_long(list(results.values()) + [extra_c])
    long.to_csv(out / "results.csv", index=False)
    write_results_md(results, cfg, settings, out / "results.md", plots, extra_c)
    rt_meta = write_runtime_md(runtimes, det, cfg, out / "runtime.md", plots.get("런타임 단계별 누적 막대 + total 박스플롯", plots_dir / "06_runtime.png")) if runtimes else {}
    payload = {
        "generated": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "config": str(Path(args.config).resolve()),
        "settings": settings,
        "variants": {f"{name}@{thr}m": strip_frames(r) for (name, thr), r in results.items()},
        "C_tracked_no_coasting@2m": strip_frames(extra_c),
        "runtime": {n: {"meta": d["meta"], "summary": d["summary"]} for n, d in runtimes.items()},
        "detection_runtime": det,
        **rt_meta,
    }
    with open(out / "results.json", "w", encoding="utf-8") as fp:
        json.dump(payload, fp, indent=1, ensure_ascii=False, default=lambda o: float(o) if isinstance(o, np.floating) else int(o))

    # 콘솔 요약
    print(f"GT 필터: FOV {settings['gt_fov_deg']}°, x ≤ {settings['gt_x_max_m']} m, 클래스 {settings['classes']}, 프레임 {settings['frames'] or '전체'}")
    print(f"{'variant':<14}{'thr':>5}{'GT':>6}{'pred':>6}{'TP':>6}{'recall':>8}{'prec':>7}{'F1':>6}{'MOTA':>7}{'BEV med':>9}{'IoU':>6}{'IDSW':>6}")
    for (name, thr), r in results.items():
        o, t = r["overall"], r["tracking"]
        print(f"{name:<14}{thr:>5.1f}{o['n_gt']:>6}{o['n_pred']:>6}{o['n_matched']:>6}{o['recall']:>8.3f}{o['precision']:>7.3f}{o['f1']:>6.2f}{o['mota']:>7.2f}"
              f"{o['bev_median']:>9.2f}{fmt(o['iou_mean']):>6}{(t['idsw'] if t else '-'):>6}")
    print(f"saved to {out}: results.csv, results.md, results.json, runtime.md, config_used.yaml, plots/ ({len(plots)} files)")


if __name__ == "__main__":
    main()
