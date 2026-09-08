"""Phase 4: 시퀀스 전체에 YOLO 2D 검출을 돌리고 결과·런타임을 outputs/phase4/ 에 저장한다.

실행: .venv/bin/python scripts/run_detection.py [--frames START END]
산출: outputs/phase4/detections.json, outputs/phase4/runtime.md
"""
from __future__ import annotations

import argparse
import collections
import os
import platform
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
import ultralytics
from tqdm import tqdm

from perceptrack3d.config import DEFAULT_CONFIG, load_config
from perceptrack3d.data.kitti_loader import KittiDataset
from perceptrack3d.detection.detector import YoloDetector, save_detections_json


def cpu_model_name() -> str:
    try:
        out = subprocess.run(["lscpu"], capture_output=True, text=True, check=True).stdout
        for line in out.splitlines():
            if line.startswith("Model name"):
                return line.split(":", 1)[1].strip()
    except Exception:
        pass
    return platform.processor() or "unknown"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(DEFAULT_CONFIG))
    ap.add_argument("--frames", nargs=2, type=int, metavar=("START", "END"), help="[START, END) 구간만 실행")
    args = ap.parse_args()

    cfg = load_config(args.config)
    ds = KittiDataset(cfg)
    frame_ids = ds.frame_ids()
    if args.frames:
        frame_ids = [f for f in frame_ids if args.frames[0] <= f < args.frames[1]]
    out_dir = Path(cfg["outputs"]["dir"]) / "phase4"
    out_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()
    det = YoloDetector(cfg)
    load_s = time.perf_counter() - t0

    dets_by_frame, ms_list = {}, []
    for fid in tqdm(frame_ids, desc="detect"):
        img = ds.load_frame(fid)["image"]
        dets, ms = det.detect_timed(img, fid)
        dets_by_frame[fid] = dets
        ms_list.append(ms)

    json_path = out_dir / "detections.json"
    save_detections_json(dets_by_frame, json_path)

    ms = np.asarray(ms_list)
    all_dets = [d for v in dets_by_frame.values() for d in v]
    per_class = collections.Counter(d.class_name for d in all_dets)
    empty_frames = sum(1 for v in dets_by_frame.values() if not v)
    h, w = img.shape[:2]

    lines = [
        "# Phase 4 런타임 (2D 검출)",
        "",
        "## 환경",
        f"- CPU: {cpu_model_name()} (논리 코어 {os.cpu_count()}, torch threads {torch.get_num_threads()})",
        f"- torch {torch.__version__}, ultralytics {ultralytics.__version__}, device={det.device}",
        f"- 모델: {det.model_path.name}, imgsz={det.imgsz}, conf={det.conf}, iou={det.iou}",
        f"- 시퀀스: {cfg['dataset']['date']}_drive_{cfg['dataset']['drive']}_sync, {cfg['dataset']['camera']}, 프레임 {len(frame_ids)}장 ({frame_ids[0]}..{frame_ids[-1]}), 원본 {w}x{h}",
        "",
        "## 측정 방법",
        "- 생성자에서 더미 이미지 2회 warm-up 후 측정 (warm-up 은 통계에서 제외).",
        "- 프레임당 `time.perf_counter()` 로 ultralytics predict 1회(letterbox 전처리 + 순전파 + NMS) 를 측정.",
        "- 디스크 로드(cv2.imread)와 Detection2D 변환은 제외.",
        "",
        "## 결과",
        f"- 모델 로드 + warm-up: {load_s:.2f} s",
        f"- 프레임당 추론 (ms): 평균 {ms.mean():.2f}, 중앙값 {np.median(ms):.2f}, p95 {np.percentile(ms, 95):.2f}, 최소 {ms.min():.2f}, 최대 {ms.max():.2f}",
        f"- 첫 프레임: {ms[0]:.2f} ms",
        f"- FPS (중앙값 기준): {1000.0 / np.median(ms):.1f}",
        "",
        "## 검출 통계",
        f"- 총 검출 수: {len(all_dets)} (프레임당 평균 {len(all_dets) / len(frame_ids):.2f})",
        f"- 검출 0개 프레임: {empty_frames}",
        "- 클래스별: " + ", ".join(f"{k} {v}" for k, v in per_class.most_common()),
        "",
        f"산출물: `{json_path.relative_to(cfg['outputs']['dir'])}`",
    ]
    (out_dir / "runtime.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
