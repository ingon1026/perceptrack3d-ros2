"""Phase 7 합성 데모: 실제 융합 결과 없이 KalmanTracker 의 동작을 검증한다.

시나리오 (Velodyne 프레임, 40 프레임, dt = 0.1 s)
    gt0 차량: 오른쪽 차선, 전방으로 멀어짐         (8, -3.5)  → v = (+6, 0)
    gt1 차량: 왼쪽 차선, 마주 옴                    (30, 3.5)  → v = (-5, 0)
    gt2 차량: 같은 차선 앞, 느리게 멀어짐           (15, 0)    → v = (+2, 0)
    gt3 차량: 오른쪽에서 왼쪽으로 교차              (22, -12)  → v = (0, +6)  (프레임 20 근처 gt2 와 ~3 m)
    관측 잡음 σ = 0.3 m, gt1 은 프레임 15–17 미검출(3 프레임), gt3 은 25–26 미검출(2 프레임)
    가짜 관측: 프레임 10 에 1개, 프레임 22–23 에 같은 자리 2 프레임 연속 1개 (min_hits=3 이므로 확정되면 안 됨)
출력: outputs/phase7/synthetic_bev_tracks.png, 콘솔에 트랙 수·ID 스위치 수
실행: .venv/bin/python scripts/demo_tracking_synthetic.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from perceptrack3d.config import load_config
from perceptrack3d.tracking.kalman_tracker import KalmanTracker
from perceptrack3d.types import Object3D
from perceptrack3d.visualization.tracks import plot_bev_tracks

N_FRAMES = 40
NOISE_STD = 0.3
MATCH_DIST = 1.5      # 평가용: 확정 트랙 ↔ GT 매칭 거리 (m)

GT_INIT = np.array([[8.0, -3.5, 0.8], [30.0, 3.5, 0.8], [15.0, 0.0, 0.8], [22.0, -12.0, 0.8]])
GT_VEL = np.array([[6.0, 0.0, 0.0], [-5.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 6.0, 0.0]])
MISSED = {1: range(15, 18), 3: range(25, 27)}
CLUTTER = {10: [[40.0, -8.0, 0.8]], 22: [[5.0, 10.0, 0.8]], 23: [[5.0, 10.0, 0.8]]}


def make_object(center, frame_id) -> Object3D:
    return Object3D(frame_id=frame_id, class_name="car", confidence=0.9,
                    xyxy=np.zeros(4, dtype=np.float32), center=np.asarray(center, float),
                    size=np.array([4.0, 1.8, 1.5]), status="ok")


def main() -> None:
    cfg = load_config()
    dt = cfg["tracking"]["dt"]
    rng = np.random.default_rng(7)
    tracker = KalmanTracker(cfg)

    gt_traj = {g: [] for g in range(len(GT_INIT))}
    track_traj: dict[int, list] = {}
    assigned: dict[int, list] = {g: [] for g in gt_traj}      # gt → 프레임별 매칭된 track_id
    n_obs_total = 0

    for f in range(N_FRAMES):
        gt_now = GT_INIT + GT_VEL * dt * f                     # (4, 3)
        for g in gt_traj:
            gt_traj[g].append(gt_now[g])
        objects = []
        for g in gt_traj:
            if f in MISSED.get(g, ()):
                continue
            objects.append(make_object(gt_now[g] + rng.normal(0, NOISE_STD, 3), f))
        for c in CLUTTER.get(f, []):
            objects.append(make_object(np.asarray(c) + rng.normal(0, NOISE_STD, 3), f))
        n_obs_total += len(objects)

        tracks = tracker.step(objects, f)
        for tr in tracks:
            track_traj.setdefault(tr.track_id, []).append(tr.position.copy())
        # 평가: 각 GT 에 가장 가까운 확정 트랙 (MATCH_DIST 안) 의 id 기록
        for g in gt_traj:
            best, best_d = None, MATCH_DIST
            for tr in tracks:
                d = float(np.linalg.norm(tr.position - gt_now[g]))
                if d < best_d:
                    best, best_d = tr.track_id, d
            assigned[g].append(best)

    # ID 스위치: 같은 GT 에 붙은 id 가 (None 제외) 바뀐 횟수
    id_switches = 0
    for g, ids in assigned.items():
        seq = [i for i in ids if i is not None]
        id_switches += sum(1 for a, b in zip(seq, seq[1:]) if a != b)
    confirmed_ids = sorted(track_traj)
    matched_ids = {i for ids in assigned.values() for i in ids if i is not None}
    ghost_ids = [i for i in confirmed_ids if i not in matched_ids]
    covered = {g: sum(i is not None for i in ids) for g, ids in assigned.items()}

    out_path = Path(cfg["outputs"]["dir"]) / "phase7" / "synthetic_bev_tracks.png"
    plot_bev_tracks({k: np.array(v) for k, v in track_traj.items()}, out_path,
                    gt={g: np.array(v) for g, v in gt_traj.items()},
                    title="Synthetic tracking demo (4 objects, 40 frames)")

    print(f"frames={N_FRAMES}, dt={dt}, observations={n_obs_total}, gt objects={len(GT_INIT)}")
    print(f"track ids created (tentative 포함) = {tracker._next_id - 1}")
    print(f"confirmed track ids = {confirmed_ids}  (n={len(confirmed_ids)})")
    print(f"ghost confirmed tracks (GT 와 매칭 안 됨) = {ghost_ids}")
    print(f"id switches = {id_switches}")
    print("frames covered per gt (of 40): " + ", ".join(f"gt{g}={c}" for g, c in covered.items()))
    for g, ids in assigned.items():
        first = next((f for f, i in enumerate(ids) if i is not None), None)
        print(f"  gt{g}: first confirmed frame={first}, ids={sorted({i for i in ids if i is not None})}")
    print(f"saved: {out_path}")


if __name__ == "__main__":
    main()
