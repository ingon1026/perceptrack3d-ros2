"""tracklet_labels.xml 좌표 규약 검증 스크립트 (Phase 8).

프레임 0, 50, 150 의 GT 박스를 LiDAR BEV 에 그려 outputs/phase8/gt_bev_frame{N}.png 로 저장하고,
각 GT 박스 안에 들어가는 LiDAR 점 수를 출력한다. 규약(바닥 중심, l↔x, w↔y, rz=yaw) 이 맞으면
차량 박스 안에 보통 수십~수백 개의 점이 들어간다. 0 에 가까우면 규약이 틀린 것이다.

실행: .venv/bin/python scripts/check_tracklets.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from perceptrack3d.config import load_config
from perceptrack3d.data.kitti_loader import KittiDataset, load_velodyne
from perceptrack3d.evaluation.tracklets import gt_boxes_by_frame, gt_in_front_fov, points_in_box_mask
from perceptrack3d.visualization.gt import plot_bev_gt

FRAMES = [0, 50, 150]


def main() -> None:
    cfg = load_config()
    ds = KittiDataset(cfg)
    gts = gt_boxes_by_frame(ds.tracklets_path)
    out_dir = Path(cfg["outputs"]["dir"]) / "phase8"
    out_dir.mkdir(parents=True, exist_ok=True)

    counts = np.array([len(gts.get(f, [])) for f in ds.frame_ids()])
    print(f"tracklet XML: {ds.tracklets_path}")
    print(f"프레임 수 {len(counts)}, 프레임별 GT 수: min {counts.min()}  mean {counts.mean():.2f}  max {counts.max()}  "
          f"(GT 0개 프레임 {(counts == 0).sum()}개)")

    for f in FRAMES:
        pts = load_velodyne(ds.velodyne_path(f))
        boxes = gts.get(f, [])
        in_fov = {b.tracklet_id for b in gt_in_front_fov(boxes)}
        print(f"\n=== frame {f}: GT {len(boxes)}개 (카메라 FOV 근사 안 {len(in_fov)}개), LiDAR 점 {len(pts)} ===")
        print(f"{'id':>3} {'type':<8} {'x':>7} {'y':>7} {'z':>6} {'l':>5} {'w':>5} {'h':>5} {'yaw':>6} {'pts':>5} fov")
        for b in boxes:
            n = int(points_in_box_mask(pts, b.center, b.size, b.yaw).sum())
            c, s = b.center, b.size
            print(f"{b.tracklet_id:>3} {b.object_type:<8} {c[0]:7.2f} {c[1]:7.2f} {c[2]:6.2f} "
                  f"{s[0]:5.2f} {s[1]:5.2f} {s[2]:5.2f} {b.yaw:6.2f} {n:5d} {'o' if b.tracklet_id in in_fov else '-'}")
        path = plot_bev_gt(pts, boxes, out_dir / f"gt_bev_frame{f}.png",
                           title=f"frame {f}: {len(boxes)} GT boxes on LiDAR BEV")
        print(f"saved {path}")


if __name__ == "__main__":
    main()
