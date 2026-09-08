"""evaluation/tracklets.py 테스트: 실제 KITTI XML (있을 때) + 합성 XML."""
from __future__ import annotations

import math
from collections import Counter
from pathlib import Path

import numpy as np
import pytest

from perceptrack3d.config import load_config
from perceptrack3d.evaluation.tracklets import (
    box_corners_bev,
    gt_boxes_by_frame,
    gt_in_front_fov,
    load_tracklets,
    points_in_box_mask,
)

# ---------- 합성 XML (tracklet 2개) ----------
_POSE = """
      <item>
        <tx>{tx}</tx><ty>{ty}</ty><tz>{tz}</tz><rx>0</rx><ry>0</ry><rz>{rz}</rz>
        <state>2</state><occlusion>{occ}</occlusion><occlusion_kf>0</occlusion_kf><truncation>{trunc}</truncation>
        <amt_occlusion>0</amt_occlusion><amt_occlusion_kf>0</amt_occlusion_kf>
        <amt_border_l>0</amt_border_l><amt_border_r>0</amt_border_r><amt_border_kf>-1</amt_border_kf>
      </item>"""


def _tracklet(obj, h, w, l, first, poses):
    body = "".join(_POSE.format(**p) for p in poses)
    return f"""
    <item class_id="1" tracking_level="0" version="1">
      <objectType>{obj}</objectType><h>{h}</h><w>{w}</w><l>{l}</l><first_frame>{first}</first_frame>
      <poses class_id="2" tracking_level="0" version="0"><count>{len(poses)}</count><item_version>2</item_version>{body}
      </poses><finished>1</finished>
    </item>"""


def _xml(count, tracklets_xml):
    return f"""<?xml version="1.0" encoding="UTF-8" standalone="yes" ?>
<!DOCTYPE boost_serialization>
<boost_serialization signature="serialization::archive" version="9">
<tracklets class_id="0" tracking_level="0" version="0">
  <count>{count}</count><item_version>1</item_version>{tracklets_xml}
</tracklets>
</boost_serialization>
"""


SYNTH = _xml(2,
    _tracklet("Car", 1.5, 1.6, 4.0, 3, [dict(tx=10.0, ty=2.0, tz=-1.5, rz=0.1, occ=0, trunc=0),
                                        dict(tx=11.0, ty=2.1, tz=-1.5, rz=0.2, occ=1, trunc=1)])
    + _tracklet("Pedestrian", 1.8, 0.6, 0.8, 7, [dict(tx=5.0, ty=-1.0, tz=-1.7, rz=math.pi + 0.5, occ=2, trunc=0)]))


@pytest.fixture
def synth_xml(tmp_path: Path) -> Path:
    p = tmp_path / "tracklet_labels.xml"
    p.write_text(SYNTH, encoding="utf-8")
    return p


def test_synthetic_parse_exact(synth_xml):
    ts = load_tracklets(synth_xml)
    assert len(ts) == 2
    car, ped = ts
    assert car["tracklet_id"] == 0 and car["object_type"] == "Car"
    assert (car["h"], car["w"], car["l"], car["first_frame"]) == (1.5, 1.6, 4.0, 3)
    assert len(car["poses"]) == 2
    assert car["poses"][0] == dict(tx=10.0, ty=2.0, tz=-1.5, rx=0.0, ry=0.0, rz=0.1, state=2, occlusion=0, truncation=0)
    assert car["poses"][1]["occlusion"] == 1 and car["poses"][1]["truncation"] == 1
    assert ped["object_type"] == "Pedestrian" and ped["first_frame"] == 7 and len(ped["poses"]) == 1


def test_synthetic_boxes_by_frame(synth_xml):
    by_frame = gt_boxes_by_frame(synth_xml)
    assert sorted(by_frame) == [3, 4, 7]                       # first_frame + pose index
    b = by_frame[3][0]
    np.testing.assert_allclose(b.center, [10.0, 2.0, -1.5 + 0.75])   # 바닥 중심 + h/2
    np.testing.assert_allclose(b.size, [4.0, 1.6, 1.5])              # [l, w, h]
    assert b.yaw == pytest.approx(0.1) and b.tracklet_id == 0
    assert by_frame[4][0].truncation == 1 and by_frame[4][0].occlusion == 1
    ped = by_frame[7][0]
    assert ped.yaw == pytest.approx(0.5 - math.pi)                   # [-pi, pi] 로 감쌈
    assert ped.occlusion == 2


def test_count_mismatch_raises(tmp_path):
    p = tmp_path / "bad.xml"
    p.write_text(_xml(3, _tracklet("Car", 1, 1, 1, 0, [dict(tx=0, ty=0, tz=0, rz=0, occ=0, trunc=0)])), encoding="utf-8")
    with pytest.raises(ValueError, match="count"):
        load_tracklets(p)


def test_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_tracklets(tmp_path / "nope.xml")


# ---------- 기하 유틸 ----------
def test_box_corners_bev_axis_aligned_and_rotated():
    c = box_corners_bev(np.array([1.0, 2.0, 0.0]), np.array([4.0, 2.0, 1.5]), 0.0)
    np.testing.assert_allclose(c, [[3, 3], [3, 1], [-1, 1], [-1, 3]])   # 앞좌, 앞우, 뒤우, 뒤좌
    c90 = box_corners_bev([0.0, 0.0], [4.0, 2.0], math.pi / 2)          # l 이 y 축을 향함
    np.testing.assert_allclose(c90, [[-1, 2], [1, 2], [1, -2], [-1, -2]], atol=1e-12)


def test_points_in_box_mask_rotated():
    pts = np.array([[0.0, 0.0, 0.0], [1.9, 0.0, 0.0], [0.0, 1.9, 0.0], [0.0, 0.0, 0.9], [2.1, 0, 0]])
    m0 = points_in_box_mask(pts, [0, 0, 0], [4.0, 2.0, 1.5], 0.0)
    assert m0.tolist() == [True, True, False, False, False]            # |x|<=2, |y|<=1, |z|<=0.75
    m90 = points_in_box_mask(pts, [0, 0, 0], [4.0, 2.0, 1.5], math.pi / 2)
    assert m90.tolist() == [True, False, True, False, False]


def test_gt_in_front_fov(synth_xml):
    boxes = gt_boxes_by_frame(synth_xml)[3] + gt_boxes_by_frame(synth_xml)[7]
    assert len(gt_in_front_fov(boxes)) == 2                            # 둘 다 전방 좁은 각
    boxes[0].center[:] = [-5.0, 0.0, 0.0]                              # 뒤쪽
    boxes[1].center[:] = [5.0, 5.0, 0.0]                               # 45° > 40.7°
    assert gt_in_front_fov(boxes) == []


# ---------- 실제 KITTI 데이터 ----------
def _real_xml() -> Path | None:
    p = Path(load_config()["dataset"]["tracklets"])
    return p if p.is_file() else None


needs_data = pytest.mark.skipif(_real_xml() is None, reason="KITTI tracklet_labels.xml 이 없음")


@needs_data
def test_real_tracklet_counts():
    ts = load_tracklets(_real_xml())
    assert len(ts) == 36
    assert Counter(t["object_type"] for t in ts) == {"Car": 33, "Van": 1, "Truck": 1, "Cyclist": 1}


@needs_data
def test_real_frames_and_centers():
    ts = load_tracklets(_real_xml())
    by_frame = gt_boxes_by_frame(_real_xml())
    assert len(by_frame[0]) > 0
    assert min(by_frame) >= 0 and max(by_frame) <= 296
    assert sum(len(v) for v in by_frame.values()) == sum(len(t["poses"]) for t in ts)
    for t in ts:                                                       # center.z = tz + h/2, 모든 pose
        for i, p in enumerate(t["poses"]):
            box = next(b for b in by_frame[t["first_frame"] + i] if b.tracklet_id == t["tracklet_id"])
            assert box.center[2] == pytest.approx(p["tz"] + t["h"] / 2)
            assert box.size[2] == pytest.approx(t["h"]) and box.size[0] == pytest.approx(t["l"])
