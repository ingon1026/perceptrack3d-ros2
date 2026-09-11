"""LiDAR 3D 뷰 — 자율주행 데모에서 보는 "자차 뒤 위에서 앞을 내려다본" 점군 화면.

Open3D 의 대화형 창(GLFW/GLEW)은 Wayland + NVIDIA 조합에서 죽으므로, EGL 오프스크린 렌더러로 프레임을 이미지로
그린 뒤 OpenCV 창/동영상으로 보여 준다. 마우스 회전은 없고 카메라는 고정(아래 CAMERA 값을 바꾸면 시점이 바뀐다).

입력: points_velo (N, 3|4) float32 Velodyne, boxes = [(center (3,), size (3,) [l,w,h], yaw, rgb (3,) 0~1), ...]
출력: (H, W, 3) uint8 BGR
"""
from __future__ import annotations

import cv2
import matplotlib
import numpy as np
import open3d as o3d
from open3d.visualization import rendering as R

from perceptrack3d.evaluation.tracklets import box_corners_bev

# 카메라: 시야각(deg), 바라보는 점(lookat), 눈 위치(eye), 위 방향(up) — 모두 Velodyne 좌표(m)
CAMERA = dict(fov=55.0, lookat=[18.0, 0.0, 0.0], eye=[-14.0, 0.0, 9.0], up=[0.0, 0.0, 1.0])
Z_COLOR_RANGE = (-2.0, 1.5)      # 이 높이 구간을 turbo 컬러맵에 대응 (지면 ≈ −1.7 m 가 파랑, 나무·건물 위쪽이 빨강)
EGO_LWH = (4.0, 1.8, 1.5)        # 자차 상자 (센서 원점이 지붕 위 1.73 m 이므로 아래로 내려 그린다)


def box_lines(center: np.ndarray, size: np.ndarray, yaw: float) -> tuple[np.ndarray, np.ndarray]:
    """3D 박스 8 모서리 (8,3) 와 12 변 인덱스 (12,2). 바닥 4점 = box_corners_bev, 윗면은 +h."""
    c4 = box_corners_bev(center, size, yaw)
    z0, z1 = center[2] - size[2] / 2, center[2] + size[2] / 2
    pts = np.array([[x, y, z0] for x, y in c4] + [[x, y, z1] for x, y in c4])
    edges = np.array([[i, (i + 1) % 4] for i in range(4)] + [[4 + i, 4 + (i + 1) % 4] for i in range(4)]
                     + [[i, 4 + i] for i in range(4)])
    return pts, edges


def track_color(track_id: int) -> tuple[float, float, float]:
    return tuple(matplotlib.colormaps["tab20"](track_id % 20)[:3])


class LidarView3D:
    def __init__(self, width: int = 1280, height: int = 720):
        self.r = R.OffscreenRenderer(width, height)
        self.r.scene.set_background([0.02, 0.02, 0.05, 1.0])
        self.pt_mat = R.MaterialRecord(); self.pt_mat.shader = "defaultUnlit"; self.pt_mat.point_size = 2.0
        self.line_mat = R.MaterialRecord(); self.line_mat.shader = "unlitLine"; self.line_mat.line_width = 3.0
        l, w, h = EGO_LWH
        ego = o3d.geometry.TriangleMesh.create_box(l, w, h).translate([-l / 2, -w / 2, -1.73])
        ego.paint_uniform_color([0.9, 0.9, 0.9])
        self.r.scene.add_geometry("ego", ego, R.MaterialRecord())
        self.cmap = matplotlib.colormaps["turbo"]

    def render(self, points_velo: np.ndarray, boxes: list, label: str = "") -> np.ndarray:
        pts = np.asarray(points_velo, dtype=np.float64)[:, :3]
        pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts))
        z = np.clip((pts[:, 2] - Z_COLOR_RANGE[0]) / (Z_COLOR_RANGE[1] - Z_COLOR_RANGE[0]), 0.0, 1.0)
        pcd.colors = o3d.utility.Vector3dVector(self.cmap(z)[:, :3])
        for name in ("pcd", "boxes"):
            if self.r.scene.has_geometry(name):
                self.r.scene.remove_geometry(name)
        self.r.scene.add_geometry("pcd", pcd, self.pt_mat)

        if boxes:
            P, E, C = [], [], []
            for center, size, yaw, rgb in boxes:
                p, e = box_lines(np.asarray(center), np.asarray(size), float(yaw))
                E.append(e + sum(len(x) for x in P)); P.append(p); C += [rgb] * len(e)
            ls = o3d.geometry.LineSet(o3d.utility.Vector3dVector(np.vstack(P)), o3d.utility.Vector2iVector(np.vstack(E)))
            ls.colors = o3d.utility.Vector3dVector(np.array(C, dtype=np.float64))
            self.r.scene.add_geometry("boxes", ls, self.line_mat)

        # 카메라는 지오메트리를 다 넣은 뒤에 설정해야 한다 (먼저 하면 LineSet 이 그려지지 않는 Open3D 0.19 동작)
        self.r.setup_camera(CAMERA["fov"], CAMERA["lookat"], CAMERA["eye"], CAMERA["up"])
        img = cv2.cvtColor(np.asarray(self.r.render_to_image()), cv2.COLOR_RGB2BGR)
        if label:
            cv2.putText(img, label, (16, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
        return img
