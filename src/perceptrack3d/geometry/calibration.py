"""KITTI raw 캘리브레이션 파서와 좌표계 변환 (Phase 2).

좌표계 (docs/architecture.md §1)
- Velodyne (velo): x 전방, y 좌, z 상. 단위 m. LiDAR 원본 점의 프레임이자 이 프로젝트의 3D 기준 프레임.
- Camera 0 (cam0): x 우, y 아래, z 전방. 단위 m. 왼쪽 흑백 카메라(기준 카메라) 광학 중심이 원점.
- Rectified (rect): cam0 를 R_rect_00 으로 회전시켜 네 카메라의 이미지 평면을 평행하게 맞춘 프레임.
- Image 02 (img): 왼쪽 컬러 카메라의 정류(rectified) 이미지 픽셀. u 우, v 아래, 원점 좌상단.

파일과 행렬
- calib_velo_to_cam.txt
    R (3x3), T (3,)   :  p_cam0 = R p_velo + T.  여기서는 4x4 동차 행렬 T_velo_to_cam 으로 보관한다.
- calib_cam_to_cam.txt
    R_rect_00 (3x3)   :  기준 카메라(cam0)의 정류 회전. 4x4 로 확장해 보관 ([3, 3] = 1).
    P_rect_02 (3x4)   :  정류 후 카메라 2 의 투영 행렬
                             [[f, 0, cx, tx],
                              [0, f, cy, ty],
                              [0, 0,  1, tz]]
                         tx ≠ 0 인 이유: 카메라 2 는 cam0 에서 옆으로 떨어져 있어(기준선 baseline)
                         그 이동을 정류 프레임 안에서 f·(baseline) 형태로 표현하기 때문이다.
    S_rect_02 (2,)    :  정류 이미지 크기 (width, height) = (1242, 375).

투영 체인 (KITTI raw devkit readme.txt)
    y = P_rect_02 @ R_rect_00 @ T_velo_to_cam @ x_velo        (x_velo 는 (4,) 동차)
    (u, v) = (y0 / y2, y1 / y2),   y2 = 카메라 앞 깊이 (m)
주의: image_02 에 투영하더라도 회전은 R_rect_00 (카메라 0 의 정류 회전) 을 쓴다. R_rect_02 가 아니다.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from perceptrack3d.geometry.transforms import (
    apply_transform,
    invert_rigid,
    make_rigid,
    nearest_rotation,
)


def read_calib_file(path: str | Path) -> dict[str, np.ndarray]:
    """`key: v1 v2 ...` 형식의 KITTI 캘리브레이션 텍스트를 {key: float64 1차원 배열} 로 읽는다.

    숫자로 바꿀 수 없는 줄(예: calib_time: 15-Mar-2012 ...)은 건너뛴다. shape 은 호출 쪽에서 맞춘다.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"캘리브레이션 파일이 없습니다: {path}")
    data: dict[str, np.ndarray] = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            key, sep, value = line.partition(":")
            if not sep:
                continue
            try:
                data[key.strip()] = np.array([float(v) for v in value.split()], dtype=np.float64)
            except ValueError:
                continue  # calib_time 처럼 숫자가 아닌 값
    return data


@dataclass
class KittiCalibration:
    """한 날짜(2011_09_26)의 캘리브레이션. 행렬 shape 은 __post_init__ 에서 검증한다."""

    T_velo_to_cam: np.ndarray            # (4, 4) Velodyne → cam0 (동차)
    R_rect_00: np.ndarray                # (4, 4) cam0 → rect (3x3 회전을 4x4 로 확장)
    P_rect_02: np.ndarray                # (3, 4) rect → image_02 픽셀 (동차)
    image_size: tuple[int, int] = field(default=(1242, 375))  # (width, height) — S_rect_02 순서 그대로

    def __post_init__(self) -> None:
        self.T_velo_to_cam = np.asarray(self.T_velo_to_cam, dtype=np.float64)
        self.R_rect_00 = np.asarray(self.R_rect_00, dtype=np.float64)
        self.P_rect_02 = np.asarray(self.P_rect_02, dtype=np.float64)
        for name, M, shape in (
            ("T_velo_to_cam", self.T_velo_to_cam, (4, 4)),
            ("R_rect_00", self.R_rect_00, (4, 4)),
            ("P_rect_02", self.P_rect_02, (3, 4)),
        ):
            if M.shape != shape:
                raise ValueError(f"{name} 은 {shape} 이어야 합니다. 받은 shape: {M.shape}")
        for name, M in (("T_velo_to_cam", self.T_velo_to_cam), ("R_rect_00", self.R_rect_00)):
            if not np.allclose(M[3], [0, 0, 0, 1]):
                raise ValueError(f"{name} 의 마지막 행은 [0, 0, 0, 1] 이어야 합니다: {M[3]}")
        w, h = self.image_size
        self.image_size = (int(w), int(h))

    # ---- 생성 ----------------------------------------------------------------
    @classmethod
    def from_dir(cls, calib_dir: str | Path) -> "KittiCalibration":
        """calib_dir/{calib_velo_to_cam.txt, calib_cam_to_cam.txt} 를 읽어 생성한다.

        회전 행렬 R, R_rect_00 은 nearest_rotation() 으로 직교 보정한다 (텍스트 반올림 오차 ~1e-7 제거).
        """
        calib_dir = Path(calib_dir)
        vc = read_calib_file(calib_dir / "calib_velo_to_cam.txt")
        cc = read_calib_file(calib_dir / "calib_cam_to_cam.txt")
        try:
            R = nearest_rotation(vc["R"].reshape(3, 3))
            t = vc["T"].reshape(3)
            R_rect = nearest_rotation(cc["R_rect_00"].reshape(3, 3))
            P = cc["P_rect_02"].reshape(3, 4)
            S = cc["S_rect_02"]
        except KeyError as e:
            raise ValueError(f"캘리브레이션 파일에 키가 없습니다: {e}") from e
        return cls(
            T_velo_to_cam=make_rigid(R, t),
            R_rect_00=make_rigid(R_rect, np.zeros(3)),
            P_rect_02=P,
            image_size=(int(S[0]), int(S[1])),
        )

    # ---- 파생 행렬 -----------------------------------------------------------
    @property
    def image_shape(self) -> tuple[int, int]:
        """(H, W). numpy 이미지 shape 순서. project_velo_to_image 의 image_shape 인자에 그대로 넣는다."""
        return (self.image_size[1], self.image_size[0])

    @property
    def T_velo_to_rect(self) -> np.ndarray:
        """(4, 4) Velodyne → rect = R_rect_00 @ T_velo_to_cam. 오른쪽부터 적용된다(먼저 velo→cam0, 그다음 정류 회전)."""
        return self.R_rect_00 @ self.T_velo_to_cam

    @property
    def P_velo_to_img(self) -> np.ndarray:
        """(3, 4) Velodyne 동차 좌표 → 이미지 동차 좌표 [u·z, v·z, z]. = P_rect_02 @ R_rect_00 @ T_velo_to_cam."""
        return self.P_rect_02 @ self.T_velo_to_rect

    # ---- 점 변환 -------------------------------------------------------------
    def velo_to_rect(self, pts_velo: np.ndarray) -> np.ndarray:
        """(N, 3) Velodyne 점 → (N, 3) rect 프레임 점 (m). float64."""
        return apply_transform(self.T_velo_to_rect, pts_velo)

    def rect_to_velo(self, pts_rect: np.ndarray) -> np.ndarray:
        """(N, 3) rect 프레임 점 → (N, 3) Velodyne 점. velo_to_rect 의 정확한 역 (invert_rigid 사용)."""
        return apply_transform(invert_rigid(self.T_velo_to_rect), pts_rect)
