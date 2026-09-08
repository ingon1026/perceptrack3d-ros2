"""칼만 필터 기반 다중 객체 추적 (Phase 7, 투명한 고전 기준선).

입력 : list[Object3D] — status == "ok" 이고 center 가 있는 객체만 관측 z (3,) 로 사용. Velodyne 프레임 (x 전방, y 좌, z 상), m.
출력 : list[Track]    — confirmed 트랙만 반환. state (6,) = [px, py, pz, vx, vy, vz] (m, m/s), covariance (6, 6).
가정 : step() 은 프레임 순서대로 매 프레임 호출한다 (dt 고정 = cfg["tracking"]["dt"]). 프레임을 건너뛰면 예측 거리가 실제보다 짧아진다.

등속(constant velocity) 모델
    x_k = F x_{k-1} + w,   w ~ N(0, Q)      F = [[I3, dt·I3], [0, I3]]
    z_k = H x_k + v,       v ~ N(0, R)      H = [I3, 0]  (위치만 관측)

칼만 필터 한 사이클
    예측  x⁻ = F x,               P⁻ = F P Fᵀ + Q
    갱신  S  = H P⁻ Hᵀ + R        (innovation 공분산, 3×3)
          K  = P⁻ Hᵀ S⁻¹          (칼만 이득, 6×3)
          x  = x⁻ + K (z − H x⁻)  (혁신 y = z − H x⁻ 를 이득만큼 반영)
          P  = (I − K H) P⁻
"""
from __future__ import annotations

import numpy as np

from perceptrack3d.tracking.association import build_cost_matrix, hungarian_assign
from perceptrack3d.types import Object3D, Track


class KalmanFilterCV:
    """등속 모델 칼만 필터. 상태를 갖지 않는다 — (x, P) 를 받아 새 (x, P) 를 돌려주는 순수 함수 모음.

    잡음 모델 (단순 대각): Q = diag(σp², σp², σp², σv², σv², σv²)
        σp = process_noise_pos [m]   : 한 프레임 동안 등속 가정에서 벗어나는 위치 오차
        σv = process_noise_vel [m/s] : 한 프레임 동안 속도가 변할 수 있는 정도 (가속·회전)
    관측 잡음: R = σz² I3,  σz = measurement_noise [m] (융합 중심 추정의 오차)
    초기 공분산: P0 = diag(R, σv0² I3),  σv0 = initial_vel_std [m/s] (첫 관측에서 속도는 모른다)
    """

    def __init__(self, dt: float, process_noise_pos: float, process_noise_vel: float,
                 measurement_noise: float, initial_vel_std: float):
        self.dt = float(dt)
        I3, Z3 = np.eye(3), np.zeros((3, 3))
        self.F = np.block([[I3, self.dt * I3], [Z3, I3]])            # (6, 6) 상태 전이
        self.H = np.hstack([I3, Z3])                                  # (3, 6) 관측 (위치만)
        self.Q = np.diag([process_noise_pos ** 2] * 3 + [process_noise_vel ** 2] * 3)   # (6, 6)
        self.R = (measurement_noise ** 2) * I3                        # (3, 3)
        self.initial_vel_std = float(initial_vel_std)

    def init_state(self, z: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """첫 관측 z (3,) 로 상태를 만든다. 위치 = z, 속도 = 0 (불확실성은 initial_vel_std 로 표현)."""
        z = _as_vec3(z)
        x = np.concatenate([z, np.zeros(3)])
        P = np.zeros((6, 6))
        P[:3, :3] = self.R
        P[3:, 3:] = (self.initial_vel_std ** 2) * np.eye(3)
        return x, P

    def predict(self, x: np.ndarray, P: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """한 프레임(dt) 앞을 예측한다. x⁻ = F x,  P⁻ = F P Fᵀ + Q"""
        F = self.F
        x_pred = F @ x
        P_pred = F @ P @ F.T + self.Q
        return x_pred, P_pred

    def innovation_covariance(self, P: np.ndarray) -> np.ndarray:
        """S = H P Hᵀ + R  (3, 3). 마할라노비스 게이트와 칼만 이득이 공유하는 행렬."""
        H = self.H
        return H @ P @ H.T + self.R

    def update(self, x: np.ndarray, P: np.ndarray, z: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """관측 z (3,) 로 상태를 보정한다.

        K = P Hᵀ (H P Hᵀ + R)⁻¹,  x = x + K (z − H x),  P = (I − K H) P
        """
        z = _as_vec3(z)
        H = self.H
        S = H @ P @ H.T + self.R                 # (3, 3)
        K = P @ H.T @ np.linalg.inv(S)           # (6, 3) 칼만 이득
        y = z - H @ x                            # (3,) 혁신(innovation)
        x_new = x + K @ y
        P_new = (np.eye(6) - K @ H) @ P
        return x_new, P_new


class KalmanTracker:
    """predict → associate → update → lifecycle 를 프레임마다 수행하는 다중 객체 추적기.

    cfg["tracking"] 키: dt, gating, gate_mahalanobis, gate_euclidean_m, max_misses, min_hits,
                        process_noise_pos, process_noise_vel, measurement_noise, initial_vel_std,
                        match_same_class (선택, 기본 False)
    생명주기:
        생성   미매칭 관측 → 새 트랙 (hits=1, misses=0, track_id 는 1부터 단조 증가, 재사용 없음)
        확정   hits >= min_hits 이면 confirmed=True (한 번 확정되면 유지). 반환은 확정 트랙만.
        유지   매칭되면 hits+1, misses=0. 미매칭이면 misses+1 (상태는 예측값 그대로 = coasting)
        삭제   misses > max_misses
    """

    def __init__(self, cfg: dict):
        t = cfg["tracking"]
        self.kf = KalmanFilterCV(
            dt=t["dt"],
            process_noise_pos=t["process_noise_pos"],
            process_noise_vel=t["process_noise_vel"],
            measurement_noise=t["measurement_noise"],
            initial_vel_std=t["initial_vel_std"],
        )
        self.gating = str(t["gating"])
        if self.gating not in ("mahalanobis", "euclidean"):
            raise ValueError(f"tracking.gating 은 'mahalanobis' 또는 'euclidean' 이어야 합니다: {self.gating!r}")
        self.gate_mahalanobis = float(t["gate_mahalanobis"])
        self.gate_euclidean = float(t["gate_euclidean_m"])
        self.max_misses = int(t["max_misses"])
        self.min_hits = int(t["min_hits"])
        self.match_same_class = bool(t.get("match_same_class", False))
        self._tracks: list[Track] = []
        self._next_id = 1

    @property
    def all_tracks(self) -> list[Track]:
        """확정 여부와 무관하게 살아 있는 모든 트랙 (tentative 포함)."""
        return list(self._tracks)

    def step(self, objects: list[Object3D], frame_id: int) -> list[Track]:
        """한 프레임을 처리하고 confirmed 트랙 목록을 돌려준다.

        반환 트랙에는 이번 프레임에 관측이 없어 예측만 한(coasting, misses > 0) 트랙도 포함된다.
        관측된 트랙만 원하면 misses == 0 으로 거른다. history 에는 관측으로 갱신된 프레임만 쌓인다.
        """
        # (1) 모든 트랙 예측
        for tr in self._tracks:
            tr.state, tr.covariance = self.kf.predict(tr.state, tr.covariance)
            tr.age += 1

        # (2) 유효한 관측만 선택
        observations = [o for o in objects if o.status == "ok" and o.center is not None]
        for o in observations:
            _as_vec3(o.center)  # shape 검증

        # (3) 비용 행렬 + 헝가리안 할당
        cost = build_cost_matrix(
            self._tracks, observations,
            gating=self.gating,
            gate_mahalanobis=self.gate_mahalanobis,
            gate_euclidean=self.gate_euclidean,
            R=self.kf.R,
            match_same_class=self.match_same_class,
        )
        matches, unmatched_tracks, unmatched_objects = hungarian_assign(cost)

        # (4) 매칭된 트랙 갱신
        for ti, oi in matches:
            tr, obj = self._tracks[ti], observations[oi]
            tr.state, tr.covariance = self.kf.update(tr.state, tr.covariance, obj.center)
            tr.hits += 1
            tr.misses = 0
            if obj.size is not None:
                tr.size = np.asarray(obj.size, dtype=float).copy()
            tr.history.append((int(frame_id), *(float(v) for v in tr.state[:3])))

        # (5) 미매칭 트랙: 놓친 횟수 증가 (상태는 예측값 유지)
        for ti in unmatched_tracks:
            self._tracks[ti].misses += 1

        # (6) 미매칭 관측 → 새 트랙
        for oi in unmatched_objects:
            self._tracks.append(self._new_track(observations[oi], frame_id))

        # (7) 확정
        for tr in self._tracks:
            if tr.hits >= self.min_hits:
                tr.confirmed = True

        # (8) 삭제
        self._tracks = [tr for tr in self._tracks if tr.misses <= self.max_misses]

        return [tr for tr in self._tracks if tr.confirmed]

    def _new_track(self, obj: Object3D, frame_id: int) -> Track:
        x, P = self.kf.init_state(obj.center)
        tr = Track(
            track_id=self._next_id,
            class_name=obj.class_name,
            state=x,
            covariance=P,
            hits=1,
            age=0,
            misses=0,
            confirmed=False,
            size=None if obj.size is None else np.asarray(obj.size, dtype=float).copy(),
            history=[(int(frame_id), *(float(v) for v in x[:3]))],
        )
        self._next_id += 1
        return tr


def _as_vec3(v) -> np.ndarray:
    arr = np.asarray(v, dtype=float)
    if arr.shape != (3,):
        raise ValueError(f"3D 위치는 shape (3,) 이어야 합니다: {arr.shape}")
    return arr
