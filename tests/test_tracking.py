"""Phase 7 추적 모듈 테스트: 칼만 필터 단위, 할당, 트랙 생명주기. 모두 결정적(시드 고정)."""
from __future__ import annotations

import numpy as np
import pytest

from perceptrack3d.config import load_config
from perceptrack3d.tracking.association import build_cost_matrix, hungarian_assign
from perceptrack3d.tracking.kalman_tracker import KalmanFilterCV, KalmanTracker
from perceptrack3d.types import Object3D, Track


def make_cfg(**over) -> dict:
    t = dict(
        dt=0.1, gating="mahalanobis", gate_mahalanobis=9.21, gate_euclidean_m=3.0,
        max_misses=3, min_hits=3, match_same_class=False,
        process_noise_pos=0.5, process_noise_vel=1.0, measurement_noise=0.5, initial_vel_std=10.0,
    )
    t.update(over)
    return {"tracking": t}


def obj(center, class_name="car", status="ok", frame_id=0, size=None) -> Object3D:
    return Object3D(
        frame_id=frame_id, class_name=class_name, confidence=0.9,
        xyxy=np.zeros(4, dtype=np.float32),
        center=None if center is None else np.asarray(center, dtype=float),
        size=None if size is None else np.asarray(size, dtype=float),
        status=status,
    )


def make_track(tid, pos, class_name="car", P_pos=0.25, P_vel=1.0) -> Track:
    state = np.concatenate([np.asarray(pos, float), np.zeros(3)])
    P = np.diag([P_pos] * 3 + [P_vel] * 3)
    return Track(track_id=tid, class_name=class_name, state=state, covariance=P)


# ---------------------------------------------------------------- 칼만 필터 단위

def test_kf_matrices_shapes():
    kf = KalmanFilterCV(0.1, 0.5, 1.0, 0.5, 10.0)
    assert kf.F.shape == (6, 6) and kf.H.shape == (3, 6)
    assert kf.Q.shape == (6, 6) and kf.R.shape == (3, 3)
    x, P = kf.init_state([1.0, 2.0, 3.0])
    assert x.shape == (6,) and P.shape == (6, 6)
    np.testing.assert_allclose(x, [1, 2, 3, 0, 0, 0])
    np.testing.assert_allclose(P[:3, :3], 0.25 * np.eye(3))
    np.testing.assert_allclose(P[3:, 3:], 100.0 * np.eye(3))


def test_kf_predict_moves_by_dt_times_velocity():
    kf = KalmanFilterCV(dt=0.1, process_noise_pos=0.5, process_noise_vel=1.0,
                        measurement_noise=0.5, initial_vel_std=10.0)
    x = np.array([10.0, -2.0, 0.5, 8.0, 1.0, -0.5])
    P = np.eye(6)
    x1, P1 = kf.predict(x, P)
    np.testing.assert_allclose(x1[:3], x[:3] + 0.1 * x[3:])
    np.testing.assert_allclose(x1[3:], x[3:])                       # 속도는 그대로
    assert np.trace(P1) > np.trace(P)                                # 예측은 불확실성을 키운다
    np.testing.assert_allclose(P1, P1.T)                             # 대칭 유지


def test_kf_update_shrinks_covariance_and_converges_to_measurement_mean():
    kf = KalmanFilterCV(0.1, 0.5, 1.0, 0.5, 10.0)
    rng = np.random.default_rng(0)
    true_pos = np.array([15.0, 3.0, 0.8])
    zs = true_pos + rng.normal(0.0, 0.5, size=(200, 3))

    # (a) 예측 없이 갱신만 반복: P 는 단조 감소, 상태는 관측 평균으로
    x, P = kf.init_state(zs[0])
    prev = np.trace(P)
    for z in zs[1:]:
        x, P = kf.update(x, P, z)
        assert np.trace(P) < prev
        prev = np.trace(P)
    np.testing.assert_allclose(x[:3], zs.mean(axis=0), atol=0.05)
    np.testing.assert_allclose(x[3:], 0.0)                           # 관측은 위치만 → 속도 불변

    # (b) 예측+갱신 사이클: 필터 오차(RMS, 마지막 100 프레임) 가 원시 관측 오차보다 작아야 한다.
    #     Q 가 크면(기본값 σp=0.5) 관측을 거의 따르므로 개선폭은 작고, Q 가 작으면 평균에 가깝게 수렴한다.
    def run(qp, qv):
        kf_ = KalmanFilterCV(0.1, qp, qv, 0.5, 10.0)
        x_, P_ = kf_.init_state(zs[0])
        errs, raws, vels = [], [], []
        for z in zs[1:]:
            x_, P_ = kf_.predict(x_, P_)
            x_, P_ = kf_.update(x_, P_, z)
            errs.append(np.linalg.norm(x_[:3] - true_pos))
            raws.append(np.linalg.norm(z - true_pos))
            vels.append(np.linalg.norm(x_[3:]))
        rms = lambda a: float(np.sqrt(np.mean(np.square(a[-100:]))))
        return rms(errs), rms(raws), float(np.mean(vels[-100:]))

    filt, raw, vel = run(0.5, 1.0)                                   # 기본 파라미터
    assert filt < raw
    filt, raw, vel = run(0.05, 0.1)                                  # 작은 Q: 강한 평활
    assert filt < 0.4 and filt < raw
    assert vel < 0.5                                                 # 정지 물체 → 속도 ≈ 0


def test_kf_gain_matches_explicit_formula():
    kf = KalmanFilterCV(0.1, 0.5, 1.0, 0.5, 10.0)
    x, P = kf.init_state([5.0, 0.0, 0.0])
    x, P = kf.predict(x, P)
    z = np.array([5.4, 0.2, -0.1])
    H, R = kf.H, kf.R
    K = P @ H.T @ np.linalg.inv(H @ P @ H.T + R)
    x_expected = x + K @ (z - H @ x)
    x_new, _ = kf.update(x, P, z)
    np.testing.assert_allclose(x_new, x_expected)


# ---------------------------------------------------------------- 할당

def test_cost_matrix_and_hungarian_clear_case():
    tracks = [make_track(1, [10, 0, 0]), make_track(2, [20, 5, 0]), make_track(3, [30, -5, 0])]
    objects = [obj([30.2, -5.1, 0.1]), obj([9.8, 0.3, 0]), obj([20.1, 4.9, 0])]   # 순서를 섞음
    R = 0.25 * np.eye(3)
    cost = build_cost_matrix(tracks, objects, "mahalanobis", 9.21, 3.0, R=R)
    assert cost.shape == (3, 3)
    matches, ut, uo = hungarian_assign(cost)
    assert sorted(matches) == [(0, 1), (1, 2), (2, 0)]
    assert ut == [] and uo == []


def test_out_of_gate_object_is_unmatched():
    tracks = [make_track(1, [10, 0, 0])]
    objects = [obj([10.1, 0, 0]), obj([40, 0, 0])]
    R = 0.25 * np.eye(3)
    cost = build_cost_matrix(tracks, objects, "mahalanobis", 9.21, 3.0, R=R)
    assert np.isfinite(cost[0, 0]) and np.isinf(cost[0, 1])
    matches, ut, uo = hungarian_assign(cost)
    assert matches == [(0, 0)] and ut == [] and uo == [1]

    # 트랙 하나가 어느 관측과도 게이트 안에 없으면 매칭되지 않는다
    far = [make_track(9, [-50, 0, 0])]
    cost = build_cost_matrix(far, objects, "mahalanobis", 9.21, 3.0, R=R)
    matches, ut, uo = hungarian_assign(cost)
    assert matches == [] and ut == [0] and uo == [0, 1]


def test_mahalanobis_value_and_euclidean_gate():
    tr = make_track(1, [0, 0, 0], P_pos=0.75)          # S = 0.75 + 0.25 = 1.0 → d² = ‖diff‖²
    o = obj([2.0, 0.0, 0.0])
    cost = build_cost_matrix([tr], [o], "mahalanobis", 9.21, 3.0, R=0.25 * np.eye(3))
    np.testing.assert_allclose(cost[0, 0], 4.0)
    cost_e = build_cost_matrix([tr], [o], "euclidean", 9.21, 3.0)
    np.testing.assert_allclose(cost_e[0, 0], 2.0)
    cost_e2 = build_cost_matrix([tr], [obj([3.5, 0, 0])], "euclidean", 9.21, 3.0)
    assert np.isinf(cost_e2[0, 0])


def test_hungarian_empty_and_all_inf():
    m, ut, uo = hungarian_assign(np.zeros((0, 2)))
    assert m == [] and ut == [] and uo == [0, 1]
    m, ut, uo = hungarian_assign(np.zeros((2, 0)))
    assert m == [] and ut == [0, 1] and uo == []
    m, ut, uo = hungarian_assign(np.full((2, 2), np.inf))
    assert m == [] and ut == [0, 1] and uo == [0, 1]


def test_hungarian_gate_argument_removes_costly_pairs():
    cost = np.array([[0.5, 5.0], [5.0, 0.4]])
    m, ut, uo = hungarian_assign(cost, gate=1.0)
    assert sorted(m) == [(0, 0), (1, 1)]
    m, ut, uo = hungarian_assign(np.array([[2.0]]), gate=1.0)
    assert m == [] and ut == [0] and uo == [0]


def test_match_same_class_option():
    tracks = [make_track(1, [10, 0, 0], class_name="car")]
    objects = [obj([10.1, 0, 0], class_name="person")]
    R = 0.25 * np.eye(3)
    assert np.isfinite(build_cost_matrix(tracks, objects, "mahalanobis", 9.21, 3.0, R=R)[0, 0])
    assert np.isinf(build_cost_matrix(tracks, objects, "mahalanobis", 9.21, 3.0, R=R,
                                      match_same_class=True)[0, 0])


# ---------------------------------------------------------------- 생명주기

def test_confirmed_only_after_min_hits():
    tk = KalmanTracker(make_cfg(min_hits=3))
    assert tk.step([obj([10, 0, 0])], 0) == []
    assert len(tk.all_tracks) == 1 and not tk.all_tracks[0].confirmed
    assert tk.step([obj([10.5, 0, 0])], 1) == []
    out = tk.step([obj([11.0, 0, 0])], 2)
    assert len(out) == 1 and out[0].track_id == 1 and out[0].confirmed and out[0].hits == 3


def test_id_survives_two_missed_frames():
    tk = KalmanTracker(make_cfg(min_hits=1, max_misses=3))
    for f in range(5):
        out = tk.step([obj([10 + 0.5 * f, 0, 0])], f)
    assert [t.track_id for t in out] == [1]
    # 2 프레임 미검출: coasting 트랙이 예측 위치로 계속 반환된다
    out = tk.step([], 5)
    assert [t.track_id for t in out] == [1] and out[0].misses == 1
    out = tk.step([], 6)
    assert [t.track_id for t in out] == [1] and out[0].misses == 2
    # 재검출: 같은 id, misses 초기화, 새 트랙 생성 없음
    out = tk.step([obj([10 + 0.5 * 7, 0, 0])], 7)
    assert [t.track_id for t in out] == [1] and out[0].misses == 0
    assert len(tk.all_tracks) == 1


def test_deleted_after_max_misses_exceeded():
    tk = KalmanTracker(make_cfg(min_hits=1, max_misses=3))
    tk.step([obj([10, 0, 0])], 0)
    for f in range(1, 4):                        # misses = 1, 2, 3 → 아직 유지
        tk.step([], f)
        assert len(tk.all_tracks) == 1
    tk.step([], 4)                                # misses = 4 > 3 → 삭제
    assert tk.all_tracks == []
    # 이후 새 관측은 새 id (재사용 금지)
    tk.step([obj([10, 0, 0])], 5)
    assert tk.all_tracks[0].track_id == 2


def test_no_id_explosion_two_objects_40_frames():
    rng = np.random.default_rng(42)
    tk = KalmanTracker(make_cfg())
    ids_seen = set()
    for f in range(40):
        a = np.array([10 + 0.8 * f, 3.0, 0.8]) + rng.normal(0, 0.3, 3)
        b = np.array([30 - 0.5 * f, -3.0, 0.8]) + rng.normal(0, 0.3, 3)
        out = tk.step([obj(a), obj(b)], f)
        ids_seen.update(t.track_id for t in out)
    assert ids_seen == {1, 2}
    assert len(tk.all_tracks) == 2
    for t in tk.all_tracks:
        assert t.hits == 40 and t.misses == 0


def test_non_ok_objects_are_ignored():
    tk = KalmanTracker(make_cfg(min_hits=1))
    out = tk.step([obj([10, 0, 0], status="sparse"), obj(None, status="empty"),
                   obj([20, 0, 0], status="invalid")], 0)
    assert out == [] and tk.all_tracks == []
    out = tk.step([obj([10, 0, 0], status="ok", size=[4, 1.8, 1.5])], 1)
    assert len(out) == 1
    np.testing.assert_allclose(out[0].size, [4, 1.8, 1.5])
    assert out[0].history == [(1, 10.0, 0.0, 0.0)]


def test_euclidean_gating_tracker_runs():
    tk = KalmanTracker(make_cfg(gating="euclidean", min_hits=1))
    tk.step([obj([10, 0, 0])], 0)
    out = tk.step([obj([10.5, 0, 0]), obj([50, 0, 0])], 1)
    assert sorted(t.track_id for t in out) == [1, 2]


def test_invalid_gating_raises():
    with pytest.raises(ValueError):
        KalmanTracker(make_cfg(gating="cosine"))


def test_repo_config_has_all_tracking_keys():
    cfg = load_config()
    tk = KalmanTracker(cfg)                        # 키 누락이면 KeyError
    assert tk.kf.dt == pytest.approx(0.1)
    assert tk.gating in ("mahalanobis", "euclidean")
