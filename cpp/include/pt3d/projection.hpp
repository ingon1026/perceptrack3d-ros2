// PercepTrack3D Phase 10 — LiDAR → image_02 투영, ROI, frustum 마스크 (C++ 구현).
//
// 규칙은 Python 기준 구현 src/perceptrack3d/geometry/projection.py 와 1:1 로 같다.
//     Velodyne 점 [x, y, z, 1]  →  P (3x4)  →  [u·d, v·d, d]
//     d > 0 인 점만            →  (u, v) = (u·d / d, v·d / d)
//     0 <= u < W, 0 <= v < H 인 점만 (하한 포함, 상한 제외, 반올림 없음)
//
// 데이터 레이아웃
//   - 점: row-major (N, 3) float32 (Velodyne 프레임, m). Eigen::Ref + OuterStride 를 쓰므로
//     로더가 주는 (N, 4) [x, y, z, reflectance] 배열의 앞 3열도 복사 없이 참조한다 (outer stride = 4).
//   - P: (3, 4) float64. Python 이 float64 로 계산하므로 여기서도 double 로 계산해야
//     경계(u < W 등) 판정이 비트 단위로 같아진다. 점은 float32 → double 로 정확히 승격된다.
//
// 소유권: 모든 함수는 입력을 const 참조로 읽기만 하고, 결과를 값(std::vector)으로 돌려준다.
//         힙 객체를 함수 밖까지 공유할 일이 없어 스마트 포인터는 쓰지 않는다 (pybind11 경계 제외).
#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

#include <Eigen/Core>

namespace pt3d {

/// row-major (N, 3) float32 점 행렬.
using PointMatrix = Eigen::Matrix<float, Eigen::Dynamic, 3, Eigen::RowMajor>;

/// 읽기 전용 점 참조. OuterStride<> 덕분에 행 간격(outer stride)이 3 이 아니어도(예: (N,4) 배열의
/// 앞 3열, stride 4) 복사 없이 바인딩된다. 함수 인자로만 쓰고 저장하지 않는다 (뷰이므로 수명은 호출자 소유).
using PointsRef = Eigen::Ref<const PointMatrix, 0, Eigen::OuterStride<>>;

/// (3, 4) float64 투영 행렬 = P_rect_02 @ R_rect_00 @ T_velo_to_cam.
using ProjMatrix = Eigen::Matrix<double, 3, 4>;

/// projectVeloToImage 결과. 값 타입 — 함수가 값으로 돌려주며 복사 생략(RVO)된다.
struct ProjectionResult {
    std::vector<float> uv;            ///< (M*2) [u0, v0, u1, v1, ...] 픽셀. numpy 로는 (M, 2).
    std::vector<float> depth;         ///< (M) 카메라 2 광학축 깊이 d (m), 항상 > 0.
    std::vector<std::uint8_t> mask;   ///< (N) 1 = 카메라 앞이면서 이미지 안. sum == M.

    std::size_t numProjected() const noexcept { return depth.size(); }
};

/// Velodyne 점을 image_02 픽셀로 투영한다.
/// @param pts    (N, 3) float32 점 (Velodyne 프레임, m). (N,4) 배열은 stride 로 앞 3열만 넘긴다.
/// @param P      (3, 4) float64 투영 행렬.
/// @param height 이미지 H (px), @param width 이미지 W (px).
/// @return uv (M,2), depth (M), mask (N). uv[i] 는 mask 가 1 인 i 번째 점에 대응한다.
ProjectionResult projectVeloToImage(const PointsRef& pts, const ProjMatrix& P, int height, int width);

/// ROI 박스 (Velodyne 프레임, m). configs/kitti.yaml 의 fusion.roi 와 같은 의미.
/// 멤버가 const 이므로 생성 후 바뀌지 않는다 (집합 초기화: RoiParams{0.0, 70.0, 40.0, -3.0, 3.0}).
/// double 인 이유: Python 기준 roi_filter 가 점을 float64 로 캐스팅한 뒤 float64 임계값과 비교하므로,
/// 여기서도 점을 double 로 승격해 double 임계값과 비교해야 임계값이 float32 로 정확히 표현되지 않는 경우(예: 0.1)에도 결과가 같다.
struct RoiParams {
    const double x_min;
    const double x_max;
    const double y_abs_max;
    const double z_min;
    const double z_max;
};

/// x_min <= x <= x_max, |y| <= y_abs_max, z_min <= z <= z_max (모두 포함 경계) 인 점의 마스크 (N).
std::vector<std::uint8_t> roiMask(const PointsRef& pts, const RoiParams& roi);

/// 2D 검출 박스 (image_02 픽셀, x1 <= x2, y1 <= y2). 축소(shrink)는 호출자가 미리 적용한다.
struct Box2D {
    double x1;
    double y1;
    double x2;
    double y2;
};

/// K 개 박스에 대한 frustum 마스크 (K*N, row-major: [k*N + i]).
/// 투영은 한 번만 하고 박스마다 비교만 반복한다 (Phase 5/6 의 검출당 반복 호출을 대체하는 핫스팟).
/// 점 i 가 박스 k 안: projectVeloToImage 의 mask 가 1 이고 (즉 d > 0 & 이미지 안),
///                    x1 <= u <= x2, y1 <= v <= y2 (포함 경계). u, v 는 float32 로 반올림한 값으로 비교해
///                    Python 의 project_velo_to_image 출력(float32 uv)에 박스 조건을 건 결과와 같다.
std::vector<std::uint8_t> frustumMasks(const PointsRef& pts, const ProjMatrix& P, int height, int width,
                                       const std::vector<Box2D>& boxes);

}  // namespace pt3d
