#include "pt3d/projection.hpp"

#include <cmath>

namespace pt3d {
namespace {

/// 점 하나의 투영 결과. u, v 는 inImage 가 true 일 때만 의미가 있다.
struct Projected {
    double u;
    double v;
    double d;
    bool inImage;
};

/// 점 i 를 투영한다. Python 과 같은 순서: d 계산 → d > 0 검사 → 나누기 → 경계 검사.
/// NaN 입력은 d > 0 이 거짓이 되어 자동으로 제외된다 (Python 의 `d > 0` 과 같은 동작).
inline Projected projectOne(const PointsRef& pts, Eigen::Index i, const ProjMatrix& P, int height,
                            int width) noexcept {
    const double x = pts(i, 0);  // float32 → double 승격 (정확)
    const double y = pts(i, 1);
    const double z = pts(i, 2);
    const double d = P(2, 0) * x + P(2, 1) * y + P(2, 2) * z + P(2, 3);
    if (!(d > 0.0)) {
        return {0.0, 0.0, d, false};
    }
    const double u = (P(0, 0) * x + P(0, 1) * y + P(0, 2) * z + P(0, 3)) / d;
    const double v = (P(1, 0) * x + P(1, 1) * y + P(1, 2) * z + P(1, 3)) / d;
    const bool inImage = u >= 0.0 && u < static_cast<double>(width) && v >= 0.0 && v < static_cast<double>(height);
    return {u, v, d, inImage};
}

}  // namespace

ProjectionResult projectVeloToImage(const PointsRef& pts, const ProjMatrix& P, int height, int width) {
    const Eigen::Index n = pts.rows();
    ProjectionResult result;
    result.mask.assign(static_cast<std::size_t>(n), 0);
    // 최악의 경우(모든 점이 이미지 안)에 맞춰 한 번만 예약해 두면 루프 안에서 재할당이 일어나지 않는다.
    result.uv.reserve(static_cast<std::size_t>(2 * n));
    result.depth.reserve(static_cast<std::size_t>(n));

    for (Eigen::Index i = 0; i < n; ++i) {
        const Projected p = projectOne(pts, i, P, height, width);
        if (!p.inImage) {
            continue;
        }
        result.mask[static_cast<std::size_t>(i)] = 1;
        result.uv.push_back(static_cast<float>(p.u));
        result.uv.push_back(static_cast<float>(p.v));
        result.depth.push_back(static_cast<float>(p.d));
    }
    return result;  // 지역 변수를 값으로 반환 → NRVO, 복사 없음
}

std::vector<std::uint8_t> roiMask(const PointsRef& pts, const RoiParams& roi) {
    const Eigen::Index n = pts.rows();
    std::vector<std::uint8_t> mask(static_cast<std::size_t>(n));
    for (Eigen::Index i = 0; i < n; ++i) {
        const double x = pts(i, 0);  // float32 → double 승격 (정확). Python 의 astype(float64) 와 같다.
        const double y = pts(i, 1);
        const double z = pts(i, 2);
        const bool inside = x >= roi.x_min && x <= roi.x_max && std::fabs(y) <= roi.y_abs_max &&
                            z >= roi.z_min && z <= roi.z_max;
        mask[static_cast<std::size_t>(i)] = inside ? 1 : 0;
    }
    return mask;
}

std::vector<std::uint8_t> frustumMasks(const PointsRef& pts, const ProjMatrix& P, int height, int width,
                                       const std::vector<Box2D>& boxes) {
    const Eigen::Index n = pts.rows();
    const std::size_t nn = static_cast<std::size_t>(n);

    // 1) 투영은 한 번만. u, v 는 Python 출력과 같은 float32 로 반올림해 저장한다.
    std::vector<float> u(nn);
    std::vector<float> v(nn);
    std::vector<std::uint8_t> inImage(nn);
    for (Eigen::Index i = 0; i < n; ++i) {
        const Projected p = projectOne(pts, i, P, height, width);
        const std::size_t k = static_cast<std::size_t>(i);
        inImage[k] = p.inImage ? 1 : 0;
        u[k] = static_cast<float>(p.u);
        v[k] = static_cast<float>(p.v);
    }

    // 2) 박스마다 포함 경계 비교만 반복한다 (박스 수 K 는 보통 5 내외).
    std::vector<std::uint8_t> out(boxes.size() * nn, 0);
    for (std::size_t b = 0; b < boxes.size(); ++b) {
        const Box2D& box = boxes[b];
        std::uint8_t* row = out.data() + b * nn;
        for (std::size_t i = 0; i < nn; ++i) {
            if (!inImage[i]) {
                continue;
            }
            const double ui = u[i];  // float32 값을 double 로 올려 double 박스 좌표와 비교
            const double vi = v[i];
            row[i] = (ui >= box.x1 && ui <= box.x2 && vi >= box.y1 && vi <= box.y2) ? 1 : 0;
        }
    }
    return out;
}

}  // namespace pt3d
