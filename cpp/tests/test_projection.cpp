// pt3d_core 단위 테스트 (GoogleTest). 손계산이 가능한 핀홀 행렬로 규칙을 하나씩 검증한다.
#include <gtest/gtest.h>

#include <array>
#include <initializer_list>
#include <vector>

#include "pt3d/projection.hpp"

namespace {

/// 점이 이미 카메라 프레임(x 우, y 아래, z 전방)에 있다고 가정한 핀홀 행렬.
/// u = f·x/z + cx, v = f·y/z + cy, d = z. 손으로 계산하기 쉽다.
pt3d::ProjMatrix pinhole(double f, double cx, double cy) {
    pt3d::ProjMatrix P;
    P << f, 0, cx, 0,
         0, f, cy, 0,
         0, 0, 1, 0;
    return P;
}

pt3d::PointMatrix points(std::initializer_list<std::array<float, 3>> rows) {
    pt3d::PointMatrix m(static_cast<Eigen::Index>(rows.size()), 3);
    Eigen::Index i = 0;
    for (const auto& r : rows) {
        m(i, 0) = r[0];
        m(i, 1) = r[1];
        m(i, 2) = r[2];
        ++i;
    }
    return m;
}

std::vector<std::uint8_t> u8(std::initializer_list<int> v) { return std::vector<std::uint8_t>(v.begin(), v.end()); }

}  // namespace

TEST(Projection, KnownPinholePoints) {
    const auto P = pinhole(100.0, 600.0, 200.0);
    const auto pts = points({{0.f, 0.f, 10.f}, {1.f, 0.f, 10.f}, {0.f, -2.f, 5.f}});
    const auto r = pt3d::projectVeloToImage(pts, P, 375, 1242);

    EXPECT_EQ(r.mask, u8({1, 1, 1}));
    ASSERT_EQ(r.numProjected(), 3u);
    EXPECT_FLOAT_EQ(r.uv[0], 600.f);  EXPECT_FLOAT_EQ(r.uv[1], 200.f);  EXPECT_FLOAT_EQ(r.depth[0], 10.f);
    EXPECT_FLOAT_EQ(r.uv[2], 610.f);  EXPECT_FLOAT_EQ(r.uv[3], 200.f);  EXPECT_FLOAT_EQ(r.depth[1], 10.f);
    EXPECT_FLOAT_EQ(r.uv[4], 600.f);  EXPECT_FLOAT_EQ(r.uv[5], 160.f);  EXPECT_FLOAT_EQ(r.depth[2], 5.f);
}

TEST(Projection, PointsBehindOrOnCameraPlaneAreRemoved) {
    const auto P = pinhole(100.0, 600.0, 200.0);
    const auto pts = points({{0.f, 0.f, -5.f}, {0.f, 0.f, 0.f}, {0.f, 0.f, 1.f}});
    const auto r = pt3d::projectVeloToImage(pts, P, 375, 1242);
    EXPECT_EQ(r.mask, u8({0, 0, 1}));
    ASSERT_EQ(r.numProjected(), 1u);
    EXPECT_FLOAT_EQ(r.depth[0], 1.f);
}

TEST(Projection, ImageBoundaryLowerInclusiveUpperExclusive) {
    // f=1, cx=cy=0, z=1 이면 (u, v) = (x, y). W=10, H=5.
    const auto P = pinhole(1.0, 0.0, 0.0);
    const auto pts = points({
        {0.f, 0.f, 1.f},       // u=0        → 포함 (0 <= u)
        {10.f, 0.f, 1.f},      // u=10=W     → 제외 (u < W)
        {9.5f, 0.f, 1.f},      // u=9.5      → 포함
        {-0.001f, 0.f, 1.f},   // u<0        → 제외
        {0.f, 5.f, 1.f},       // v=5=H      → 제외
        {0.f, 4.999f, 1.f},    // v=4.999    → 포함
    });
    const auto r = pt3d::projectVeloToImage(pts, P, 5, 10);
    EXPECT_EQ(r.mask, u8({1, 0, 1, 0, 0, 1}));
    EXPECT_EQ(r.numProjected(), 3u);
}

TEST(Projection, EmptyInput) {
    const pt3d::PointMatrix pts(0, 3);
    const auto r = pt3d::projectVeloToImage(pts, pinhole(1.0, 0.0, 0.0), 5, 10);
    EXPECT_TRUE(r.mask.empty());
    EXPECT_TRUE(r.uv.empty());
    EXPECT_TRUE(r.depth.empty());
    EXPECT_EQ(r.numProjected(), 0u);
}

TEST(Projection, StridedN4InputIsZeroCopyAndEqualToN3) {
    // (N,4) [x,y,z,r] 행렬의 앞 3열을 stride 4 로 참조한다. 데이터 포인터가 같아야(복사 없음) 한다.
    Eigen::Matrix<float, Eigen::Dynamic, 4, Eigen::RowMajor> n4(3, 4);
    n4 << 0.f, 0.f, 10.f, 0.9f,
          1.f, 0.f, 10.f, 0.1f,
          0.f, -2.f, 5.f, 0.5f;
    const Eigen::Map<const pt3d::PointMatrix, 0, Eigen::OuterStride<>> view(n4.data(), 3, 3, Eigen::OuterStride<>(4));
    const pt3d::PointsRef ref(view);
    EXPECT_EQ(ref.data(), n4.data());
    EXPECT_EQ(ref.outerStride(), 4);

    const pt3d::PointMatrix n3 = n4.leftCols<3>();
    const auto P = pinhole(100.0, 600.0, 200.0);
    const auto a = pt3d::projectVeloToImage(ref, P, 375, 1242);
    const auto b = pt3d::projectVeloToImage(n3, P, 375, 1242);
    EXPECT_EQ(a.mask, b.mask);
    EXPECT_EQ(a.uv, b.uv);
    EXPECT_EQ(a.depth, b.depth);
}

TEST(Roi, InclusiveBoundaries) {
    const pt3d::RoiParams roi{0.0, 70.0, 40.0, -3.0, 3.0};
    const auto pts = points({
        {0.f, 0.f, 0.f},        // x = x_min       → 포함
        {70.f, 0.f, 0.f},       // x = x_max       → 포함
        {70.001f, 0.f, 0.f},    // x > x_max       → 제외
        {-0.001f, 0.f, 0.f},    // x < x_min       → 제외
        {10.f, 40.f, 0.f},      // y = +y_abs_max  → 포함
        {10.f, -40.f, 0.f},     // y = -y_abs_max  → 포함
        {10.f, -40.001f, 0.f},  // |y| > y_abs_max → 제외
        {10.f, 0.f, -3.f},      // z = z_min       → 포함
        {10.f, 0.f, 3.f},       // z = z_max       → 포함
        {10.f, 0.f, 3.001f},    // z > z_max       → 제외
    });
    EXPECT_EQ(pt3d::roiMask(pts, roi), u8({1, 1, 0, 0, 1, 1, 0, 1, 1, 0}));
}

TEST(Roi, ThresholdNotRepresentableInFloat32IsComparedInDouble) {
    // 0.1 은 float32 로 0.100000001490116 이다. 점 x = float32(0.1) 을 double 로 올리면 0.1000000015 > 0.1 이므로
    // x_max = 0.1 인 ROI 에서 제외되어야 한다 (Python: float64 캐스팅 후 비교). float32 로 비교하면 잘못 포함된다.
    const pt3d::RoiParams roi{0.0, 0.1, 40.0, -3.0, 3.0};
    const auto pts = points({{0.1f, 0.f, 0.f}, {0.09f, 0.f, 0.f}});
    EXPECT_EQ(pt3d::roiMask(pts, roi), u8({0, 1}));
}

TEST(Roi, EmptyInput) {
    const pt3d::PointMatrix pts(0, 3);
    EXPECT_TRUE(pt3d::roiMask(pts, pt3d::RoiParams{0.0, 70.0, 40.0, -3.0, 3.0}).empty());
}

TEST(Frustum, InclusiveBoxRequiresInImageAndInFront) {
    // (u, v) = (x, y) at z=1. W=100, H=50.
    const auto P = pinhole(1.0, 0.0, 0.0);
    const auto pts = points({
        {10.f, 10.f, 1.f},   // 박스 A 모서리 (10,10)    → A 포함
        {20.f, 20.f, 1.f},   // 박스 A 모서리 (20,20)    → A 포함 (상한 포함)
        {20.5f, 15.f, 1.f},  // u > 20                   → A 제외
        {15.f, 15.f, -1.f},  // 카메라 뒤                → 전부 제외
        {-1.f, 15.f, 1.f},   // 이미지 밖 (u<0)          → 박스 B 가 덮어도 제외
        {60.f, 30.f, 1.f},   // 박스 B 안                → B 포함
    });
    const std::vector<pt3d::Box2D> boxes{{10.0, 10.0, 20.0, 20.0}, {-5.0, 5.0, 70.0, 40.0}};
    const auto out = pt3d::frustumMasks(pts, P, 50, 100, boxes);
    ASSERT_EQ(out.size(), 2u * 6u);
    const std::vector<std::uint8_t> rowA(out.begin(), out.begin() + 6);
    const std::vector<std::uint8_t> rowB(out.begin() + 6, out.end());
    EXPECT_EQ(rowA, u8({1, 1, 0, 0, 0, 0}));
    EXPECT_EQ(rowB, u8({1, 1, 1, 0, 0, 1}));
}

TEST(Frustum, NoBoxesGivesEmptyOutput) {
    const auto pts = points({{10.f, 10.f, 1.f}});
    EXPECT_TRUE(pt3d::frustumMasks(pts, pinhole(1.0, 0.0, 0.0), 50, 100, {}).empty());
}
