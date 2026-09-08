// pybind11 바인딩: numpy (N,3)/(N,4) float32 → pt3d::PointsRef (zero-copy), 결과 std::vector → numpy (복사 없음).
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <pybind11/eigen.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include "pt3d/projection.hpp"

namespace py = pybind11;

namespace {

// c_style: C 연속 배열만 허용(아니면 복사), forcecast: dtype 이 다르면 float32 로 변환(복사).
// 로더가 주는 float32 C 연속 (N,4) 는 두 조건을 이미 만족하므로 복사가 일어나지 않는다.
using PointsArray = py::array_t<float, py::array::c_style | py::array::forcecast>;
using BoxArray = py::array_t<double, py::array::c_style | py::array::forcecast>;

/// (N,3) 또는 (N,4) 배열의 앞 3열을 가리키는 stride 뷰. 데이터 포인터를 그대로 쓰므로 복사가 없다.
using StridedPointsMap = Eigen::Map<const pt3d::PointMatrix, 0, Eigen::OuterStride<>>;

StridedPointsMap asPointsMap(const PointsArray& a) {
    if (a.ndim() != 2 || (a.shape(1) != 3 && a.shape(1) != 4)) {
        std::string shape = "(";
        for (py::ssize_t i = 0; i < a.ndim(); ++i) {
            shape += std::to_string(a.shape(i)) + (i + 1 < a.ndim() ? ", " : "");
        }
        throw py::value_error("points 는 (N, 3) 또는 (N, 4) 이어야 합니다. 받은 shape: " + shape + ")");
    }
    return StridedPointsMap(a.data(), a.shape(0), 3, Eigen::OuterStride<>(a.shape(1)));
}

/// std::vector 의 버퍼를 numpy 배열에 그대로 넘긴다 (복사 없음).
/// 소유권: vector 를 힙으로 옮긴 뒤(unique_ptr) capsule 에 넘긴다. numpy 배열이 capsule 을 base 로 붙잡고,
/// 배열이 사라질 때 capsule 소멸자가 vector 를 delete 한다. 이 프로젝트에서 스마트 포인터가 실제로 필요한 유일한 곳.
template <typename T>
py::array_t<T> moveToNumpy(std::vector<T>&& vec, const std::vector<py::ssize_t>& shape) {
    auto owner = std::make_unique<std::vector<T>>(std::move(vec));
    T* data = owner->data();
    py::capsule base(owner.get(), [](void* p) { delete static_cast<std::vector<T>*>(p); });
    owner.release();  // 이제 capsule 이 소유한다
    return py::array_t<T>(shape, data, base);
}

}  // namespace

PYBIND11_MODULE(pt3d_cpp, m) {
    m.doc() = "PercepTrack3D Phase 10: LiDAR→image_02 투영 / ROI / frustum 마스크 (C++)";

    m.def(
        "project_velo_to_image",
        [](const PointsArray& points, const pt3d::ProjMatrix& P, int height, int width) {
            pt3d::ProjectionResult r = pt3d::projectVeloToImage(asPointsMap(points), P, height, width);
            const auto n = static_cast<py::ssize_t>(r.mask.size());
            const auto mcount = static_cast<py::ssize_t>(r.numProjected());
            return py::make_tuple(moveToNumpy(std::move(r.uv), {mcount, 2}), moveToNumpy(std::move(r.depth), {mcount}),
                                  moveToNumpy(std::move(r.mask), {n}));
        },
        py::arg("points"), py::arg("P"), py::arg("height"), py::arg("width"),
        "points (N,3)|(N,4) float32, P (3,4) float64, H, W → (uv (M,2) float32, depth (M,) float32, mask (N,) uint8)");

    m.def(
        "roi_mask",
        [](const PointsArray& points, double x_min, double x_max, double y_abs_max, double z_min, double z_max) {
            const pt3d::RoiParams roi{x_min, x_max, y_abs_max, z_min, z_max};
            std::vector<std::uint8_t> mask = pt3d::roiMask(asPointsMap(points), roi);
            const auto n = static_cast<py::ssize_t>(mask.size());
            return moveToNumpy(std::move(mask), {n});
        },
        py::arg("points"), py::arg("x_min"), py::arg("x_max"), py::arg("y_abs_max"), py::arg("z_min"), py::arg("z_max"),
        "points (N,3)|(N,4) float32 → mask (N,) uint8. 포함 경계.");

    m.def(
        "frustum_masks",
        [](const PointsArray& points, const pt3d::ProjMatrix& P, int height, int width, const BoxArray& boxes) {
            if (boxes.ndim() != 2 || boxes.shape(1) != 4) {
                throw py::value_error("boxes 는 (K, 4) [x1, y1, x2, y2] 이어야 합니다");
            }
            const auto k = boxes.shape(0);
            std::vector<pt3d::Box2D> bx;
            bx.reserve(static_cast<std::size_t>(k));
            auto b = boxes.unchecked<2>();
            for (py::ssize_t i = 0; i < k; ++i) {
                bx.push_back({b(i, 0), b(i, 1), b(i, 2), b(i, 3)});
            }
            const StridedPointsMap pts = asPointsMap(points);
            std::vector<std::uint8_t> out = pt3d::frustumMasks(pts, P, height, width, bx);
            return moveToNumpy(std::move(out), {k, static_cast<py::ssize_t>(pts.rows())});
        },
        py::arg("points"), py::arg("P"), py::arg("height"), py::arg("width"), py::arg("boxes"),
        "points, P, H, W, boxes (K,4) float64 → masks (K, N) uint8. 투영 1회 + 박스별 포함 경계 비교.");
}
