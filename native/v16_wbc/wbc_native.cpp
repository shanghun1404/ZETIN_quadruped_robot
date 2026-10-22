#include <Eigen/Core>
#include <Eigen/Cholesky>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <cmath>
#include <cstring>
#include <string>

namespace py = pybind11;
using Array = py::array_t<double, py::array::c_style | py::array::forcecast>;
using Mask = py::array_t<bool, py::array::c_style | py::array::forcecast>;
template<int N> using Vector = Eigen::Matrix<double, N, 1>;
template<int R, int C> using RowMatrix = Eigen::Matrix<double, R, C, Eigen::RowMajor>;

template<int N> Vector<N> vector_input(const Array& a, const char* name) {
    if (a.ndim() != 1 || a.shape(0) != N)
        throw py::value_error(std::string(name) + " has an invalid shape");
    return Eigen::Map<const Vector<N>>(a.data());
}

template<int R, int C> RowMatrix<R, C> matrix_input(const Array& a, const char* name) {
    if (a.ndim() != 2 || a.shape(0) != R || a.shape(1) != C)
        throw py::value_error(std::string(name) + " has an invalid shape");
    return Eigen::Map<const RowMatrix<R, C>>(a.data());
}

template<int N> py::array_t<double> array_output(const Vector<N>& value) {
    py::array_t<double> result(N);
    std::memcpy(result.mutable_data(), value.data(), N * sizeof(double));
    return result;
}

py::tuple iterate(const Array& factor_input, const Array& constraint_input,
                  const Array& q_input, const Array& lower_input, const Array& upper_input,
                  const Mask& mask_input, const Array& x_input, const Array& z_input,
                  const Array& y_input, double rho, double relaxation, int iterations) {
    const auto factor = matrix_input<30, 30>(factor_input, "factor");
    const auto C = matrix_input<58, 30>(constraint_input, "constraints");
    const auto q = vector_input<30>(q_input, "q");
    const auto lower = vector_input<58>(lower_input, "lower");
    const auto upper = vector_input<58>(upper_input, "upper");
    auto x = vector_input<30>(x_input, "x");
    auto z = vector_input<58>(z_input, "z");
    auto y = vector_input<58>(y_input, "y");
    if (mask_input.ndim() != 1 || mask_input.shape(0) != 30)
        throw py::value_error("mask has an invalid shape");
    bool mask[30];
    std::memcpy(mask, mask_input.data(), sizeof(mask));
    if (!factor.allFinite() || !C.allFinite() || !q.allFinite() ||
        !x.allFinite() || !z.allFinite() || !y.allFinite() ||
        lower.array().isNaN().any() || upper.array().isNaN().any() ||
        (lower.array() > upper.array()).any() ||
        (factor.diagonal().array() <= 0.).any())
        throw py::value_error("invalid finite native QP inputs or bounds");
    if (!std::isfinite(rho) || rho <= 0. || !std::isfinite(relaxation) ||
        relaxation <= 0. || relaxation >= 2. || iterations <= 0)
        throw py::value_error("invalid native iteration parameters");
    {
        py::gil_scoped_release release;
        // Fixed-size stack workspaces; no Python calls or per-iteration allocations.
        const Eigen::Matrix<double, 30, 58> CT = C.transpose();
        Vector<30> rhs, forward;
        Vector<58> Cx, relaxed;
        for (int iteration = 0; iteration < iterations; ++iteration) {
            rhs.noalias() = -q + rho * CT * (z - y);
            forward = factor.triangularView<Eigen::Lower>().solve(rhs);
            x = factor.transpose().triangularView<Eigen::Upper>().solve(forward);
            for (int i = 0; i < 30; ++i)
                if (!mask[i]) x[i] = 0.;
            Cx.noalias() = C * x;
            relaxed = relaxation * Cx + (1. - relaxation) * z;
            z = (relaxed + y).cwiseMax(lower).cwiseMin(upper);
            y += relaxed - z;
        }
    }
    if (!x.allFinite() || !z.allFinite() || !y.allFinite())
        throw std::runtime_error("native QP iteration produced non-finite output");
    return py::make_tuple(array_output(x), array_output(z), array_output(y));
}

PYBIND11_MODULE(_v16_wbc_native, module) {
    module.doc() = "Fixed-size double-precision V16 ADMM iterations, preserving the reference algorithm";
    module.attr("__version__") = "1";
    module.def("iterate", &iterate, py::arg("factor"), py::arg("constraints"),
               py::arg("q"), py::arg("lower"), py::arg("upper"), py::arg("mask"),
               py::arg("x"), py::arg("z"), py::arg("y"), py::arg("rho"),
               py::arg("relaxation"), py::arg("iterations"));
}
