#pragma once

#include "bemt/dual.hpp"
#include "bemt/types.hpp"
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace bemt {

namespace bspline {

/**
 * @brief 根据控制点数量 n_ctrl 和阶数 degree 自动生成 [0, 1] 上的均匀夹紧节点向量 (Clamped Uniform Knots)
 *        节点总数 = n_ctrl + degree + 1：
 *        - 前 degree + 1 个节点恒为 0.0
 *        - 中间 n_ctrl - degree - 1 个内部节点在 (0, 1) 内均匀分布
 *        - 后 degree + 1 个节点恒为 1.0
 */
inline ArrayXd build_clamped_uniform_knots(int n_ctrl, int degree) {
    if (degree < 1) {
        throw std::invalid_argument("BSplineCurve: degree must be >= 1.");
    }
    if (n_ctrl < degree + 1) {
        throw std::invalid_argument(
            "BSplineCurve: number of control points n_ctrl (" + std::to_string(n_ctrl) +
            ") must be >= degree + 1 (" + std::to_string(degree + 1) + ").");
    }

    const int n_knots = n_ctrl + degree + 1;
    const int n_spans = n_ctrl - degree;
    ArrayXd knots(n_knots);

    for (int i = 0; i <= degree; ++i) {
        knots[i] = 0.0;
    }
    for (int j = 1; j < n_spans; ++j) {
        knots[degree + j] = static_cast<double>(j) / static_cast<double>(n_spans);
    }
    for (int i = n_ctrl; i < n_knots; ++i) {
        knots[i] = 1.0;
    }
    return knots;
}

/**
 * @brief 计算 [x_min, x_max] 区间上的 Greville 节点横坐标
 *        由 B 样条线性精确性，Greville 横坐标满足 sum_k x_greville[k] * N_{k,p}(u) == x_min + u * (x_max - x_min)
 */
inline ArrayXd build_greville_abscissae(int n_ctrl, int degree, double x_min = 0.0, double x_max = 1.0) {
    ArrayXd knots = build_clamped_uniform_knots(n_ctrl, degree);
    ArrayXd x_ctrl(n_ctrl);
    for (int k = 0; k < n_ctrl; ++k) {
        double sum_u = 0.0;
        for (int m = 1; m <= degree; ++m) {
            sum_u += knots[k + m];
        }
        const double xi_k = sum_u / static_cast<double>(degree);
        x_ctrl[k] = x_min + xi_k * (x_max - x_min);
    }
    return x_ctrl;
}

/**
 * @brief 在给定参数 u in [0, 1] 处计算全部 n_ctrl 个 B 样条基函数值 N_{k,p}(u)（兼容 double 与 Dual1 自动微分）
 */
template <typename Scalar>
inline void eval_basis_row_scalar(const Scalar& u,
                                  const ArrayXd& knots,
                                  int n_ctrl,
                                  int degree,
                                  std::vector<Scalar>& basis_row) {
    basis_row.assign(n_ctrl, Scalar(0.0));
    const int n_spans = n_ctrl - degree;
    const double uv   = std::clamp(val_of(u), 0.0, 1.0);

    int span = degree;
    if (uv >= 1.0 - 1e-15) {
        span = n_ctrl - 1;
    } else if (uv <= 1e-15) {
        span = degree;
    } else {
        span = std::clamp(static_cast<int>(std::floor(uv * n_spans)) + degree, degree, n_ctrl - 1);
        while (span < n_ctrl - 1 && uv >= knots[span + 1]) {
            ++span;
        }
        while (span > degree && uv < knots[span]) {
            --span;
        }
    }

    std::vector<Scalar> N(degree + 1, Scalar(0.0));
    std::vector<Scalar> left(degree + 1, Scalar(0.0));
    std::vector<Scalar> right(degree + 1, Scalar(0.0));

    N[0] = Scalar(1.0);
    for (int j = 1; j <= degree; ++j) {
        left[j]  = u - knots[span + 1 - j];
        right[j] = knots[span + j] - u;
        Scalar saved(0.0);
        for (int k = 0; k < j; ++k) {
            const double denom = knots[span + k + 1] - knots[span + 1 - j + k];
            const Scalar temp  = (denom > 1e-15) ? (N[k] / denom) : Scalar(0.0);
            N[k]  = saved + right[k + 1] * temp;
            saved = left[j - k] * temp;
        }
        N[j] = saved;
    }

    for (int k = 0; k <= degree; ++k) {
        basis_row[span - degree + k] = N[k];
    }
}

/**
 * @brief 针对二维参数化 B 样条曲线 (x(u), y(u)) = sum_k N_{k,p}(u) * (x_ctrl[k], y_ctrl[k])，
 *        已知目标无量纲径向站位 target_x = r / R，通过牛顿-二分混合法反解 x(u_i*) = target_x[i]，
 *        构建基函数求值矩阵 B (维度 N_r x n_ctrl，满足 y(r_i) = (B * y_ctrl)[i])。
 */
inline MatrixXd build_parametric_basis_matrix(const ConstRefArrayXd& target_x,
                                              const ConstRefArrayXd& x_ctrl,
                                              int degree,
                                              ArrayXd* out_knots = nullptr,
                                              ArrayXd* out_u_params = nullptr) {
    const int n_ctrl = static_cast<int>(x_ctrl.size());
    ArrayXd knots = build_clamped_uniform_knots(n_ctrl, degree);
    for (int k = 0; k < n_ctrl - 1; ++k) {
        if (x_ctrl[k + 1] < x_ctrl[k] - 1e-12) {
            throw std::invalid_argument("BSplineCurve: control point r/R abscissae ctrl_r_norm must be non-decreasing.");
        }
    }
    if (x_ctrl[n_ctrl - 1] <= x_ctrl[0] + 1e-12) {
        throw std::invalid_argument("BSplineCurve: control point r/R span must satisfy ctrl_r_norm[-1] > ctrl_r_norm[0].");
    }

    const int nr = static_cast<int>(target_x.size());
    MatrixXd basis = MatrixXd::Zero(nr, n_ctrl);
    if (out_u_params != nullptr) {
        out_u_params->resize(nr);
    }

    const double x_min = x_ctrl[0];
    const double x_max = x_ctrl[n_ctrl - 1];
    std::vector<Dual1> row_d1;
    std::vector<double> row_val;

    for (int i = 0; i < nr; ++i) {
        const double x_t = std::clamp(target_x[i], x_min, x_max);
        double u_lo = 0.0;
        double u_hi = 1.0;
        double u_curr = std::clamp((x_t - x_min) / (x_max - x_min), 0.0, 1.0);

        if (x_t <= x_min + 1e-15) {
            u_curr = 0.0;
        } else if (x_t >= x_max - 1e-15) {
            u_curr = 1.0;
        } else {
            for (int iter = 0; iter < 20; ++iter) {
                eval_basis_row_scalar<Dual1>(Dual1(u_curr, 1.0), knots, n_ctrl, degree, row_d1);
                double x_val = 0.0;
                double dx_du = 0.0;
                for (int k = 0; k < n_ctrl; ++k) {
                    x_val += row_d1[k].val * x_ctrl[k];
                    dx_du += row_d1[k].grad(0) * x_ctrl[k];
                }
                const double err = x_val - x_t;
                if (std::abs(err) <= 1e-15) {
                    break;
                }
                if (err < 0.0) {
                    u_lo = u_curr;
                } else {
                    u_hi = u_curr;
                }
                if (dx_du > 1e-12) {
                    const double u_newton = u_curr - err / dx_du;
                    if (u_newton > u_lo && u_newton < u_hi) {
                        u_curr = u_newton;
                    } else {
                        u_curr = 0.5 * (u_lo + u_hi);
                    }
                } else {
                    u_curr = 0.5 * (u_lo + u_hi);
                }
            }
        }

        if (out_u_params != nullptr) {
            (*out_u_params)[i] = u_curr;
        }
        eval_basis_row_scalar<double>(u_curr, knots, n_ctrl, degree, row_val);
        for (int k = 0; k < n_ctrl; ++k) {
            basis(i, k) = row_val[k];
        }
    }

    if (out_knots != nullptr) {
        *out_knots = std::move(knots);
    }
    return basis;
}

inline MatrixXd build_basis_matrix(const ConstRefArrayXd& r,
                                   double hub_radius,
                                   double radius,
                                   int n_ctrl,
                                   int degree,
                                   ArrayXd* out_knots = nullptr) {
    const double x_hub = hub_radius / std::max(radius, 1e-12);
    ArrayXd x_ctrl = build_greville_abscissae(n_ctrl, degree, x_hub, 1.0);
    ArrayXd target_x = r / std::max(radius, 1e-12);
    return build_parametric_basis_matrix(target_x, x_ctrl, degree, out_knots, nullptr);
}

inline ArrayXd compute_parametric_slope_dy_dx(const ConstRefArrayXd& u_params,
                                              const ConstRefArrayXd& x_ctrl,
                                              const ConstRefArrayXd& y_ctrl,
                                              const ArrayXd& knots,
                                              int degree) {
    const int nr     = static_cast<int>(u_params.size());
    const int n_ctrl = static_cast<int>(x_ctrl.size());
    ArrayXd slope    = ArrayXd::Zero(nr);
    std::vector<Dual1> row_d1;

    for (int i = 0; i < nr; ++i) {
        eval_basis_row_scalar<Dual1>(Dual1(u_params[i], 1.0), knots, n_ctrl, degree, row_d1);
        double dx_du = 0.0;
        double dy_du = 0.0;
        for (int k = 0; k < n_ctrl; ++k) {
            dx_du += row_d1[k].grad(0) * x_ctrl[k];
            dy_du += row_d1[k].grad(0) * y_ctrl[k];
        }
        slope[i] = (std::abs(dx_du) > 1e-12) ? (dy_du / dx_du) : 0.0;
    }
    return slope;
}

} // namespace bspline

// ============================================================================
// 独立的展向分布二维参数化 B 样条曲线类 BSplineCurve
// 数学形式: (r/R(u), val(u)) = sum_{k=0}^{N_ctrl-1} N_{k,p}(u) * (ctrl_r_norm[k], ctrl_vals[k]), u in [0, 1]
// 用途: 可在 Python 中单独实例化弦长曲线 chord_curve 与扭转曲线 twist_curve，
//       支持独立采样绘图、按 r/R 反求求值、获取基函数矩阵，并直接传入 RotorGeometry！
// ============================================================================
class BSplineCurve {
public:
    BSplineCurve() = default;

    /**
     * @brief 构造二维参数化 B 样条曲线: 给定控制点的无量纲径向位置 ctrl_r_norm 与物理量值 ctrl_vals
     */
    BSplineCurve(const ConstRefArrayXd& ctrl_r_norm,
                 const ConstRefArrayXd& ctrl_vals,
                 int degree = 3)
        : degree_(degree),
          ctrl_r_norm_(ctrl_r_norm),
          ctrl_vals_(ctrl_vals) {
        validate_and_build();
    }

    /**
     * @brief 由二维控制点矩阵 ctrl_points (形状 N_ctrl x 2，第 0 列为 r/R，第 1 列为物理量值) 构造
     */
    explicit BSplineCurve(const ConstRefMatrixXd& ctrl_points, int degree = 3)
        : degree_(degree) {
        if (ctrl_points.cols() != 2) {
            throw std::invalid_argument("BSplineCurve: ctrl_points matrix must have shape (N_ctrl, 2) with columns [r/R, value].");
        }
        const Eigen::Index nc = ctrl_points.rows();
        ctrl_r_norm_.resize(nc);
        ctrl_vals_.resize(nc);
        for (Eigen::Index k = 0; k < nc; ++k) {
            ctrl_r_norm_[k] = ctrl_points(k, 0);
            ctrl_vals_[k]   = ctrl_points(k, 1);
        }
        validate_and_build();
    }

    /**
     * @brief 便捷静态构造方法：仅给定一维控制点纵坐标 ctrl_vals，横坐标自动按 [r_norm_min, r_norm_max] 上的 Greville 节点生成
     */
    static BSplineCurve from_1d_values(const ConstRefArrayXd& ctrl_vals,
                                       int degree = 3,
                                       double r_norm_min = 0.0,
                                       double r_norm_max = 1.0) {
        const int nc = static_cast<int>(ctrl_vals.size());
        ArrayXd x_ctrl = bspline::build_greville_abscissae(nc, degree, r_norm_min, r_norm_max);
        return BSplineCurve(x_ctrl, ctrl_vals, degree);
    }

    // ------------------------------------------------------------------------
    // 1. 按曲线参数 u in [0, 1] 直接正向求值與均匀采样（专用于曲线检查与可视化绘图）
    // ------------------------------------------------------------------------
    std::pair<double, double> eval_u(double u) const {
        const int nc = static_cast<int>(ctrl_vals_.size());
        std::vector<double> row;
        bspline::eval_basis_row_scalar<double>(std::clamp(u, 0.0, 1.0), knots_, nc, degree_, row);
        double x = 0.0;
        double y = 0.0;
        for (int k = 0; k < nc; ++k) {
            x += row[k] * ctrl_r_norm_[k];
            y += row[k] * ctrl_vals_[k];
        }
        return {x, y};
    }

    std::pair<ArrayXd, ArrayXd> eval_u_batch(const ConstRefArrayXd& u_vec) const {
        const Eigen::Index n = u_vec.size();
        ArrayXd x_out(n), y_out(n);
        for (Eigen::Index i = 0; i < n; ++i) {
            auto [x, y] = eval_u(u_vec[i]);
            x_out[i]    = x;
            y_out[i]    = y;
        }
        return {x_out, y_out};
    }

    /**
     * @brief 在参数域 u in [0, 1] 上等间距采样 n_points 个点，返回 (r_norm_array, val_array)，一行代码即可在 Python 中绘图
     */
    std::pair<ArrayXd, ArrayXd> sample(int n_points = 200) const {
        const int n = std::max(n_points, 2);
        ArrayXd u_vec(n);
        for (int i = 0; i < n; ++i) {
            u_vec[i] = static_cast<double>(i) / static_cast<double>(n - 1);
        }
        return eval_u_batch(u_vec);
    }

    // ------------------------------------------------------------------------
    // 2. 按无量纲径向位置 r/R 反求曲线参数 u* 并计算对应物理量值与导数
    // ------------------------------------------------------------------------
    double eval_at_r_norm(double target_r_norm) const {
        ArrayXd target_arr(1);
        target_arr[0] = target_r_norm;
        MatrixXd B = basis_matrix(target_arr);
        double val = 0.0;
        for (Eigen::Index k = 0; k < ctrl_vals_.size(); ++k) {
            val += B(0, k) * ctrl_vals_[k];
        }
        return val;
    }

    ArrayXd eval_batch(const ConstRefArrayXd& target_r_norm) const {
        MatrixXd B = basis_matrix(target_r_norm);
        return (B * ctrl_vals_.matrix()).array();
    }

    ArrayXd invert_r_norm(const ConstRefArrayXd& target_r_norm) const {
        ArrayXd u_params;
        (void)bspline::build_parametric_basis_matrix(target_r_norm, ctrl_r_norm_, degree_, nullptr, &u_params);
        return u_params;
    }

    MatrixXd basis_matrix(const ConstRefArrayXd& target_r_norm, ArrayXd* out_u_params = nullptr) const {
        return bspline::build_parametric_basis_matrix(target_r_norm, ctrl_r_norm_, degree_, nullptr, out_u_params);
    }

    ArrayXd slope_dy_dx(const ConstRefArrayXd& target_r_norm) const {
        ArrayXd u_params = invert_r_norm(target_r_norm);
        return bspline::compute_parametric_slope_dy_dx(u_params, ctrl_r_norm_, ctrl_vals_, knots_, degree_);
    }

    // ------------------------------------------------------------------------
    // 3. 元数据与控制点访问器
    // ------------------------------------------------------------------------
    bool empty() const { return ctrl_vals_.size() == 0; }
    int degree() const { return degree_; }
    int num_ctrl_points() const { return static_cast<int>(ctrl_vals_.size()); }
    const ArrayXd& knots() const { return knots_; }
    const ArrayXd& ctrl_r_norm() const { return ctrl_r_norm_; }
    const ArrayXd& ctrl_vals() const { return ctrl_vals_; }

    MatrixXd ctrl_points() const {
        const Eigen::Index nc = ctrl_vals_.size();
        MatrixXd mat(nc, 2);
        for (Eigen::Index k = 0; k < nc; ++k) {
            mat(k, 0) = ctrl_r_norm_[k];
            mat(k, 1) = ctrl_vals_[k];
        }
        return mat;
    }

private:
    int degree_ = 3;
    ArrayXd ctrl_r_norm_;
    ArrayXd ctrl_vals_;
    ArrayXd knots_;

    void validate_and_build() {
        const int nc = static_cast<int>(ctrl_vals_.size());
        if (ctrl_r_norm_.size() != nc) {
            throw std::invalid_argument("BSplineCurve: ctrl_r_norm and ctrl_vals must have the same length.");
        }
        knots_ = bspline::build_clamped_uniform_knots(nc, degree_);
        for (int k = 0; k < nc - 1; ++k) {
            if (ctrl_r_norm_[k + 1] < ctrl_r_norm_[k] - 1e-12) {
                throw std::invalid_argument("BSplineCurve: ctrl_r_norm must be non-decreasing.");
            }
        }
        if (ctrl_r_norm_[nc - 1] <= ctrl_r_norm_[0] + 1e-12) {
            throw std::invalid_argument("BSplineCurve: ctrl_r_norm span must satisfy ctrl_r_norm[-1] > ctrl_r_norm[0].");
        }
    }
};

} // namespace bemt
