#include "bemt/airfoil.hpp"
#include <cmath>

namespace bemt {

namespace {

// ============================================================================
// Fritsch-Carlson 保单调一维导数计算辅助函数
// 给定节点坐标 x (长度 n) 与函数值 y (长度 n)，输出每个节点处的保单调一阶导数 d (长度 n)
// ============================================================================
void compute_monotone_slopes(const ArrayXd& x, const ArrayXd& y, ArrayXd& d) {
    const int n = static_cast<int>(x.size());
    d.resize(n);
    if (n <= 1) {
        d.setZero();
        return;
    }
    if (n == 2) {
        const double slope = (y[1] - y[0]) / (x[1] - x[0]);
        d[0] = slope;
        d[1] = slope;
        return;
    }

    ArrayXd h(n - 1);
    ArrayXd delta(n - 1);
    for (int i = 0; i < n - 1; ++i) {
        h[i]     = x[i + 1] - x[i];
        delta[i] = (y[i + 1] - y[i]) / h[i];
    }

    // 内部节点采用加权调和平均 (Brodlie 修改版 Fritsch-Carlson)，严格保证单调区间不过冲
    for (int i = 1; i < n - 1; ++i) {
        if (delta[i - 1] * delta[i] <= 0.0) {
            d[i] = 0.0; // 极值点处导数精确置零
        } else {
            const double w1 = 2.0 * h[i] + h[i - 1];
            const double w2 = h[i] + 2.0 * h[i - 1];
            d[i] = (w1 + w2) / (w1 / delta[i - 1] + w2 / delta[i]);
        }
    }

    // 端点采用三点非中心差分并施加保单调约束
    double d0 = ((2.0 * h[0] + h[1]) * delta[0] - h[0] * delta[1]) / (h[0] + h[1]);
    if (d0 * delta[0] <= 0.0) {
        d0 = 0.0;
    } else if ((delta[0] * delta[1] <= 0.0) && (std::abs(d0) > 3.0 * std::abs(delta[0]))) {
        d0 = 3.0 * delta[0];
    }
    d[0] = d0;

    double dn = ((2.0 * h[n - 2] + h[n - 3]) * delta[n - 2] - h[n - 2] * delta[n - 3]) / (h[n - 2] + h[n - 3]);
    if (dn * delta[n - 2] <= 0.0) {
        dn = 0.0;
    } else if ((delta[n - 2] * delta[n - 3] <= 0.0) && (std::abs(dn) > 3.0 * std::abs(delta[n - 2]))) {
        dn = 3.0 * delta[n - 2];
    }
    d[n - 1] = dn;
}

} // namespace

// ============================================================================
// Airfoil3D 构造函数与预计算实现
// ============================================================================
Airfoil3D::Airfoil3D(const ConstRefArrayXd& alpha_grid,
                     const ConstRefArrayXd& re_grid,
                     const ConstRefArrayXd& ma_grid,
                     const ConstRefArrayXd& cl_table,
                     const ConstRefArrayXd& cd_table,
                     double aspect_ratio,
                     bool enable_extrapolation)
    : n_a_(static_cast<int>(alpha_grid.size())),
      n_re_(static_cast<int>(re_grid.size())),
      n_ma_(static_cast<int>(ma_grid.size())),
      alpha_grid_(alpha_grid),
      ma_grid_(ma_grid),
      aspect_ratio_(aspect_ratio),
      enable_extrapolation_(enable_extrapolation) {

    if (n_a_ < 2) {
        throw std::invalid_argument("Airfoil3D: alpha_grid must contain at least 2 points.");
    }
    if (n_re_ < 1 || n_ma_ < 1) {
        throw std::invalid_argument("Airfoil3D: re_grid and ma_grid must contain at least 1 point.");
    }
    const int total_size = n_a_ * n_re_ * n_ma_;
    if (cl_table.size() != total_size || cd_table.size() != total_size) {
        throw std::invalid_argument("Airfoil3D: cl_table and cd_table size must equal N_a * N_re * N_ma.");
    }

    // 转换 Re 网格到 log10(Re)
    log_re_grid_.resize(n_re_);
    for (int i = 0; i < n_re_; ++i) {
        if (re_grid[i] <= 0.0) {
            throw std::invalid_argument("Airfoil3D: Reynolds numbers in re_grid must be strictly positive.");
        }
        log_re_grid_[i] = std::log10(re_grid[i]);
    }

    // Viterna-Corrigan 最大平板阻力系数经验公式
    const double ar_clamped = std::clamp(aspect_ratio_, 1.0, 50.0);
    cd_max_ = 1.11 + 0.018 * ar_clamped;

    build_hermite_jets(cl_table, cd_table);
}

Airfoil3D Airfoil3D::from_1d_polar(const ConstRefArrayXd& alpha_grid,
                                   const ConstRefArrayXd& cl_1d,
                                   const ConstRefArrayXd& cd_1d,
                                   double ref_re,
                                   double aspect_ratio,
                                   bool enable_extrapolation) {
    ArrayXd re_grid(1);
    re_grid[0] = ref_re;
    ArrayXd ma_grid(1);
    ma_grid[0] = 0.0;
    return Airfoil3D(alpha_grid, re_grid, ma_grid, cl_1d, cd_1d, aspect_ratio, enable_extrapolation);
}

void Airfoil3D::build_hermite_jets(const ConstRefArrayXd& cl_raw, const ConstRefArrayXd& cd_raw) {
    const int total = n_a_ * n_re_ * n_ma_;
    cl_jets_.resize(total);
    cd_jets_.resize(total);

    for (int k = 0; k < total; ++k) {
        cl_jets_[k].f = cl_raw[k];
        cd_jets_[k].f = cd_raw[k];
    }

    // 1. 沿迎角 alpha 维度计算保单调导数 d/d(alpha)
    ArrayXd f_line(n_a_), d_line(n_a_);
    for (int ire = 0; ire < n_re_; ++ire) {
        for (int ima = 0; ima < n_ma_; ++ima) {
            for (int ia = 0; ia < n_a_; ++ia) {
                f_line[ia] = cl_jets_[idx(ia, ire, ima)].f;
            }
            compute_monotone_slopes(alpha_grid_, f_line, d_line);
            for (int ia = 0; ia < n_a_; ++ia) {
                cl_jets_[idx(ia, ire, ima)].fa = d_line[ia];
            }

            for (int ia = 0; ia < n_a_; ++ia) {
                f_line[ia] = cd_jets_[idx(ia, ire, ima)].f;
            }
            compute_monotone_slopes(alpha_grid_, f_line, d_line);
            for (int ia = 0; ia < n_a_; ++ia) {
                cd_jets_[idx(ia, ire, ima)].fa = d_line[ia];
            }
        }
    }

    // 2. 沿 log10(Re) 维度计算保单调导数 d/d(log10_Re)
    if (n_re_ > 1) {
        f_line.resize(n_re_);
        d_line.resize(n_re_);
        for (int ia = 0; ia < n_a_; ++ia) {
            for (int ima = 0; ima < n_ma_; ++ima) {
                for (int ire = 0; ire < n_re_; ++ire) {
                    f_line[ire] = cl_jets_[idx(ia, ire, ima)].f;
                }
                compute_monotone_slopes(log_re_grid_, f_line, d_line);
                for (int ire = 0; ire < n_re_; ++ire) {
                    cl_jets_[idx(ia, ire, ima)].flre = d_line[ire];
                }

                for (int ire = 0; ire < n_re_; ++ire) {
                    f_line[ire] = cd_jets_[idx(ia, ire, ima)].f;
                }
                compute_monotone_slopes(log_re_grid_, f_line, d_line);
                for (int ire = 0; ire < n_re_; ++ire) {
                    cd_jets_[idx(ia, ire, ima)].flre = d_line[ire];
                }
            }
        }
    }

    // 3. 沿 Ma 维度计算保单调导数 d/d(Ma)
    if (n_ma_ > 1) {
        f_line.resize(n_ma_);
        d_line.resize(n_ma_);
        for (int ia = 0; ia < n_a_; ++ia) {
            for (int ire = 0; ire < n_re_; ++ire) {
                for (int ima = 0; ima < n_ma_; ++ima) {
                    f_line[ima] = cl_jets_[idx(ia, ire, ima)].f;
                }
                compute_monotone_slopes(ma_grid_, f_line, d_line);
                for (int ima = 0; ima < n_ma_; ++ima) {
                    cl_jets_[idx(ia, ire, ima)].fma = d_line[ima];
                }

                for (int ima = 0; ima < n_ma_; ++ima) {
                    f_line[ima] = cd_jets_[idx(ia, ire, ima)].f;
                }
                compute_monotone_slopes(ma_grid_, f_line, d_line);
                for (int ima = 0; ima < n_ma_; ++ima) {
                    cd_jets_[idx(ia, ire, ima)].fma = d_line[ima];
                }
            }
        }
    }

    // 4. 自动提取中间雷诺数、最低马赫数下的零升迎角 alpha_0 与小迎角线性斜率（用于 Du-Selig 3D 失速修正）
    const int mid_re = n_re_ / 2;
    const int low_ma = 0;

    // 寻找 alpha = 0 附近的斜率与零升迎角
    int ia_zero = 0;
    double min_abs_a = 1e9;
    for (int ia = 0; ia < n_a_; ++ia) {
        if (std::abs(alpha_grid_[ia]) < min_abs_a) {
            min_abs_a = std::abs(alpha_grid_[ia]);
            ia_zero   = ia;
        }
    }
    const double cl_near_0 = cl_jets_[idx(ia_zero, mid_re, low_ma)].f;
    const double slope_0   = cl_jets_[idx(ia_zero, mid_re, low_ma)].fa;
    cl_slope_linear_ = (slope_0 > 2.0 && slope_0 < 10.0) ? slope_0 : kTwoPi;
    alpha_zero_lift_ = std::clamp(alpha_grid_[ia_zero] - cl_near_0 / cl_slope_linear_, -0.25, 0.25);
}

void Airfoil3D::eval_batch(const ConstRefArrayXd& alpha_vec,
                           const ConstRefArrayXd& re_vec,
                           const ConstRefArrayXd& ma_vec,
                           RefArrayXd cl_out,
                           RefArrayXd cd_out) const {
    const Eigen::Index n = alpha_vec.size();
    for (Eigen::Index i = 0; i < n; ++i) {
        auto [cl, cd] = eval<double>(alpha_vec[i], re_vec[i], ma_vec[i]);
        cl_out[i]     = cl;
        cd_out[i]     = cd;
    }
}

} // namespace bemt