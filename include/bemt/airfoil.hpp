#pragma once

#include "bemt/dual.hpp"
#include "bemt/types.hpp"
#include <algorithm>
#include <cmath>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace bemt {

// ============================================================================
// 三维翼型极曲线插值器 Airfoil3D
// 维度约定: (alpha [rad], Re [-], Ma [-]) -> (Cl, Cd)
// 支持标量 double、自动微分 Dual<N>、批量向量化求值以及展向多翼型截面自动线性融合插值
// ============================================================================
class Airfoil3D {
public:
    Airfoil3D() = default;

    /**
     * @brief 构造函数：输入三维张量网格数据并预计算保单调三次 Hermite 导数表
     *
     * @param alpha_grid  迎角网格向量 [rad]，严格递增，长度 N_a >= 2
     * @param re_grid     雷诺数网格向量 [-]，严格递增 (>0)，长度 N_re >= 1
     * @param ma_grid     马赫数网格向量 [-]，严格递增 (>=0)，长度 N_ma >= 1
     * @param cl_table    展平的一维 Cl 数组，按 Row-Major (i_a * N_re * N_ma + i_re * N_ma + i_ma) 排列
     * @param cd_table    展平的一维 Cd 数组，排列方式同 cl_table
     * @param aspect_ratio 展弦比参数（用于 Viterna-Corrigan 大迎角外推，默认 10.0）
     * @param enable_extrapolation 是否启用超出网格范围时的 Viterna-Corrigan [-pi, pi] 全迎角平滑外推
     */
    Airfoil3D(const ConstRefArrayXd& alpha_grid,
              const ConstRefArrayXd& re_grid,
              const ConstRefArrayXd& ma_grid,
              const ConstRefArrayXd& cl_table,
              const ConstRefArrayXd& cd_table,
              double aspect_ratio = 10.0,
              bool enable_extrapolation = true);

    /**
     * @brief 便捷静态构造方法：由一维极曲线 (Cl(alpha), Cd(alpha)) 构造
     */
    static Airfoil3D from_1d_polar(const ConstRefArrayXd& alpha_grid,
                                   const ConstRefArrayXd& cl_1d,
                                   const ConstRefArrayXd& cd_1d,
                                   double ref_re = 1e6,
                                   double aspect_ratio = 10.0,
                                   bool enable_extrapolation = true);

    /**
     * @brief 展向两相邻截面翼型表线性插值融合: (1 - weight) * af0 + weight * af1
     *        若两翼型网格相同，直接在线性空间融合 Hermite 导数表（O(1) 零损耗）；
     *        若网格不同，自动在合并网格上重采样构造新表。
     */
    static std::shared_ptr<const Airfoil3D> interpolate_section(
        const std::shared_ptr<const Airfoil3D>& af0,
        const std::shared_ptr<const Airfoil3D>& af1,
        double weight);

    /**
     * @brief 将有限个控制截面的翼型表 (section_r_norm, section_airfoils) 自动插值映射到 N_r 个离散叶素站位 r_norm = r / R 上
     */
    static std::vector<std::shared_ptr<const Airfoil3D>> interpolate_spanwise_sections(
        const ConstRefArrayXd& section_r_norm,
        const std::vector<std::shared_ptr<const Airfoil3D>>& section_airfoils,
        const ConstRefArrayXd& target_r_norm);

    // ------------------------------------------------------------------------
    // 核心泛型求值接口：同时兼容 double 与 Dual<N>
    // 输入: alpha [rad], re [-], ma [-]
    // 输出: std::pair<Scalar, Scalar> -> {Cl, Cd}
    // ------------------------------------------------------------------------
    template <typename Scalar>
    inline std::pair<Scalar, Scalar> eval(const Scalar& alpha,
                                          const Scalar& re,
                                          const Scalar& ma) const {
        using std::cos;
        using std::log10;
        using std::sin;

        const Scalar safe_re     = math::smooth_max(re, Scalar(100.0), 1.0);
        const Scalar log_re      = log10(safe_re);
        const Scalar lre_clamped = (n_re_ > 1)
                                       ? math::smooth_clamp(log_re, log_re_grid_[0], log_re_grid_[n_re_ - 1], 1e-4)
                                       : Scalar(log_re_grid_[0]);
        const Scalar ma_clamped  = (n_ma_ > 1)
                                       ? math::smooth_clamp(ma, ma_grid_[0], ma_grid_[n_ma_ - 1], 1e-4)
                                       : Scalar(ma_grid_[0]);

        const double a_min = alpha_grid_[0];
        const double a_max = alpha_grid_[n_a_ - 1];
        const double blend_width = std::min(0.8 * kDeg2Rad, 0.05 * (a_max - a_min));
        const double a_val = val_of(alpha);

        // 1. 快速通道：禁用外推，或迎角在表格内部 [a_min + 8*w, a_max - 8*w] 时，sigmoid 权重为 1.0 (误差 < 3e-4)
        if (!enable_extrapolation_) {
            const Scalar alpha_clamped = math::smooth_clamp(alpha, a_min, a_max, 1e-4);
            return eval_tricubic_table(alpha_clamped, lre_clamped, ma_clamped);
        }
        if (a_val >= a_min + 8.0 * blend_width && a_val <= a_max - 8.0 * blend_width) {
            return eval_tricubic_table(alpha, lre_clamped, ma_clamped);
        }

        // 2. 计算高低失速边界点处的气动系数
        auto [cl_hi, cd_hi] = eval_tricubic_table(Scalar(a_max), lre_clamped, ma_clamped);
        auto [cl_lo, cd_lo] = eval_tricubic_table(Scalar(a_min), lre_clamped, ma_clamped);

        auto [cl_vc, cd_vc] = eval_viterna_corrigan(alpha, cl_hi, cd_hi, cl_lo, cd_lo);

        // 若完全超出过渡带 (> 8*w)，直接返回纯 Viterna-Corrigan 外推
        if (a_val <= a_min - 8.0 * blend_width || a_val >= a_max + 8.0 * blend_width) {
            return {cl_vc, cd_vc};
        }

        // 3. 在边界过渡带内：用 smooth_clamp 代替硬 clamp_coord，彻底消除 a_min/a_max 处的导数断崖！
        const Scalar alpha_smooth = math::smooth_clamp(alpha, a_min, a_max, 0.5 * blend_width);
        auto [cl_tbl, cd_tbl] = eval_tricubic_table(alpha_smooth, lre_clamped, ma_clamped);

        const Scalar w_low  = math::sigmoid_weight(alpha, a_min, blend_width);
        const Scalar w_high = Scalar(1.0) - math::sigmoid_weight(alpha, a_max, blend_width);
        const Scalar w_in   = w_low * w_high;

        return {w_in * cl_tbl + (Scalar(1.0) - w_in) * cl_vc,
                w_in * cd_tbl + (Scalar(1.0) - w_in) * cd_vc};
    }

    // ------------------------------------------------------------------------
    // 批量向量化求值接口（针对同一径向站位上 N_psi 个方位角微元同时求值）
    // ------------------------------------------------------------------------
    void eval_batch(const ConstRefArrayXd& alpha_vec,
                    const ConstRefArrayXd& re_vec,
                    const ConstRefArrayXd& ma_vec,
                    RefArrayXd cl_out,
                    RefArrayXd cd_out) const;

    // ------------------------------------------------------------------------
    // 获取零升迎角 alpha_0 与线性段升力线斜率 dCl/dalpha（供 Du-Selig 3D 失速延迟修正调用）
    // ------------------------------------------------------------------------
    double zero_lift_alpha() const { return alpha_zero_lift_; }
    double linear_lift_slope() const { return cl_slope_linear_; }
    double alpha_min() const { return alpha_grid_.size() > 0 ? alpha_grid_[0] : -0.35; }
    double alpha_max() const { return alpha_grid_.size() > 0 ? alpha_grid_[n_a_ - 1] : 0.35; }

private:
    // 网格节点数据结构：存储每个 (i_a, i_re, i_ma) 节点的原值与一阶导数，实现极速局部 Hermite 插值
    struct NodeJet {
        double f    = 0.0; // 函数值
        double fa   = 0.0; // d/d(alpha)
        double flre = 0.0; // d/d(log10_Re)
        double fma  = 0.0; // d/d(Ma)
    };

    int n_a_  = 0;
    int n_re_ = 0;
    int n_ma_ = 0;

    ArrayXd alpha_grid_;
    ArrayXd log_re_grid_;
    ArrayXd ma_grid_;

    std::vector<NodeJet> cl_jets_;
    std::vector<NodeJet> cd_jets_;

    double aspect_ratio_         = 10.0;
    double cd_max_               = 1.98;
    double alpha_zero_lift_      = 0.0;
    double cl_slope_linear_      = kTwoPi;
    bool   enable_extrapolation_ = true;

    // 三维展平索引
    inline int idx(int ia, int ire, int ima) const {
        return (ia * n_re_ + ire) * n_ma_ + ima;
    }

    // 预计算 Fritsch-Carlson 保单调导数与零升迎角特征
    void build_hermite_jets(const ConstRefArrayXd& cl_raw, const ConstRefArrayXd& cd_raw);

    // 二分查找所在网格区间索引 i (满足 grid[i] <= x <= grid[i+1]) 及归一化坐标 t in [0, 1]
    template <typename Scalar>
    inline void locate_interval(const Scalar& x,
                                const ArrayXd& grid,
                                int& i_cell,
                                Scalar& t,
                                double& h) const {
        const int n = static_cast<int>(grid.size());
        if (n <= 1) {
            i_cell = 0;
            t      = Scalar(0.0);
            h      = 1.0;
            return;
        }
        const double xv = val_of(x);
        if (xv <= grid[0]) {
            i_cell = 0;
        } else if (xv >= grid[n - 1]) {
            i_cell = n - 2;
        } else {
            auto it = std::upper_bound(grid.data(), grid.data() + n, xv);
            i_cell  = std::clamp(static_cast<int>(it - grid.data()) - 1, 0, n - 2);
        }
        h = grid[i_cell + 1] - grid[i_cell];
        t = (x - grid[i_cell]) / h;
    }

    // 1D 三次 Hermite 基函数求值: 给定左端点 (f0, d0) 与右端点 (f1, d1)，步长 h，归一化坐标 t in [0,1]
    template <typename Scalar>
    static inline Scalar hermite_1d(const Scalar& t,
                                    double h,
                                    const Scalar& f0,
                                    const Scalar& d0,
                                    const Scalar& f1,
                                    const Scalar& d1) {
        const Scalar t2  = t * t;
        const Scalar t3  = t2 * t;
        const Scalar h00 = 2.0 * t3 - 3.0 * t2 + 1.0;
        const Scalar h10 = t3 - 2.0 * t2 + t;
        const Scalar h01 = -2.0 * t3 + 3.0 * t2;
        const Scalar h11 = t3 - t2;
        return h00 * f0 + (h10 * h) * d0 + h01 * f1 + (h11 * h) * d1;
    }

    // 在三维网格内部执行三立方 Hermite 插值 (保证跨 (alpha, Re, Ma) 网格线的 C^1 值与一阶导数连续性)
    template <typename Scalar>
    inline std::pair<Scalar, Scalar> eval_tricubic_table(const Scalar& alpha,
                                                         const Scalar& log_re,
                                                         const Scalar& ma) const {
        int ia = 0, ire = 0, ima = 0;
        Scalar ta, tlre, tma;
        double ha = 1.0, hlre = 1.0, hma = 1.0;

        locate_interval(alpha, alpha_grid_, ia, ta, ha);
        locate_interval(log_re, log_re_grid_, ire, tlre, hlre);
        locate_interval(ma, ma_grid_, ima, tma, hma);

        const Scalar sa   = ta * ta * (3.0 - 2.0 * ta);
        const Scalar slre = tlre * tlre * (3.0 - 2.0 * tlre);

        auto interp_alpha_col = [&](const std::vector<NodeJet>& jets, int r_idx, int m_idx) -> Scalar {
            const NodeJet& n0 = jets[idx(ia, r_idx, m_idx)];
            const NodeJet& n1 = jets[idx(ia + 1, r_idx, m_idx)];
            return hermite_1d(ta, ha, Scalar(n0.f), Scalar(n0.fa), Scalar(n1.f), Scalar(n1.fa));
        };

        if (n_re_ == 1 && n_ma_ == 1) {
            return {interp_alpha_col(cl_jets_, 0, 0), interp_alpha_col(cd_jets_, 0, 0)};
        }

        auto interp_re_plane = [&](const std::vector<NodeJet>& jets, int m_idx) -> Scalar {
            if (n_re_ == 1) {
                return interp_alpha_col(jets, 0, m_idx);
            }
            const Scalar f_re0 = interp_alpha_col(jets, ire, m_idx);
            const Scalar f_re1 = interp_alpha_col(jets, ire + 1, m_idx);

            const double d_re0_a0 = jets[idx(ia, ire, m_idx)].flre;
            const double d_re0_a1 = jets[idx(ia + 1, ire, m_idx)].flre;
            const Scalar d_re0    = (1.0 - sa) * d_re0_a0 + sa * d_re0_a1;

            const double d_re1_a0 = jets[idx(ia, ire + 1, m_idx)].flre;
            const double d_re1_a1 = jets[idx(ia + 1, ire + 1, m_idx)].flre;
            const Scalar d_re1    = (1.0 - sa) * d_re1_a0 + sa * d_re1_a1;

            return hermite_1d(tlre, hlre, f_re0, d_re0, f_re1, d_re1);
        };

        if (n_ma_ == 1) {
            return {interp_re_plane(cl_jets_, 0), interp_re_plane(cd_jets_, 0)};
        }

        auto interp_full_3d = [&](const std::vector<NodeJet>& jets) -> Scalar {
            const Scalar f_ma0 = interp_re_plane(jets, ima);
            const Scalar f_ma1 = interp_re_plane(jets, ima + 1);

            Scalar d_ma0, d_ma1;
            if (n_re_ == 1) {
                d_ma0 = (1.0 - sa) * jets[idx(ia, 0, ima)].fma + sa * jets[idx(ia + 1, 0, ima)].fma;
                d_ma1 = (1.0 - sa) * jets[idx(ia, 0, ima + 1)].fma + sa * jets[idx(ia + 1, 0, ima + 1)].fma;
            } else {
                const Scalar d_ma0_re0 = (1.0 - sa) * jets[idx(ia, ire, ima)].fma +
                                         sa * jets[idx(ia + 1, ire, ima)].fma;
                const Scalar d_ma0_re1 = (1.0 - sa) * jets[idx(ia, ire + 1, ima)].fma +
                                         sa * jets[idx(ia + 1, ire + 1, ima)].fma;
                d_ma0 = (1.0 - slre) * d_ma0_re0 + slre * d_ma0_re1;

                const Scalar d_ma1_re0 = (1.0 - sa) * jets[idx(ia, ire, ima + 1)].fma +
                                         sa * jets[idx(ia + 1, ire, ima + 1)].fma;
                const Scalar d_ma1_re1 = (1.0 - sa) * jets[idx(ia, ire + 1, ima + 1)].fma +
                                         sa * jets[idx(ia + 1, ire + 1, ima + 1)].fma;
                d_ma1 = (1.0 - slre) * d_ma1_re0 + slre * d_ma1_re1;
            }

            return hermite_1d(tma, hma, f_ma0, d_ma0, f_ma1, d_ma1);
        };

        return {interp_full_3d(cl_jets_), interp_full_3d(cd_jets_)};
    }

    // Viterna-Corrigan 全迎角 [-pi, pi] 外推核函数
    template <typename Scalar>
    inline std::pair<Scalar, Scalar> eval_viterna_corrigan(const Scalar& alpha,
                                                           const Scalar& cl_hi,
                                                           const Scalar& cd_hi,
                                                           const Scalar& cl_lo,
                                                           const Scalar& cd_lo) const {
        using std::cos;
        using std::sin;

        const double a_hi = std::clamp(alpha_grid_[n_a_ - 1], 0.15, 0.75);
        const double a_lo = std::clamp(alpha_grid_[0], -0.75, -0.15);

        const double sin_hi = std::sin(a_hi);
        const double cos_hi = std::cos(a_hi);
        const double sin_lo = std::sin(a_lo);
        const double cos_lo = std::cos(a_lo);

        // 正迎角侧 Viterna 系数
        const Scalar B1_pos = Scalar(cd_max_);
        const Scalar B2_pos = (cd_hi - cd_max_ * sin_hi * sin_hi) / std::max(cos_hi, 0.1);
        const Scalar A1_pos = 0.5 * B1_pos;
        const Scalar A2_pos = (cl_hi - cd_max_ * sin_hi * cos_hi) * (sin_hi / std::max(cos_hi * cos_hi, 0.05));

        // 负迎角侧 Viterna 系数
        const Scalar B1_neg = Scalar(cd_max_);
        const Scalar B2_neg = (cd_lo - cd_max_ * sin_lo * sin_lo) / std::max(cos_lo, 0.1);
        const Scalar A1_neg = 0.5 * B1_neg;
        const Scalar A2_neg = (cl_lo - cd_max_ * sin_lo * cos_lo) * (std::abs(sin_lo) / std::max(cos_lo * cos_lo, 0.05));

        const Scalar sin_a     = sin(alpha);
        const Scalar cos_a     = cos(alpha);
        const Scalar sin_2a    = 2.0 * sin_a * cos_a;
        const Scalar sin_a_abs = math::smooth_abs(sin_a, 1e-4);
        const Scalar cos_a_abs = math::smooth_abs(cos_a, 1e-4);

        const Scalar w_pos = math::sigmoid_weight(alpha, 0.0, 0.05);
        const Scalar A1    = w_pos * A1_pos + (1.0 - w_pos) * A1_neg;
        const Scalar A2    = w_pos * A2_pos + (1.0 - w_pos) * A2_neg;
        const Scalar B1    = w_pos * B1_pos + (1.0 - w_pos) * B1_neg;
        const Scalar B2    = w_pos * B2_pos + (1.0 - w_pos) * B2_neg;

        const Scalar cl_vc = A1 * sin_2a + A2 * (cos_a * cos_a) / (sin_a_abs + 0.02);
        const Scalar cd_vc = B1 * (sin_a * sin_a) + B2 * cos_a_abs;

        return {cl_vc, math::smooth_max(cd_vc, Scalar(0.005), 1e-4)};
    }
};

} // namespace bemt
