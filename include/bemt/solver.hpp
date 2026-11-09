#pragma once

#include "bemt/config.hpp"
#include "bemt/dual.hpp"
#include "bemt/types.hpp"
#include <cmath>

namespace bemt {

template <typename Scalar>
struct ElementKernelOut {
    Scalar residual;
    Scalar vi_base;
    Scalar vi_local;
    Scalar vt_local;
    Scalar alpha;
    Scalar cl;
    Scalar cd;
    Scalar cn;
    Scalar ct;
    Scalar f_loss;
    Scalar w_sq;
    Scalar re;
    Scalar ma;
    Scalar df_z;
    Scalar df_theta;
};

class BEMTSolver {
public:
    explicit BEMTSolver(RotorGeometry geom, SolverConfig cfg = SolverConfig{});

    BEMTResult solve(const OperatingCondition& cond, bool compute_gradients = false) const;

    std::vector<BEMTResult> solve_batch(const std::vector<OperatingCondition>& conds,
                                        bool compute_gradients = false) const;

    const RotorGeometry& geometry() const { return geom_; }
    const SolverConfig& config() const { return cfg_; }

    // ========================================================================
    // 鲁棒性优先版：全象限可微单微元物理核函数 (支持反流区、深度涡环、风车态与极端迎角)
    // ========================================================================
    template <typename Scalar>
    inline ElementKernelOut<Scalar> eval_element_kernel(
        int i_r,
        const Scalar& phi,
        const Scalar& chord_in,
        const Scalar& theta_local,
        const Scalar& u_p,
        const Scalar& u_t,
        const Scalar& v_inplane_sq,
        const Scalar& omega_in,
        const Scalar& wake_factor,
        const OperatingCondition& cond) const {

        using std::acos;
        using std::cos;
        using std::exp;
        using std::pow;
        using std::sin;
        using std::sqrt;

        ElementKernelOut<Scalar> out;

        // 0. 防御性输入保护（防止优化器试探非物理负弦长或零转速导致除零）
        const Scalar chord = math::smooth_max(chord_in, Scalar(1e-4), 1e-5);
        const Scalar omega = math::smooth_max(omega_in, Scalar(1e-2), 1e-3);

        const double r_val = geom_.r[i_r];
        const double R_tip = geom_.radius;
        const double R_hub = geom_.hub_radius;
        const double B     = static_cast<double>(geom_.num_blades);
        const Scalar sigma = (B * chord) / (kTwoPi * r_val);

        const Scalar sin_phi = sin(phi);
        const Scalar cos_phi = cos(phi);
        // 平滑绝对值下界设为 1e-3，防止 phi -> 0 或 pi 时 Prandtl 因子与动量分母奇异
        const Scalar abs_sin = math::smooth_abs(sin_phi, 1e-3);
        const Scalar abs_cos = math::smooth_abs(cos_phi, 1e-3);

        // --------------------------------------------------------------------
        // 1. Prandtl 叶尖与叶根损失因子 F(phi) (带全区间有界平滑保护)
        // --------------------------------------------------------------------
        Scalar F = Scalar(1.0);
        if (cfg_.enable_tip_loss) {
            const Scalar f_tip    = (0.5 * B * std::max(R_tip - r_val, 1e-6)) / (r_val * abs_sin);
            const Scalar f_root   = (0.5 * B * std::max(r_val - R_hub, 1e-6)) / (r_val * abs_sin);
            const Scalar exp_tip  = exp(-math::smooth_min(f_tip, Scalar(30.0), 1e-2));
            const Scalar exp_root = exp(-math::smooth_min(f_root, Scalar(30.0), 1e-2));
            const Scalar F_tip    = (2.0 / kPi) * acos(math::smooth_min(exp_tip, Scalar(1.0 - 1e-7), 1e-6));
            const Scalar F_root   = (2.0 / kPi) * acos(math::smooth_min(exp_root, Scalar(1.0 - 1e-7), 1e-6));
            F = math::smooth_max(F_tip * F_root, Scalar(0.05), 1e-3);
        }
        out.f_loss = F;

        // --------------------------------------------------------------------
        // 2. 局部合成速度、雷诺数 Re 与马赫数 Ma（支持反流区 u_t < 0，不强行抹去负号）
        // --------------------------------------------------------------------
        const Scalar w_approx = sqrt(u_p * u_p + u_t * u_t + 0.25); // 0.5 m/s 正则化下界
        out.re = (cond.rho * w_approx * chord) / cond.mu;
        out.ma = math::smooth_min(w_approx / cond.a_sound, Scalar(0.95), 1e-2);

        // --------------------------------------------------------------------
        // 3. 迎角周期包裹 [-pi, pi] 与三维翼型极曲线插值
        // --------------------------------------------------------------------
        out.alpha = math::wrap_to_pi(theta_local - phi);
        auto [cl_2d, cd_2d] = geom_.airfoils[i_r]->eval(out.alpha, out.re, out.ma);

        // --------------------------------------------------------------------
        // 4. Du-Selig (1998) 三维旋转失速延迟修正（全连续光滑窗口，消除 if 硬分支跳变）
        // --------------------------------------------------------------------
        Scalar cl = cl_2d;
        Scalar cd = cd_2d;
        if (cfg_.enable_stall_delay) {
            const double alpha_0 = geom_.airfoils[i_r]->zero_lift_alpha();
            const double m_slope = geom_.airfoils[i_r]->linear_lift_slope();
            const Scalar cl_pot  = m_slope * (out.alpha - alpha_0);

            // 仅在 |alpha| < 45 deg 范围内需要评估三维失速延迟，避免在深反流区 (±180 deg) 产生非物理放大
            if (std::abs(val_of(out.alpha)) < 50.0 * kDeg2Rad) {
                const Scalar c_over_r         = math::smooth_min(chord / r_val, Scalar(0.8), 1e-3);
                const Scalar v_tip            = omega * R_tip;
                const Scalar v_tot            = sqrt(u_p * u_p + v_inplane_sq + v_tip * v_tip + 1e-4);
                const Scalar lambda_du        = v_tip / v_tot;
                const Scalar exp_term         = (R_tip / r_val) / math::smooth_max(lambda_du, Scalar(0.1), 1e-2);
                const Scalar d_factor         = math::smooth_min(exp_term, Scalar(12.0), 1e-1);

                const Scalar pow_cr  = pow(c_over_r, d_factor);
                const Scalar f_l     = (1.0 / kTwoPi) * ((1.6 * c_over_r / 0.1267) *
                                       (1.0 - pow_cr) / (1.0 + pow_cr) - 1.0);
                const Scalar f_l_pos = math::smooth_max(f_l, Scalar(0.0), 1e-3);

                const Scalar w_stall_zone = Scalar(1.0) - math::sigmoid_weight(
                    math::smooth_abs(out.alpha, 1e-3), 35.0 * kDeg2Rad, 5.0 * kDeg2Rad);

                const Scalar delta_cl = math::smooth_max(cl_pot - cl_2d, Scalar(0.0), 1e-3);
                cl = cl_2d + w_stall_zone * f_l_pos * delta_cl;

                const double cd_0_val = geom_.airfoils[i_r]->eval<double>(
                    alpha_0, val_of(out.re), val_of(out.ma)).second;
                const Scalar delta_cd = math::smooth_max(cd_2d - Scalar(cd_0_val), Scalar(0.0), 1e-4);
                cd = cd_2d - w_stall_zone * 0.3 * f_l_pos * delta_cd;
            }
        }
        out.cl = cl;
        out.cd = math::smooth_max(cd, Scalar(0.002), 1e-4);

        // --------------------------------------------------------------------
        // 5. 分解法向力系数 Cn 与周向阻力系数 Ct
        // --------------------------------------------------------------------
        out.cn = out.cl * cos_phi - out.cd * sin_phi;
        out.ct = out.cl * sin_phi + out.cd * cos_phi;

        // --------------------------------------------------------------------
        // 6. 动量理论闭合：切向诱导速度 vt 与轴向诱导速度 vi (含 VRS 和斜入流无奇异闭合)
        // --------------------------------------------------------------------
        const Scalar denom_sc = 4.0 * F * abs_sin * abs_cos + 1e-4;
        const Scalar kappa_t  = (sigma * out.ct) / denom_sc;
        if (cfg_.enable_tangential_induction) {
            // 使用平滑 tanh 型饱和限幅替代硬夹断，防止 kappa_t -> -1 时分母爆炸
            const Scalar vt_ratio   = kappa_t / sqrt(1.0 + kappa_t * kappa_t + 1e-4);
            const Scalar vt_clamped = math::smooth_clamp(vt_ratio, -0.35, 0.45, 1e-2);
            out.vt_local = u_t * vt_clamped;
        } else {
            out.vt_local = Scalar(0.0);
        }

        // 局部有效切向速度 (保留真实正负号以支持反流区，同时计算平滑模长用于动压与诱导速度尺度)
        const Scalar u_t_eff     = u_t - out.vt_local;
        const Scalar abs_u_t_eff = math::smooth_abs(u_t_eff, 0.1);
        const Scalar w_mag       = abs_u_t_eff / math::smooth_max(abs_cos, Scalar(0.08), 1e-2);
        out.w_sq = w_mag * w_mag;

        // 局部比推力动量速度平方尺度 v_h^2 = dT / (2 * rho * dA * F)
        const Scalar v_thrust_sq = 0.25 * sigma * out.cn * out.w_sq / F;
        const Scalar vh_ref      = sqrt(math::smooth_abs(v_thrust_sq, 1e-5));
        const Scalar sign_t      = v_thrust_sq / (vh_ref * vh_ref + 1e-10);

        // 将面内来流速度 V_xy 直接融合进等效轴向特征速度 U_eff_axial = sqrt(U_P^2 + V_xy^2)
        // 彻底避免 Glauert 四次方程在下降涡环区出现导数变号/奇异！
        const Scalar v_tot_inflow = sqrt(u_p * u_p + v_inplane_sq + 1e-6);
        const Scalar u_p_eff      = u_p;

        Scalar vi0;
        if (cfg_.enable_vrs_correction) {
            const Scalar ratio_vc = u_p_eff / (vh_ref + 1e-3);
            // Glauert 斜入流 + 轴流统一解析求根:
            // 当 V_xy > 0 时，远前方动量流率由 sqrt(V_xy^2 + (U_P/2)^2) 主导
            const Scalar half_up_eff = 0.5 * sqrt(u_p_eff * u_p_eff + 2.0 * v_inplane_sq + 1e-6);
            const Scalar vi_mom      = -0.5 * u_p_eff + sign_t * (sqrt(half_up_eff * half_up_eff + vh_ref * vh_ref) -
                                        (half_up_eff - 0.5 * math::smooth_abs(u_p_eff, 1e-3)));

            // Buhl (2005) 涡环/风车态平滑经验公式
            const Scalar vi_vrs = vh_ref * (1.0 - 0.5 * ratio_vc + 0.15 * ratio_vc * ratio_vc);

            // 仅当轴向处于下降涡环区 (-1.8 < V_z/v_h < -0.2) 且面内风速未完全吹散涡环时平滑启用 VRS
            const Scalar v_xy_suppress = Scalar(1.0) / (1.0 + v_inplane_sq / (vh_ref * vh_ref + 0.1));
            const Scalar w_vrs_lo      = math::sigmoid_weight(ratio_vc, -1.8, 0.25);
            const Scalar w_vrs_hi      = Scalar(1.0) - math::sigmoid_weight(ratio_vc, -0.2, 0.25);
            const Scalar w_vrs         = v_xy_suppress * w_vrs_lo * w_vrs_hi *
                                         math::sigmoid_weight(v_thrust_sq, 0.0, 0.05);
            vi0 = (Scalar(1.0) - w_vrs) * vi_mom + w_vrs * vi_vrs;
        } else {
            const Scalar half_up_eff = 0.5 * sqrt(u_p_eff * u_p_eff + 2.0 * v_inplane_sq + 1e-6);
            vi0 = -0.5 * u_p_eff + sign_t * (sqrt(half_up_eff * half_up_eff + vh_ref * vh_ref) -
                  (half_up_eff - 0.5 * math::smooth_abs(u_p_eff, 1e-3)));
        }

        // 在非涡环区执行 1 步带正定分母保护的 Glauert 四次方程精修，确保斜入流动量精确闭合且永不发散
        if (val_of(v_inplane_sq) > 1e-6) {
            const Scalar vz_tot = u_p + vi0;
            const Scalar v_eff  = sqrt(v_inplane_sq + vz_tot * vz_tot + 1e-4);
            const Scalar g_val  = vi0 * v_eff - v_thrust_sq;
            // 正定安全导数分母: sqrt(dg^2 + 1.0)，绝不允许分母过零或变负
            const Scalar dg_raw = v_eff + (vi0 * vz_tot) / v_eff;
            const Scalar dg_safe = math::smooth_max(dg_raw, Scalar(0.5 * v_tot_inflow + 0.5), 0.1);
            const Scalar delta_vi = math::smooth_clamp(g_val / dg_safe, -0.5 * val_of(vh_ref) - 1.0, 0.5 * val_of(vh_ref) + 1.0, 1e-2);
            vi0 = vi0 - delta_vi;
        }

        out.vi_base  = vi0;
        out.vi_local = vi0 * wake_factor;

        // --------------------------------------------------------------------
        // 7. 全象限连续可微 Ning 氏无奇异残差方程 R(phi) = 0
        //    使用 abs_u_t_eff 保证当 phi 跨越 [-pi/2, pi/2] 时的单调性，同时兼顾反流区
        // --------------------------------------------------------------------
        out.residual = sin_phi * abs_u_t_eff - cos_phi * (u_p + out.vi_local);

        // --------------------------------------------------------------------
        // 8. 真实合成速度与单桨叶单位展长气动载荷
        // --------------------------------------------------------------------
        const Scalar w_true_sq = (u_p + out.vi_local) * (u_p + out.vi_local) + u_t_eff * u_t_eff;
        out.w_sq     = w_true_sq;
        const Scalar q_dyn_c = 0.5 * cond.rho * w_true_sq * chord;
        out.df_z     = q_dyn_c * out.cn;
        out.df_theta = q_dyn_c * out.ct;

        return out;
    }

private:
    RotorGeometry geom_;
    SolverConfig cfg_;

    double solve_cell_brent(int i_r,
                            double chord,
                            double theta_local,
                            double u_p,
                            double u_t,
                            double v_inplane_sq,
                            double omega,
                            double wake_factor,
                            const OperatingCondition& cond,
                            double phi_init,
                            bool& brent_ok) const;
};

} // namespace bemt