#pragma once

#include "bemt/config.hpp"
#include "bemt/dual.hpp"
#include "bemt/types.hpp"
#include <cmath>

namespace bemt {

// ============================================================================
// 单微元评估输出结构体 ElementKernelOut<Scalar>
// 同时支持 double 和 Dual<N>，供求解器迭代与隐函数定理 (IFT) 求导共用
// ============================================================================
template <typename Scalar>
struct ElementKernelOut {
    Scalar residual; // 无奇异 BEMT 残差 R(phi)
    Scalar vi_base;  // 对应的圆环基准轴向诱导速度 v_i0 [m/s]
    Scalar vi_local; // 经 Pitt-Peters 偏斜修正后的局部诱导速度 v_i(r, psi) [m/s]
    Scalar vt_local; // 局部切向诱导速度 v_t(r, psi) [m/s]
    Scalar alpha;    // 局部有效迎角 [rad]
    Scalar cl;       // 修正后的局部升力系数
    Scalar cd;       // 修正后的局部阻力系数
    Scalar cn;       // 法向力系数 (垂直桨盘向上)
    Scalar ct;       // 切向力系数 (阻碍旋转方向)
    Scalar f_loss;   // Prandtl 叶尖/叶根损失因子 F
    Scalar w_sq;     // 局部合成气流速度平方 W^2 [(m/s)^2]
    Scalar re;       // 局部雷诺数
    Scalar ma;       // 局部马赫数
    Scalar df_z;     // 单桨叶单位展长法向推力 dFz/dr [N/m]
    Scalar df_theta; // 单桨叶单位展长周向阻力 dF_theta/dr [N/m]
};

// ============================================================================
// BEMT 核心求解器类 BEMTSolver
// ============================================================================
class BEMTSolver {
public:
    explicit BEMTSolver(RotorGeometry geom, SolverConfig cfg = SolverConfig{});

    // 求解单个运行工况（可选是否同时计算解析敏度/导数）
    BEMTResult solve(const OperatingCondition& cond, bool compute_gradients = false) const;

    // 多工况批量并行求解（用于构建包线或全包线多点气动优化）
    std::vector<BEMTResult> solve_batch(const std::vector<OperatingCondition>& conds,
                                        bool compute_gradients = false) const;

    const RotorGeometry& geometry() const { return geom_; }
    const SolverConfig& config() const { return cfg_; }

    // ========================================================================
    // 核心内联泛型物理核函数（支持 double 与 Dual<N>）
    // 给定径向索引 i_r、局部入流角 phi、几何参数、局部来流分量及尾迹谐波修正因子 wake_factor
    // ========================================================================
    template <typename Scalar>
    inline ElementKernelOut<Scalar> eval_element_kernel(
        int i_r,
        const Scalar& phi,
        const Scalar& chord,
        const Scalar& theta_local,
        const Scalar& u_p,          // 局部轴向来流 U_P = Vz [m/s]
        const Scalar& u_t,          // 局部切向来流 U_T = omega*r + spin_dir*(Vx*sin(psi) - Vy*cos(psi)) [m/s]
        const Scalar& v_inplane_sq, // 桨盘面内远前方来流平方 Vx^2 + Vy^2 [(m/s)^2]
        const Scalar& omega,        // 旋转角速度 [rad/s]
        const Scalar& wake_factor,  // Pitt-Peters 局部谐波因子: 1 + kx*(r/R)*cos(psi) + ky*(r/R)*sin(psi)
        const OperatingCondition& cond) const {

        using std::acos;
        using std::cos;
        using std::exp;
        using std::pow;
        using std::sin;
        using std::sqrt;

        ElementKernelOut<Scalar> out;

        const double r_val    = geom_.r[i_r];
        const double R_tip    = geom_.radius;
        const double R_hub    = geom_.hub_radius;
        const double B        = static_cast<double>(geom_.num_blades);
        const Scalar sigma    = (B * chord) / (kTwoPi * r_val); // 局部实度

        const Scalar sin_phi  = sin(phi);
        const Scalar cos_phi  = cos(phi);
        const Scalar abs_sin  = math::smooth_abs(sin_phi, 1e-4);

        // --------------------------------------------------------------------
        // 1. 修正 1: Prandtl 叶尖与叶根损失因子 F(phi)
        // --------------------------------------------------------------------
        Scalar F = Scalar(1.0);
        if (cfg_.enable_tip_loss) {
            const Scalar f_tip  = (0.5 * B * (R_tip - r_val)) / (r_val * abs_sin);
            const Scalar f_root = (0.5 * B * (r_val - R_hub)) / (r_val * abs_sin);
            const Scalar exp_tip  = exp(-math::smooth_min(f_tip, Scalar(35.0), 1e-3));
            const Scalar exp_root = exp(-math::smooth_min(f_root, Scalar(35.0), 1e-3));
            const Scalar F_tip  = (2.0 / kPi) * acos(math::smooth_min(exp_tip, Scalar(1.0 - 1e-9), 1e-7));
            const Scalar F_root = (2.0 / kPi) * acos(math::smooth_min(exp_root, Scalar(1.0 - 1e-9), 1e-7));
            F = math::smooth_max(F_tip * F_root, Scalar(0.02), 1e-4);
        }
        out.f_loss = F;

        // --------------------------------------------------------------------
        // 2. 估算局部合成速度 W、雷诺数 Re 与马赫数 Ma
        // --------------------------------------------------------------------
        const Scalar u_t_safe = math::smooth_max(u_t, Scalar(0.5), 1e-2);
        const Scalar w_approx = math::smooth_max(
            sqrt(u_p * u_p + u_t_safe * u_t_safe), Scalar(0.5), 1e-2);
        out.re = (cond.rho * w_approx * chord) / cond.mu;
        out.ma = w_approx / cond.a_sound;

        // --------------------------------------------------------------------
        // 3. 三维翼型表插值 Cl_2D(alpha, Re, Ma), Cd_2D(alpha, Re, Ma)
        // --------------------------------------------------------------------
        out.alpha = theta_local - phi;
        auto [cl_2d, cd_2d] = geom_.airfoils[i_r]->eval(out.alpha, out.re, out.ma);

        // --------------------------------------------------------------------
        // 4. 修正 3: Du-Selig (1998) 三维旋转失速延迟修正
        // --------------------------------------------------------------------
        Scalar cl = cl_2d;
        Scalar cd = cd_2d;
        if (cfg_.enable_stall_delay) {
            const double alpha_0 = geom_.airfoils[i_r]->zero_lift_alpha();
            const double m_slope = geom_.airfoils[i_r]->linear_lift_slope();
            const Scalar c_over_r = chord / r_val;
            const Scalar c_over_r_clamped = math::smooth_min(c_over_r, Scalar(0.8), 1e-3);

            // 修正版叶尖速比特征量 Lambda = Omega * R / sqrt(V_inf^2 + (Omega*R)^2)
            const Scalar v_tip = omega * R_tip;
            const Scalar v_tot = sqrt(u_p * u_p + v_inplane_sq + v_tip * v_tip + 1e-6);
            const Scalar lambda_du = v_tip / v_tot;
            const Scalar exp_term  = (R_tip / r_val) / math::smooth_max(lambda_du, Scalar(0.1), 1e-3);
            const Scalar d_factor  = math::smooth_min(exp_term, Scalar(15.0), 1e-2);

            // Du-Selig 升力与阻力三维延迟因子 f_L, f_D
            const Scalar pow_cr = pow(c_over_r_clamped, d_factor);
            const Scalar f_l = (1.0 / kTwoPi) * ((1.6 * c_over_r_clamped / 0.1267) *
                               (1.0 - pow_cr) / (1.0 + pow_cr) - 1.0);
            const Scalar f_l_pos = math::smooth_max(f_l, Scalar(0.0), 1e-3);

            // 仅在失速偏离线性段但未进入背面反流区 (|alpha| < 35 deg) 时平滑激活修正
            const Scalar w_stall_zone = Scalar(1.0) - math::sigmoid_weight(
                math::smooth_abs(out.alpha, 1e-3), 35.0 * kDeg2Rad, 5.0 * kDeg2Rad);

            const Scalar cl_pot   = m_slope * (out.alpha - alpha_0);
            const Scalar delta_cl = math::smooth_max(cl_pot - cl_2d, Scalar(0.0), 1e-3);
            cl = cl_2d + w_stall_zone * f_l_pos * delta_cl;

            // 对应的阻力延迟修正
            const Scalar cd_0 = geom_.airfoils[i_r]->eval(Scalar(alpha_0), out.re, out.ma).second;
            const Scalar delta_cd = math::smooth_max(cd_2d - cd_0, Scalar(0.0), 1e-4);
            cd = cd_2d - w_stall_zone * 0.3 * f_l_pos * delta_cd;
        }
        out.cl = cl;
        out.cd = math::smooth_max(cd, Scalar(0.002), 1e-4);

        // --------------------------------------------------------------------
        // 5. 分解法向力系数 Cn 与周向阻力系数 Ct
        // --------------------------------------------------------------------
        out.cn = out.cl * cos_phi - out.cd * sin_phi;
        out.ct = out.cl * sin_phi + out.cd * cos_phi;

        // --------------------------------------------------------------------
        // 6. 动量理论闭合与修正 2: Glauert 斜入流 + Buhl 高诱导区 (VRS) 修正
        // --------------------------------------------------------------------
        // 计算切向诱导速度 v_t
        const Scalar denom_sc = 4.0 * F * abs_sin * math::smooth_abs(cos_phi, 1e-4) + 1e-6;
        const Scalar kappa_t  = (sigma * out.ct) / denom_sc;
        if (cfg_.enable_tangential_induction) {
            const Scalar vt_ratio = kappa_t / (1.0 + kappa_t);
            const Scalar vt_clamped = math::smooth_clamp(vt_ratio, -0.35, 0.5, 1e-3);
            out.vt_local = u_t * vt_clamped;
        } else {
            out.vt_local = Scalar(0.0);
        }

        // 局部有效切向速度 W * cos(phi) = U_T - v_t
        const Scalar u_t_eff = math::smooth_max(u_t - out.vt_local, Scalar(0.1), 1e-2);
        // 由叶素角度关系给出的局部合成速度 W = (U_T - v_t) / cos(phi)
        const Scalar w_mag   = u_t_eff / math::smooth_max(cos_phi, Scalar(0.05), 1e-3);
        out.w_sq = w_mag * w_mag;

        // 单叶素产生的局部气动比推力速度平方尺度 v_h^2 = (dT_BE / (2 * rho * dA))
        const Scalar v_thrust_sq = 0.25 * sigma * out.cn * out.w_sq / F;

        // 求解 Glauert 斜入流动量四次方程: v_i0 * sqrt(V_xy^2 + (U_P + v_i0)^2) = v_thrust_sq
        // 并结合 Buhl (2005) 涡环/风车高诱导速度区平滑修正
        const Scalar vh_ref = sqrt(math::smooth_abs(v_thrust_sq, 1e-6));
        const Scalar sign_t = v_thrust_sq / (vh_ref * vh_ref + 1e-12);

        Scalar vi_axial;
        if (cfg_.enable_vrs_correction) {
            const Scalar ratio_vc = u_p / (vh_ref + 1e-4);
            const Scalar vi_mom = -0.5 * u_p + sign_t * sqrt(0.25 * u_p * u_p + vh_ref * vh_ref);
            const Scalar vi_vrs = vh_ref * (1.0 - 0.5 * ratio_vc + 0.15 * ratio_vc * ratio_vc);

            const Scalar w_vrs_lo = math::sigmoid_weight(ratio_vc, -1.8, 0.25);
            const Scalar w_vrs_hi = Scalar(1.0) - math::sigmoid_weight(ratio_vc, -0.2, 0.25);
            const Scalar w_vrs    = w_vrs_lo * w_vrs_hi * math::sigmoid_weight(v_thrust_sq, 0.0, 0.01);
            vi_axial = (Scalar(1.0) - w_vrs) * vi_mom + w_vrs * vi_vrs;
        } else {
            vi_axial = -0.5 * u_p + sign_t * sqrt(0.25 * u_p * u_p + vh_ref * vh_ref);
        }

        // 引入面内斜入流速度 V_xy 对有效质量流率的增强效应 (Glauert Oblique Momentum)
        Scalar vi0 = vi_axial;
        for (int iter = 0; iter < 2; ++iter) {
            const Scalar vz_tot = u_p + vi0;
            const Scalar v_eff  = sqrt(v_inplane_sq + vz_tot * vz_tot + 1e-6);
            const Scalar g_val  = vi0 * v_eff - v_thrust_sq;
            const Scalar dg_dvi = v_eff + (vi0 * vz_tot) / v_eff;
            vi0 = vi0 - g_val / math::smooth_max(dg_dvi, Scalar(0.1), 1e-3);
        }

        out.vi_base  = vi0;
        out.vi_local = vi0 * wake_factor;

        // --------------------------------------------------------------------
        // 7. 构造全域连续可微、无分母奇异的 Ning 氏残差方程 R(phi) = 0
        // --------------------------------------------------------------------
        out.residual = sin_phi * u_t_eff - cos_phi * (u_p + out.vi_local);

        // --------------------------------------------------------------------
        // 8. 计算真实合成速度与单桨叶单位展长气动载荷
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
                            double phi_init) const;
};

} // namespace bemt