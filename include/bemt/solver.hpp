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
        const OperatingCondition& cond,
        const Scalar& cl_2d_offset = Scalar(0.0),
        const Scalar& cd_2d_offset = Scalar(0.0)) const {

        using std::acos;
        using std::cos;
        using std::exp;
        using std::pow;
        using std::sin;
        using std::sqrt;

        ElementKernelOut<Scalar> out;

        const Scalar chord = math::smooth_max(chord_in, Scalar(1e-4), 1e-5);
        const Scalar omega = math::smooth_max(omega_in, Scalar(1e-2), 1e-3);

        const double r_val = geom_.r[i_r];
        const double R_tip = geom_.radius;
        const double R_hub = geom_.hub_radius;
        const double B     = static_cast<double>(geom_.num_blades);
        const Scalar sigma = (B * chord) / (kTwoPi * r_val);

        const Scalar sin_phi = sin(phi);
        const Scalar cos_phi = cos(phi);
        const Scalar abs_sin = math::smooth_abs(sin_phi, 1e-3);
        const Scalar abs_cos = math::smooth_abs(cos_phi, 1e-3);

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

        const Scalar w_approx = sqrt(u_p * u_p + u_t * u_t + 0.25);
        out.re = (cond.rho * w_approx * chord) / cond.mu;
        out.ma = math::smooth_min(w_approx / cond.a_sound, Scalar(0.95), 1e-2);

        out.alpha = math::wrap_to_pi(theta_local - phi);
        auto [cl_2d_raw, cd_2d_raw] = geom_.airfoils[i_r]->eval(out.alpha, out.re, out.ma);
        const Scalar cl_2d = cl_2d_raw + cl_2d_offset;
        const Scalar cd_2d = cd_2d_raw + cd_2d_offset;

        Scalar cl = cl_2d;
        Scalar cd = cd_2d;
        if (cfg_.enable_stall_delay) {
            const double alpha_0 = geom_.airfoils[i_r]->zero_lift_alpha();
            const double m_slope = geom_.airfoils[i_r]->linear_lift_slope();
            const Scalar cl_pot  = m_slope * (out.alpha - alpha_0);

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

                const Scalar cd_0 = geom_.airfoils[i_r]->eval<Scalar>(
                    Scalar(alpha_0), out.re, out.ma).second;
                const Scalar delta_cd = math::smooth_max(cd_2d - cd_0, Scalar(0.0), 1e-4);
                cd = cd_2d - w_stall_zone * 0.3 * f_l_pos * delta_cd;
            }
        }
        out.cl = cl;
        out.cd = math::smooth_max(cd, Scalar(0.002), 1e-4);

        out.cn = out.cl * cos_phi - out.cd * sin_phi;
        out.ct = out.cl * sin_phi + out.cd * cos_phi;

        const Scalar denom_sc = 4.0 * F * abs_sin * abs_cos + 1e-4;
        const Scalar kappa_t  = (sigma * out.ct) / denom_sc;
        if (cfg_.enable_tangential_induction) {
            const Scalar vt_ratio   = kappa_t / sqrt(1.0 + kappa_t * kappa_t + 1e-4);
            const Scalar vt_clamped = math::smooth_clamp(vt_ratio, -0.35, 0.45, 1e-2);
            out.vt_local = u_t * vt_clamped;
        } else {
            out.vt_local = Scalar(0.0);
        }

        const Scalar u_t_eff     = u_t - out.vt_local;
        const Scalar abs_u_t_eff = math::smooth_abs(u_t_eff, 1e-6);
        const Scalar w_mag       = abs_u_t_eff / math::smooth_max(abs_cos, Scalar(0.05), 1e-5);
        out.w_sq = w_mag * w_mag;

        const Scalar v_thrust_sq = 0.25 * sigma * out.cn * out.w_sq / F;
        const Scalar vh_ref      = sqrt(math::smooth_abs(v_thrust_sq, 1e-9));
        const Scalar sign_t      = v_thrust_sq / (vh_ref * vh_ref + 1e-16);

        const Scalar v_tot_inflow = sqrt(u_p * u_p + v_inplane_sq + 1e-9);
        const Scalar vc_bar = u_p / (vh_ref + 1e-5);
        const Scalar vi_climb = vh_ref * (-0.5 * vc_bar + sign_t * sqrt(0.25 * vc_bar * vc_bar + 1.0));

        Scalar vi0;
        if (cfg_.enable_vrs_correction) {
            const Scalar vc_clamped = math::smooth_clamp(vc_bar, -2.1, 0.0, 0.05);
            const Scalar vc2 = vc_clamped * vc_clamped;
            const Scalar vc3 = vc2 * vc_clamped;
            const Scalar vc4 = vc2 * vc2;
            const Scalar vi_vrs_poly = vh_ref * sign_t * (1.0 - 1.125 * vc_clamped - 1.372 * vc2 - 1.718 * vc3 - 0.655 * vc4);

            const Scalar half_vc_neg = math::smooth_max(-0.5 * vc_bar, Scalar(1.02), 0.05);
            const Scalar rad_wm      = sqrt(math::smooth_max(half_vc_neg * half_vc_neg - 1.0, Scalar(0.04), 0.02));
            const Scalar vi_windmill = (vh_ref * sign_t) / (half_vc_neg + rad_wm);

            const Scalar w_climb = math::sigmoid_weight(vc_bar, -0.05, 0.08);
            const Scalar w_windm = Scalar(1.0) - math::sigmoid_weight(vc_bar, -1.95, 0.10);
            const Scalar w_vrs   = math::smooth_max(Scalar(1.0) - w_climb - w_windm, Scalar(0.0), 1e-4);

            const Scalar vi_descent = w_vrs * vi_vrs_poly + w_windm * vi_windmill;
            const Scalar v_xy_suppress = Scalar(1.0) / (1.0 + v_inplane_sq / (vh_ref * vh_ref + 0.1));
            vi0 = w_climb * vi_climb + (Scalar(1.0) - w_climb) * (v_xy_suppress * vi_descent + (Scalar(1.0) - v_xy_suppress) * vi_climb);
        } else {
            vi0 = vi_climb;
        }

        if (val_of(v_inplane_sq) > 1e-8) {
            for (int iter = 0; iter < 2; ++iter) {
                const Scalar vz_tot  = u_p + vi0;
                const Scalar v_eff   = sqrt(v_inplane_sq + vz_tot * vz_tot + 1e-6);
                const Scalar g_val   = vi0 * v_eff - v_thrust_sq;
                const Scalar dg_raw  = v_eff + (vi0 * vz_tot) / v_eff;
                const Scalar dg_safe = math::smooth_max(dg_raw, 0.5 * v_tot_inflow + 0.5, 0.05);
                vi0 = vi0 - g_val / dg_safe;
            }
        }

        out.vi_base  = vi0;
        out.vi_local = vi0 * wake_factor;

        out.residual = sin_phi * abs_u_t_eff - cos_phi * (u_p + out.vi_local);

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