#include "bemt/sensitivity.hpp"
#include "bemt/dual.hpp"
#include <cmath>
#include <omp.h>

namespace bemt {

void compute_ift_sensitivities(const BEMTSolver& solver,
                               const OperatingCondition& cond,
                               BEMTResult& res) {
    const RotorGeometry& geom = solver.geometry();
    const SolverConfig& cfg   = solver.config();

    const int nr   = static_cast<int>(geom.r.size());
    const int npsi = cfg.n_azimuth;
    const double B = static_cast<double>(geom.num_blades);
    const double R_tip = geom.radius;
    const double s_dir = static_cast<double>(geom.spin_dir);
    const double inv_npsi = 1.0 / static_cast<double>(npsi);
    const double A_disk   = kPi * (R_tip * R_tip - geom.hub_radius * geom.hub_radius);

    BEMTSensitivity& sens = res.sensitivity;
    sens.valid = true;

    sens.dT_dchord  = ArrayXd::Zero(nr);
    sens.dT_dtwist  = ArrayXd::Zero(nr);
    sens.dQ_dchord  = ArrayXd::Zero(nr);
    sens.dQ_dtwist  = ArrayXd::Zero(nr);
    sens.dP_dchord  = ArrayXd::Zero(nr);
    sens.dP_dtwist  = ArrayXd::Zero(nr);
    sens.dMx_dchord = ArrayXd::Zero(nr);
    sens.dMx_dtwist = ArrayXd::Zero(nr);
    sens.dMy_dchord = ArrayXd::Zero(nr);
    sens.dMy_dtwist = ArrayXd::Zero(nr);

    sens.dTr_dchord_diag = ArrayXd::Zero(nr);
    sens.dTr_dtwist_diag = ArrayXd::Zero(nr);
    sens.dForces_dCond   = MatrixXd::Zero(7, 5);

    ArrayXd cos_psi = res.psi_grid.cos();
    ArrayXd sin_psi = res.psi_grid.sin();

    const double v_inplane_sq = cond.vx * cond.vx + cond.vy * cond.vy;
    const double v_inplane    = std::sqrt(v_inplane_sq);
    const double psi_wind     = std::atan2(cond.vy, cond.vx);
    ArrayXd cos_psi_rel       = (res.psi_grid - psi_wind).cos();
    ArrayXd sin_psi_rel       = (res.psi_grid - psi_wind).sin();

    const bool use_pitt_peters = (cfg.wake_model == WakeModel::kGlauertPittPeters) && (v_inplane > 1e-5);

    using Dual8 = Dual<8>;

    MatrixXd ring_cond_jac  = MatrixXd::Zero(nr, 40);
    ArrayXd dvi_mean_dchord = ArrayXd::Zero(nr);
    ArrayXd dvi_mean_dtwist = ArrayXd::Zero(nr);

    double dT_dkx  = 0.0;
    double dQ_dkx  = 0.0;
    double dFx_dkx = 0.0;
    double dFy_dkx = 0.0;
    double dMx_dkx = 0.0;
    double dMy_dkx = 0.0;
    double dvi_mean_dkx = 0.0;

    #pragma omp parallel for schedule(static) \
        reduction(+:dT_dkx, dQ_dkx, dFx_dkx, dFy_dkx, dMx_dkx, dMy_dkx, dvi_mean_dkx) \
        if(!omp_in_parallel())
    for (int i = 0; i < nr; ++i) {
        const double r_i    = geom.r[i];
        const double dr_i   = geom.dr[i];
        const double r_norm = r_i / R_tip;
        const double c_i    = geom.chord[i];
        const double tw_i   = geom.twist[i];
        const double w_ring = B * dr_i * inv_npsi;
        const double w_area = (kTwoPi * r_i * dr_i / A_disk) * inv_npsi;

        Eigen::Matrix<double, 7, 5> local_cond_jac  = Eigen::Matrix<double, 7, 5>::Zero();
        Eigen::Matrix<double, 5, 1> local_dvi_dcond = Eigen::Matrix<double, 5, 1>::Zero();

        for (int j = 0; j < npsi; ++j) {
            const double phi_star = res.phi(i, j);
            const double c_psi    = cos_psi[j];
            const double s_psi    = sin_psi[j];
            const double c_rel    = cos_psi_rel[j];
            const double s_rel    = sin_psi_rel[j];

            Dual8 phi_d(phi_star, 0, 1.0);
            Dual8 chord_d(c_i,    1, 1.0);
            Dual8 twist_d(tw_i,   2, 1.0);
            Dual8 omega_d(cond.omega,    3, 1.0);
            Dual8 pitch0_d(cond.pitch_0, 4, 1.0);
            Dual8 vx_d(cond.vx,   5, 1.0);
            Dual8 vy_d(cond.vy,   6, 1.0);
            Dual8 vz_d(cond.vz,   7, 1.0);

            Dual8 theta_d   = twist_d + pitch0_d + cond.pitch_1c * c_psi + cond.pitch_1s * s_psi;
            Dual8 u_t_d     = omega_d * r_i + s_dir * (vx_d * s_psi - vy_d * c_psi);
            Dual8 u_p_d     = vz_d;
            Dual8 v_in_sq_d = vx_d * vx_d + vy_d * vy_d;

            const double raw_wake   = 1.0 + res.pitt_kx * r_norm * c_rel + res.pitt_ky * r_norm * s_rel;
            const Dual1 raw_wake_d1(raw_wake, 1.0);
            const Dual1 wake_smooth = math::smooth_max(raw_wake_d1, Dual1(0.1), 0.02);
            const double wake_fac   = wake_smooth.val;
            const double dwake_draw = wake_smooth.grad(0);
            Dual8 wake_fac_d(wake_fac);

            auto out_d = solver.eval_element_kernel<Dual8>(
                i, phi_d, chord_d, theta_d, u_p_d, u_t_d, v_in_sq_d, omega_d, wake_fac_d, cond);

            // Tikhonov 正则化安全求逆，杜绝 dR/dphi -> 0 时梯度爆炸
            const double dR_dphi     = out_d.residual.grad(0);
            const double inv_dR_dphi = -dR_dphi / (dR_dphi * dR_dphi + 1e-16);

            Eigen::Array<double, 7, 1> dphi_dx;
            for (int k = 0; k < 7; ++k) {
                dphi_dx[k] = out_d.residual.grad(k + 1) * inv_dR_dphi;
            }

            const double dfz_dphi = out_d.df_z.grad(0);
            const double dft_dphi = out_d.df_theta.grad(0);
            const double dvi_dphi = out_d.vi_local.grad(0);

            Eigen::Array<double, 7, 1> dfz_dx, dft_dx, dvi_dx;
            for (int k = 0; k < 7; ++k) {
                dfz_dx[k] = out_d.df_z.grad(k + 1)     + dfz_dphi * dphi_dx[k];
                dft_dx[k] = out_d.df_theta.grad(k + 1) + dft_dphi * dphi_dx[k];
                dvi_dx[k] = out_d.vi_local.grad(k + 1) + dvi_dphi * dphi_dx[k];
            }

            sens.dT_dchord[i]  += w_ring * dfz_dx[0];
            sens.dT_dtwist[i]  += w_ring * dfz_dx[1];
            sens.dQ_dchord[i]  += w_ring * r_i * dft_dx[0];
            sens.dQ_dtwist[i]  += w_ring * r_i * dft_dx[1];
            sens.dMx_dchord[i] += w_ring * r_i * s_psi * dfz_dx[0];
            sens.dMx_dtwist[i] += w_ring * r_i * s_psi * dfz_dx[1];
            sens.dMy_dchord[i] += -w_ring * r_i * c_psi * dfz_dx[0];
            sens.dMy_dtwist[i] += -w_ring * r_i * c_psi * dfz_dx[1];

            dvi_mean_dchord[i] += w_area * dvi_dx[0];
            dvi_mean_dtwist[i] += w_area * dvi_dx[1];

            for (int c_idx = 0; c_idx < 5; ++c_idx) {
                const double g_fz = dfz_dx[c_idx + 2];
                const double g_ft = dft_dx[c_idx + 2];

                local_cond_jac(0, c_idx) += w_ring * g_fz;
                local_cond_jac(1, c_idx) += w_ring * r_i * g_ft;
                local_cond_jac(3, c_idx) += w_ring * s_dir * s_psi * g_ft;
                local_cond_jac(4, c_idx) += -w_ring * s_dir * c_psi * g_ft;
                local_cond_jac(5, c_idx) += w_ring * r_i * s_psi * g_fz;
                local_cond_jac(6, c_idx) += -w_ring * r_i * c_psi * g_fz;
                local_dvi_dcond[c_idx]   += w_area * dvi_dx[c_idx + 2];
            }

            if (use_pitt_peters) {
                const double dwake_dkx = dwake_draw * r_norm * c_rel;
                const double dR_dwake   = -std::cos(phi_star) * out_d.vi_base.val;
                const double dphi_dwake = dR_dwake * inv_dR_dphi;

                const double u_p_tot = cond.vz + out_d.vi_local.val;
                const double dfz_dwake_explicit = cond.rho * u_p_tot * out_d.vi_base.val * c_i * out_d.cn.val;
                const double dft_dwake_explicit = cond.rho * u_p_tot * out_d.vi_base.val * c_i * out_d.ct.val;

                const double dfz_dkx_cell = (dfz_dwake_explicit + dfz_dphi * dphi_dwake) * dwake_dkx;
                const double dft_dkx_cell = (dft_dwake_explicit + dft_dphi * dphi_dwake) * dwake_dkx;
                const double dvi_dkx_cell = (out_d.vi_base.val + dvi_dphi * dphi_dwake) * dwake_dkx;

                dT_dkx  += w_ring * dfz_dkx_cell;
                dQ_dkx  += w_ring * r_i * dft_dkx_cell;
                dFx_dkx += w_ring * s_dir * s_psi * dft_dkx_cell;
                dFy_dkx += -w_ring * s_dir * c_psi * dft_dkx_cell;
                dMx_dkx += w_ring * r_i * s_psi * dfz_dkx_cell;
                dMy_dkx += -w_ring * r_i * c_psi * dfz_dkx_cell;
                dvi_mean_dkx += w_area * dvi_dkx_cell;
            }
        }

        sens.dTr_dchord_diag[i] = sens.dT_dchord[i] / dr_i;
        sens.dTr_dtwist_diag[i] = sens.dT_dtwist[i] / dr_i;

        for (int r_idx = 0; r_idx < 7; ++r_idx) {
            for (int c_idx = 0; c_idx < 5; ++c_idx) {
                ring_cond_jac(i, r_idx * 5 + c_idx) = local_cond_jac(r_idx, c_idx);
            }
        }
        for (int c_idx = 0; c_idx < 5; ++c_idx) {
            ring_cond_jac(i, 35 + c_idx) = local_dvi_dcond[c_idx];
        }
    }

    Eigen::Matrix<double, 5, 1> dvi_mean_dcond = Eigen::Matrix<double, 5, 1>::Zero();
    for (int i = 0; i < nr; ++i) {
        for (int r_idx = 0; r_idx < 7; ++r_idx) {
            for (int c_idx = 0; c_idx < 5; ++c_idx) {
                sens.dForces_dCond(r_idx, c_idx) += ring_cond_jac(i, r_idx * 5 + c_idx);
            }
        }
        for (int c_idx = 0; c_idx < 5; ++c_idx) {
            dvi_mean_dcond[c_idx] += ring_cond_jac(i, 35 + c_idx);
        }
    }

    if (use_pitt_peters) {
        const Dual1 vz_sum_d1(cond.vz + res.vi_disk_mean, 1.0);
        const Dual1 vz_wake_d1    = math::smooth_abs(vz_sum_d1, 0.2);
        const double vz_wake      = vz_wake_d1.val;
        const double dvz_wake_dvi = vz_wake_d1.grad(0);

        const double denom_chi    = std::max(v_inplane_sq + vz_wake * vz_wake, 1e-4);
        const double cos_half_chi = std::max(std::cos(0.5 * res.wake_skew_chi), 0.1);
        const double dkx_dchi     = (15.0 * kPi / 46.0) / (cos_half_chi * cos_half_chi);

        const double dchi_dvi = (-v_inplane * dvz_wake_dvi) / denom_chi;
        const double dkx_dvi  = dkx_dchi * dchi_dvi;

        const double raw_denom = 1.0 - dkx_dvi * dvi_mean_dkx;
        const double self_consistent_fac = (std::abs(raw_denom) > 0.1) ? (1.0 / raw_denom) : 1.0;
        const double dkx_dvi_total       = dkx_dvi * self_consistent_fac;

        sens.dT_dchord  += dT_dkx  * dkx_dvi_total * dvi_mean_dchord;
        sens.dT_dtwist  += dT_dkx  * dkx_dvi_total * dvi_mean_dtwist;
        sens.dQ_dchord  += dQ_dkx  * dkx_dvi_total * dvi_mean_dchord;
        sens.dQ_dtwist  += dQ_dkx  * dkx_dvi_total * dvi_mean_dtwist;
        sens.dMx_dchord += dMx_dkx * dkx_dvi_total * dvi_mean_dchord;
        sens.dMx_dtwist += dMx_dkx * dkx_dvi_total * dvi_mean_dtwist;
        sens.dMy_dchord += dMy_dkx * dkx_dvi_total * dvi_mean_dchord;
        sens.dMy_dtwist += dMy_dkx * dkx_dvi_total * dvi_mean_dtwist;

        const double dchi_dvx = (vz_wake * cond.vx / v_inplane) / denom_chi;
        const double dchi_dvy = (vz_wake * cond.vy / v_inplane) / denom_chi;
        const double dchi_dvz = dchi_dvi;

        Eigen::Matrix<double, 5, 1> dkx_dcond_explicit = Eigen::Matrix<double, 5, 1>::Zero();
        dkx_dcond_explicit[2] = dkx_dchi * dchi_dvx;
        dkx_dcond_explicit[3] = dkx_dchi * dchi_dvy;
        dkx_dcond_explicit[4] = dkx_dchi * dchi_dvz;

        Eigen::Matrix<double, 5, 1> dkx_dcond_total =
            (dkx_dcond_explicit + dkx_dvi * dvi_mean_dcond) * self_consistent_fac;

        for (int c_idx = 0; c_idx < 5; ++c_idx) {
            const double gkx = dkx_dcond_total[c_idx];
            sens.dForces_dCond(0, c_idx) += dT_dkx  * gkx;
            sens.dForces_dCond(1, c_idx) += dQ_dkx  * gkx;
            sens.dForces_dCond(3, c_idx) += dFx_dkx * gkx;
            sens.dForces_dCond(4, c_idx) += dFy_dkx * gkx;
            sens.dForces_dCond(5, c_idx) += dMx_dkx * gkx;
            sens.dForces_dCond(6, c_idx) += dMy_dkx * gkx;
        }
    }

    sens.dP_dchord = cond.omega * sens.dQ_dchord;
    sens.dP_dtwist = cond.omega * sens.dQ_dtwist;

    for (int c_idx = 0; c_idx < 5; ++c_idx) {
        sens.dForces_dCond(2, c_idx) = cond.omega * sens.dForces_dCond(1, c_idx);
    }
    sens.dForces_dCond(2, 0) += res.torque;
}

} // namespace bemt