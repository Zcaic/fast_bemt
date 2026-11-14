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
    sens.dFx_dchord = ArrayXd::Zero(nr);
    sens.dFx_dtwist = ArrayXd::Zero(nr);
    sens.dFy_dchord = ArrayXd::Zero(nr);
    sens.dFy_dtwist = ArrayXd::Zero(nr);
    sens.dMx_dchord = ArrayXd::Zero(nr);
    sens.dMx_dtwist = ArrayXd::Zero(nr);
    sens.dMy_dchord = ArrayXd::Zero(nr);
    sens.dMy_dtwist = ArrayXd::Zero(nr);

    sens.dT_dcl  = ArrayXXd::Zero(nr, npsi);
    sens.dT_dcd  = ArrayXXd::Zero(nr, npsi);
    sens.dQ_dcl  = ArrayXXd::Zero(nr, npsi);
    sens.dQ_dcd  = ArrayXXd::Zero(nr, npsi);
    sens.dP_dcl  = ArrayXXd::Zero(nr, npsi);
    sens.dP_dcd  = ArrayXXd::Zero(nr, npsi);
    sens.dFx_dcl = ArrayXXd::Zero(nr, npsi);
    sens.dFx_dcd = ArrayXXd::Zero(nr, npsi);
    sens.dFy_dcl = ArrayXXd::Zero(nr, npsi);
    sens.dFy_dcd = ArrayXXd::Zero(nr, npsi);
    sens.dMx_dcl = ArrayXXd::Zero(nr, npsi);
    sens.dMx_dcd = ArrayXXd::Zero(nr, npsi);
    sens.dMy_dcl = ArrayXXd::Zero(nr, npsi);
    sens.dMy_dcd = ArrayXXd::Zero(nr, npsi);

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

    MatrixXd ring_cond_jac  = MatrixXd::Zero(nr, 40);
    ArrayXd dvi_mean_dchord = ArrayXd::Zero(nr);
    ArrayXd dvi_mean_dtwist = ArrayXd::Zero(nr);
    ArrayXXd dvi_mean_dcl   = ArrayXXd::Zero(nr, npsi);
    ArrayXXd dvi_mean_dcd   = ArrayXXd::Zero(nr, npsi);
    ArrayXd dT_ring_dkx     = ArrayXd::Zero(nr);

    double dT_dkx       = 0.0;
    double dQ_dkx       = 0.0;
    double dFx_dkx      = 0.0;
    double dFy_dkx      = 0.0;
    double dMx_dkx      = 0.0;
    double dMy_dkx      = 0.0;
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

            Dual11 phi_d(phi_star, 0, 1.0);
            Dual11 chord_d(c_i,    1, 1.0);
            Dual11 twist_d(tw_i,   2, 1.0);
            Dual11 omega_d(cond.omega,    3, 1.0);
            Dual11 pitch0_d(cond.pitch_0, 4, 1.0);
            Dual11 vx_d(cond.vx,   5, 1.0);
            Dual11 vy_d(cond.vy,   6, 1.0);
            Dual11 vz_d(cond.vz,   7, 1.0);

            Dual11 theta_d   = twist_d + pitch0_d + cond.pitch_1c * c_psi + cond.pitch_1s * s_psi;
            Dual11 u_t_d     = omega_d * r_i + s_dir * (vx_d * s_psi - vy_d * c_psi);
            Dual11 u_p_d     = vz_d;
            Dual11 v_in_sq_d = vx_d * vx_d + vy_d * vy_d;

            const double raw_wake   = 1.0 + res.pitt_kx * r_norm * c_rel + res.pitt_ky * r_norm * s_rel;
            const Dual1 raw_wake_d1(raw_wake, 1.0);
            const Dual1 wake_smooth = math::smooth_max(raw_wake_d1, Dual1(0.1), 0.02);
            const double wake_fac   = wake_smooth.val;
            const double dwake_draw = wake_smooth.grad(0);

            Dual11 wake_fac_d(wake_fac, 8, 1.0);
            Dual11 cl_off_d(0.0, 9, 1.0);
            Dual11 cd_off_d(0.0, 10, 1.0);

            auto out_d = solver.eval_element_kernel<Dual11>(
                i, phi_d, chord_d, theta_d, u_p_d, u_t_d, v_in_sq_d, omega_d, wake_fac_d, cond, cl_off_d, cd_off_d);

            const double dR_dphi     = out_d.residual.grad(0);
            const double inv_dR_dphi = -dR_dphi / (dR_dphi * dR_dphi + 1e-20);

            Eigen::Array<double, 10, 1> dphi_dx, dfz_dx, dft_dx, dvi_dx;
            const double dfz_dphi = out_d.df_z.grad(0);
            const double dft_dphi = out_d.df_theta.grad(0);
            const double dvi_dphi = out_d.vi_local.grad(0);

            for (int k = 0; k < 10; ++k) {
                dphi_dx[k] = out_d.residual.grad(k + 1) * inv_dR_dphi;
                dfz_dx[k]  = out_d.df_z.grad(k + 1)     + dfz_dphi * dphi_dx[k];
                dft_dx[k]  = out_d.df_theta.grad(k + 1) + dft_dphi * dphi_dx[k];
                dvi_dx[k]  = out_d.vi_local.grad(k + 1) + dvi_dphi * dphi_dx[k];
            }

            if (use_pitt_peters && v_inplane > 1e-5) {
                const double dc_rel_dvx = (c_psi * v_inplane - (cond.vx * c_psi + cond.vy * s_psi) * (cond.vx / v_inplane)) / v_inplane_sq;
                const double dc_rel_dvy = (s_psi * v_inplane - (cond.vx * c_psi + cond.vy * s_psi) * (cond.vy / v_inplane)) / v_inplane_sq;
                const double dwake_dvx_wind = dwake_draw * res.pitt_kx * r_norm * dc_rel_dvx;
                const double dwake_dvy_wind = dwake_draw * res.pitt_kx * r_norm * dc_rel_dvy;

                dfz_dx[4] += dfz_dx[7] * dwake_dvx_wind;
                dft_dx[4] += dft_dx[7] * dwake_dvx_wind;
                dvi_dx[4] += dvi_dx[7] * dwake_dvx_wind;

                dfz_dx[5] += dfz_dx[7] * dwake_dvy_wind;
                dft_dx[5] += dft_dx[7] * dwake_dvy_wind;
                dvi_dx[5] += dvi_dx[7] * dwake_dvy_wind;
            }

            sens.dT_dchord[i]  += w_ring * dfz_dx[0];
            sens.dT_dtwist[i]  += w_ring * dfz_dx[1];
            sens.dQ_dchord[i]  += w_ring * r_i * dft_dx[0];
            sens.dQ_dtwist[i]  += w_ring * r_i * dft_dx[1];
            sens.dFx_dchord[i] += w_ring * s_dir * s_psi * dft_dx[0];
            sens.dFx_dtwist[i] += w_ring * s_dir * s_psi * dft_dx[1];
            sens.dFy_dchord[i] += -w_ring * s_dir * c_psi * dft_dx[0];
            sens.dFy_dtwist[i] += -w_ring * s_dir * c_psi * dft_dx[1];
            sens.dMx_dchord[i] += w_ring * r_i * s_psi * dfz_dx[0];
            sens.dMx_dtwist[i] += w_ring * r_i * s_psi * dfz_dx[1];
            sens.dMy_dchord[i] += -w_ring * r_i * c_psi * dfz_dx[0];
            sens.dMy_dtwist[i] += -w_ring * r_i * c_psi * dfz_dx[1];

            dvi_mean_dchord[i] += w_area * dvi_dx[0];
            dvi_mean_dtwist[i] += w_area * dvi_dx[1];

            // 对局部微元 (i, j) 处二维翼型 Cl_2D (k=8) 与 Cd_2D (k=9) 的直接 IFT 偏导
            sens.dT_dcl(i, j)  = w_ring * dfz_dx[8];
            sens.dT_dcd(i, j)  = w_ring * dfz_dx[9];
            sens.dQ_dcl(i, j)  = w_ring * r_i * dft_dx[8];
            sens.dQ_dcd(i, j)  = w_ring * r_i * dft_dx[9];
            sens.dFx_dcl(i, j) = w_ring * s_dir * s_psi * dft_dx[8];
            sens.dFx_dcd(i, j) = w_ring * s_dir * s_psi * dft_dx[9];
            sens.dFy_dcl(i, j) = -w_ring * s_dir * c_psi * dft_dx[8];
            sens.dFy_dcd(i, j) = -w_ring * s_dir * c_psi * dft_dx[9];
            sens.dMx_dcl(i, j) = w_ring * r_i * s_psi * dfz_dx[8];
            sens.dMx_dcd(i, j) = w_ring * r_i * s_psi * dfz_dx[9];
            sens.dMy_dcl(i, j) = -w_ring * r_i * c_psi * dfz_dx[8];
            sens.dMy_dcd(i, j) = -w_ring * r_i * c_psi * dfz_dx[9];

            dvi_mean_dcl(i, j) = w_area * dvi_dx[8];
            dvi_mean_dcd(i, j) = w_area * dvi_dx[9];

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
                const double dwake_dkx    = dwake_draw * r_norm * c_rel;
                const double dfz_dkx_cell = dfz_dx[7] * dwake_dkx;
                const double dft_dkx_cell = dft_dx[7] * dwake_dkx;
                const double dvi_dkx_cell = dvi_dx[7] * dwake_dkx;

                dT_ring_dkx[i] += w_ring * dfz_dkx_cell;
                dT_dkx         += w_ring * dfz_dkx_cell;
                dQ_dkx         += w_ring * r_i * dft_dkx_cell;
                dFx_dkx        += w_ring * s_dir * s_psi * dft_dkx_cell;
                dFy_dkx        += -w_ring * s_dir * c_psi * dft_dkx_cell;
                dMx_dkx        += w_ring * r_i * s_psi * dfz_dkx_cell;
                dMy_dkx        += -w_ring * r_i * c_psi * dfz_dkx_cell;
                dvi_mean_dkx   += w_area * dvi_dkx_cell;
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

        const double raw_denom           = 1.0 - dkx_dvi * dvi_mean_dkx;
        const double self_consistent_fac = (std::abs(raw_denom) > 0.1) ? (1.0 / raw_denom) : 1.0;
        const double dkx_dvi_total       = dkx_dvi * self_consistent_fac;

        for (int i = 0; i < nr; ++i) {
            const double d_dr_dkx = dT_ring_dkx[i] / geom.dr[i];
            sens.dTr_dchord_diag[i] += d_dr_dkx * dkx_dvi_total * dvi_mean_dchord[i];
            sens.dTr_dtwist_diag[i] += d_dr_dkx * dkx_dvi_total * dvi_mean_dtwist[i];
        }

        sens.dT_dchord  += dT_dkx  * dkx_dvi_total * dvi_mean_dchord;
        sens.dT_dtwist  += dT_dkx  * dkx_dvi_total * dvi_mean_dtwist;
        sens.dQ_dchord  += dQ_dkx  * dkx_dvi_total * dvi_mean_dchord;
        sens.dQ_dtwist  += dQ_dkx  * dkx_dvi_total * dvi_mean_dtwist;
        sens.dFx_dchord += dFx_dkx * dkx_dvi_total * dvi_mean_dchord;
        sens.dFx_dtwist += dFx_dkx * dkx_dvi_total * dvi_mean_dtwist;
        sens.dFy_dchord += dFy_dkx * dkx_dvi_total * dvi_mean_dchord;
        sens.dFy_dtwist += dFy_dkx * dkx_dvi_total * dvi_mean_dtwist;
        sens.dMx_dchord += dMx_dkx * dkx_dvi_total * dvi_mean_dchord;
        sens.dMx_dtwist += dMx_dkx * dkx_dvi_total * dvi_mean_dtwist;
        sens.dMy_dchord += dMy_dkx * dkx_dvi_total * dvi_mean_dchord;
        sens.dMy_dtwist += dMy_dkx * dkx_dvi_total * dvi_mean_dtwist;

        sens.dT_dcl  += dT_dkx  * dkx_dvi_total * dvi_mean_dcl;
        sens.dT_dcd  += dT_dkx  * dkx_dvi_total * dvi_mean_dcd;
        sens.dQ_dcl  += dQ_dkx  * dkx_dvi_total * dvi_mean_dcl;
        sens.dQ_dcd  += dQ_dkx  * dkx_dvi_total * dvi_mean_dcd;
        sens.dFx_dcl += dFx_dkx * dkx_dvi_total * dvi_mean_dcl;
        sens.dFx_dcd += dFx_dkx * dkx_dvi_total * dvi_mean_dcd;
        sens.dFy_dcl += dFy_dkx * dkx_dvi_total * dvi_mean_dcl;
        sens.dFy_dcd += dFy_dkx * dkx_dvi_total * dvi_mean_dcd;
        sens.dMx_dcl += dMx_dkx * dkx_dvi_total * dvi_mean_dcl;
        sens.dMx_dcd += dMx_dkx * dkx_dvi_total * dvi_mean_dcd;
        sens.dMy_dcl += dMy_dkx * dkx_dvi_total * dvi_mean_dcl;
        sens.dMy_dcd += dMy_dkx * dkx_dvi_total * dvi_mean_dcd;

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
    sens.dP_dcl    = cond.omega * sens.dQ_dcl;
    sens.dP_dcd    = cond.omega * sens.dQ_dcd;

    for (int c_idx = 0; c_idx < 5; ++c_idx) {
        sens.dForces_dCond(2, c_idx) = cond.omega * sens.dForces_dCond(1, c_idx);
    }
    sens.dForces_dCond(2, 0) += res.torque;

    // 1. 对二维参数化 B 样条控制点纵坐标值 (chord_ctrl, twist_ctrl) 的链式法则解析梯度
    const MatrixXd BcT = geom.chord_basis.transpose();
    const MatrixXd BtT = geom.twist_basis.transpose();

    sens.dT_dchord_ctrl  = (BcT * sens.dT_dchord.matrix()).array();
    sens.dT_dtwist_ctrl  = (BtT * sens.dT_dtwist.matrix()).array();
    sens.dQ_dchord_ctrl  = (BcT * sens.dQ_dchord.matrix()).array();
    sens.dQ_dtwist_ctrl  = (BtT * sens.dQ_dtwist.matrix()).array();
    sens.dP_dchord_ctrl  = (BcT * sens.dP_dchord.matrix()).array();
    sens.dP_dtwist_ctrl  = (BtT * sens.dP_dtwist.matrix()).array();
    sens.dFx_dchord_ctrl = (BcT * sens.dFx_dchord.matrix()).array();
    sens.dFx_dtwist_ctrl = (BtT * sens.dFx_dtwist.matrix()).array();
    sens.dFy_dchord_ctrl = (BcT * sens.dFy_dchord.matrix()).array();
    sens.dFy_dtwist_ctrl = (BtT * sens.dFy_dtwist.matrix()).array();
    sens.dMx_dchord_ctrl = (BcT * sens.dMx_dchord.matrix()).array();
    sens.dMx_dtwist_ctrl = (BtT * sens.dMx_dtwist.matrix()).array();
    sens.dMy_dchord_ctrl = (BcT * sens.dMy_dchord.matrix()).array();
    sens.dMy_dtwist_ctrl = (BtT * sens.dMy_dtwist.matrix()).array();

    // 2. 对二维参数化 B 样条控制点无量纲径向横坐标 (chord_ctrl_r_norm, twist_ctrl_r_norm) 的隐函数解析梯度
    const ArrayXd neg_sc = -geom.dchord_dr_norm;
    const ArrayXd neg_st = -geom.dtwist_dr_norm;

    sens.dT_dchord_ctrl_r  = (BcT * (neg_sc * sens.dT_dchord).matrix()).array();
    sens.dT_dtwist_ctrl_r  = (BtT * (neg_st * sens.dT_dtwist).matrix()).array();
    sens.dQ_dchord_ctrl_r  = (BcT * (neg_sc * sens.dQ_dchord).matrix()).array();
    sens.dQ_dtwist_ctrl_r  = (BtT * (neg_st * sens.dQ_dtwist).matrix()).array();
    sens.dP_dchord_ctrl_r  = (BcT * (neg_sc * sens.dP_dchord).matrix()).array();
    sens.dP_dtwist_ctrl_r  = (BtT * (neg_st * sens.dP_dtwist).matrix()).array();
    sens.dFx_dchord_ctrl_r = (BcT * (neg_sc * sens.dFx_dchord).matrix()).array();
    sens.dFx_dtwist_ctrl_r = (BtT * (neg_st * sens.dFx_dtwist).matrix()).array();
    sens.dFy_dchord_ctrl_r = (BcT * (neg_sc * sens.dFy_dchord).matrix()).array();
    sens.dFy_dtwist_ctrl_r = (BtT * (neg_st * sens.dFy_dtwist).matrix()).array();
    sens.dMx_dchord_ctrl_r = (BcT * (neg_sc * sens.dMx_dchord).matrix()).array();
    sens.dMx_dtwist_ctrl_r = (BtT * (neg_st * sens.dMx_dtwist).matrix()).array();
    sens.dMy_dchord_ctrl_r = (BcT * (neg_sc * sens.dMy_dchord).matrix()).array();
    sens.dMy_dtwist_ctrl_r = (BtT * (neg_st * sens.dMy_dtwist).matrix()).array();
}

} // namespace bemt
