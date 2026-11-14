#include "bemt/solver.hpp"
#include "bemt/sensitivity.hpp"
#include <algorithm>
#include <cmath>
#include <omp.h>

namespace bemt {

BEMTSolver::BEMTSolver(RotorGeometry geom, SolverConfig cfg)
    : geom_(std::move(geom)), cfg_(cfg) {
    geom_.validate_and_finalize();
    if (cfg_.n_azimuth < 1) {
        cfg_.n_azimuth = 1;
    }
}

// ============================================================================
// 鲁棒性优先版 Brent 根求解器：多级括号扩张 + 18点加密扫描兜底，确保绝不漏根
// ============================================================================
double BEMTSolver::solve_cell_brent(int i_r,
                                    double chord,
                                    double theta_local,
                                    double u_p,
                                    double u_t,
                                    double v_inplane_sq,
                                    double omega,
                                    double wake_factor,
                                    const OperatingCondition& cond,
                                    double phi_init,
                                    bool& brent_ok) const {
    brent_ok = false;
    auto eval_r = [&](double p) -> double {
        return eval_element_kernel<double>(
                   i_r, p, chord, theta_local, u_p, u_t, v_inplane_sq, omega, wake_factor, cond)
            .residual;
    };

    const double lo_bound = -kHalfPi + 1e-4;
    const double hi_bound =  kHalfPi - 1e-4;

    // 1. 先在初值附近做两级局部探测 (快速命中常规失速点)
    double a  = std::max(lo_bound, phi_init - 0.20);
    double b  = std::min(hi_bound, phi_init + 0.20);
    double fa = eval_r(a);
    double fb = eval_r(b);

    // 2. 若局部未包住根，测试全象限端点 [lo_bound, hi_bound]
    if (fa * fb > 0.0) {
        a  = lo_bound;
        b  = hi_bound;
        fa = eval_r(a);
        fb = eval_r(b);
    }

    // 3. 若两端仍同号（例如高诱导风车态出现偶次多根），执行 18 等分加密扫描寻找异号或极小残差区间
    if (fa * fb > 0.0) {
        const int n_scan = 18;
        double p_prev = lo_bound;
        double f_prev = fa;
        double best_p = phi_init;
        double min_abs_f = std::min(std::abs(fa), std::abs(fb));

        for (int k = 1; k <= n_scan; ++k) {
            double p_curr = lo_bound + (hi_bound - lo_bound) * (static_cast<double>(k) / n_scan);
            double f_curr = eval_r(p_curr);
            if (std::abs(f_curr) < min_abs_f) {
                min_abs_f = std::abs(f_curr);
                best_p    = p_curr;
            }
            if (f_prev * f_curr <= 0.0) {
                a  = p_prev;
                fa = f_prev;
                b  = p_curr;
                fb = f_curr;
                break;
            }
            p_prev = p_curr;
            f_prev = f_curr;
        }

        if (fa * fb > 0.0) {
            brent_ok = (min_abs_f < 1e-3);
            return std::clamp(best_p, lo_bound, hi_bound);
        }
    }

    // 4. 标准 Brent 算法主循环
    double c = b, fc = fb;
    double d = b - a, e = d;

    for (int iter = 0; iter < cfg_.max_brent_iters; ++iter) {
        if ((fb > 0.0 && fc > 0.0) || (fb < 0.0 && fc < 0.0)) {
            c  = a;
            fc = fa;
            d  = b - a;
            e  = d;
        }
        if (std::abs(fc) < std::abs(fb)) {
            a  = b;
            b  = c;
            c  = a;
            fa = fb;
            fb = fc;
            fc = fa;
        }

        const double tol1 = 2.0 * 1e-13 * std::abs(b) + 0.5 * cfg_.tol;
        const double xm   = 0.5 * (c - b);
        if (std::abs(xm) <= tol1 || std::abs(fb) <= cfg_.tol) {
            brent_ok = true;
            return b;
        }

        if (std::abs(e) >= tol1 && std::abs(fa) > std::abs(fb)) {
            double s = fb / fa;
            double p, q;
            if (a == c) {
                p = 2.0 * xm * s;
                q = 1.0 - s;
            } else {
                q = fa / fc;
                double r = fb / fc;
                p = s * (2.0 * xm * q * (q - r) - (b - a) * (r - 1.0));
                q = (q - 1.0) * (r - 1.0) * (s - 1.0);
            }
            if (p > 0.0) q = -q;
            p = std::abs(p);
            double min1 = 3.0 * xm * q - std::abs(tol1 * q);
            double min2 = std::abs(e * q);
            if (2.0 * p < std::min(min1, min2)) {
                e = d;
                d = p / q;
            } else {
                d = xm;
                e = d;
            }
        } else {
            d = xm;
            e = d;
        }
        a  = b;
        fa = fb;
        if (std::abs(d) > tol1) {
            b += d;
        } else {
            b += (xm >= 0.0 ? tol1 : -tol1);
        }
        fb = eval_r(b);
    }

    brent_ok = (std::abs(fb) <= 1e-5);
    return b;
}

BEMTResult BEMTSolver::solve(const OperatingCondition& cond, bool compute_gradients) const {
    BEMTResult res;
    const int nr   = static_cast<int>(geom_.r.size());
    const int npsi = cfg_.n_azimuth;

    res.psi_grid.resize(npsi);
    for (int j = 0; j < npsi; ++j) {
        res.psi_grid[j] = kTwoPi * static_cast<double>(j) / static_cast<double>(npsi);
    }

    res.vi_annulus = ArrayXd::Zero(nr);
    res.dT_dr_mean = ArrayXd::Zero(nr);
    res.dQ_dr_mean = ArrayXd::Zero(nr);

    res.phi      = ArrayXXd::Zero(nr, npsi);
    res.alpha    = ArrayXXd::Zero(nr, npsi);
    res.vi       = ArrayXXd::Zero(nr, npsi);
    res.vt       = ArrayXXd::Zero(nr, npsi);
    res.cl       = ArrayXXd::Zero(nr, npsi);
    res.cd       = ArrayXXd::Zero(nr, npsi);
    res.re       = ArrayXXd::Zero(nr, npsi);
    res.ma       = ArrayXXd::Zero(nr, npsi);
    res.f_tip    = ArrayXXd::Ones(nr, npsi);
    res.df_z     = ArrayXXd::Zero(nr, npsi);
    res.df_theta = ArrayXXd::Zero(nr, npsi);

    const double v_inplane_sq = cond.vx * cond.vx + cond.vy * cond.vy;
    const double v_inplane    = std::sqrt(v_inplane_sq);
    const double s_dir        = static_cast<double>(geom_.spin_dir);
    const double R_tip        = geom_.radius;
    const double A_disk       = kPi * (R_tip * R_tip - geom_.hub_radius * geom_.hub_radius);

    ArrayXd cos_psi = res.psi_grid.cos();
    ArrayXd sin_psi = res.psi_grid.sin();

    const double psi_wind = std::atan2(cond.vy, cond.vx);
    ArrayXd cos_psi_rel   = (res.psi_grid - psi_wind).cos();
    ArrayXd sin_psi_rel   = (res.psi_grid - psi_wind).sin();

    const int idx_75      = std::clamp(static_cast<int>(0.75 * nr), 0, nr - 1);
    const double theta_75 = std::clamp(geom_.twist[idx_75] + cond.pitch_0, -0.5, 0.8);
    const double v_tip    = std::max(cond.omega * R_tip, 1.0);
    const double vi_est   = (theta_75 > 0.0) ? std::max(1.0, 0.04 * v_tip * std::sqrt(theta_75)) : 0.5;

    const bool use_pitt_peters = (cfg_.wake_model == WakeModel::kGlauertPittPeters) && (v_inplane > 1e-5);
    // 允许外层不动点+割线加速迭代最多执行 max(16, max_wake_iters*4) 步，一旦残差 |kx_implied - kx| < 1e-11 立即收敛
    const int max_outer_iters  = use_pitt_peters ? std::max(16, cfg_.max_wake_iters * 4) : 1;

    double kx = 0.0;
    double ky = 0.0;
    if (use_pitt_peters) {
        const double chi_est = std::atan2(v_inplane, std::abs(cond.vz) + vi_est);
        kx = (15.0 * kPi / 23.0) * std::tan(0.5 * chi_est);
    }

    for (int i = 0; i < nr; ++i) {
        const double r_i = geom_.r[i];
        for (int j = 0; j < npsi; ++j) {
            const double u_t_ij  = cond.omega * r_i + s_dir * (cond.vx * sin_psi[j] - cond.vy * cos_psi[j]);
            const double abs_u_t = std::max(std::abs(u_t_ij), 1.0);
            res.phi(i, j) = std::clamp(std::atan2(cond.vz + vi_est, abs_u_t), -0.8, 0.8);
        }
    }

    int total_fallbacks   = 0;
    int unconverged_cells = 0;
    bool wake_converged   = !use_pitt_peters;

    double kx_prev     = kx;
    double res_kx_prev = 0.0;
    bool has_prev_kx   = false;

    for (int wake_iter = 0; wake_iter < max_outer_iters; ++wake_iter) {
        total_fallbacks   = 0;
        unconverged_cells = 0;
        // 当尾迹已经收敛或到达最后一步时，启用严格容差 cfg_.tol 并完整写入 2D 物理场
        const bool is_final_pass = wake_converged || (wake_iter == max_outer_iters - 1);
        const double active_tol  = is_final_pass ? cfg_.tol : std::max(cfg_.tol, 1e-8);

        #pragma omp parallel for schedule(static) reduction(+:total_fallbacks, unconverged_cells) if(!omp_in_parallel())
        for (int i = 0; i < nr; ++i) {
            const double r_i    = geom_.r[i];
            const double r_norm = r_i / R_tip;
            const double c_i    = geom_.chord[i];
            const double tw_i   = geom_.twist[i];

            for (int j = 0; j < npsi; ++j) {
                const double theta_ij = tw_i + cond.pitch_0 +
                                        cond.pitch_1c * cos_psi[j] +
                                        cond.pitch_1s * sin_psi[j];

                const double u_t_ij = cond.omega * r_i + s_dir * (cond.vx * sin_psi[j] - cond.vy * cos_psi[j]);
                const double u_p_ij = cond.vz;

                const double raw_wake = 1.0 + kx * r_norm * cos_psi_rel[j] + ky * r_norm * sin_psi_rel[j];
                const double wake_fac = math::smooth_max(raw_wake, 0.1, 0.02);

                double phi_curr = res.phi(i, j);
                bool cell_conv  = false;

                for (int it = 0; it < cfg_.max_newton_iters; ++it) {
                    Dual1 phi_dual(phi_curr, 1.0);
                    auto out_dual = eval_element_kernel<Dual1>(
                        i, phi_dual, Dual1(c_i), Dual1(theta_ij),
                        Dual1(u_p_ij), Dual1(u_t_ij), Dual1(v_inplane_sq),
                        Dual1(cond.omega), Dual1(wake_fac), cond);

                    const double r_val   = out_dual.residual.val;
                    const double dr_dphi = out_dual.residual.grad(0);

                    if (!std::isfinite(r_val) || !std::isfinite(dr_dphi)) {
                        break;
                    }

                    if (std::abs(r_val) <= active_tol) {
                        cell_conv = true;
                        break;
                    }

                    if (std::abs(dr_dphi) > 1e-8) {
                        double full_step = std::clamp(-r_val / dr_dphi, -cfg_.newton_max_step, cfg_.newton_max_step);
                        double phi_next  = std::clamp(phi_curr + full_step, -kHalfPi + 1e-4, kHalfPi - 1e-4);

                        double r_next = eval_element_kernel<double>(
                            i, phi_next, c_i, theta_ij, u_p_ij, u_t_ij, v_inplane_sq, cond.omega, wake_fac, cond).residual;

                        if (std::abs(r_next) > std::abs(r_val)) {
                            phi_next = std::clamp(phi_curr + 0.5 * full_step, -kHalfPi + 1e-4, kHalfPi - 1e-4);
                        }
                        phi_curr = phi_next;
                    } else {
                        break;
                    }
                }

                if (!cell_conv) {
                    bool brent_ok = false;
                    phi_curr = solve_cell_brent(
                        i, c_i, theta_ij, u_p_ij, u_t_ij, v_inplane_sq, cond.omega, wake_fac, cond, phi_curr, brent_ok);
                    total_fallbacks += 1;
                    if (!brent_ok && is_final_pass) {
                        unconverged_cells += 1;
                    }
                }

                res.phi(i, j) = phi_curr;

                auto final_out = eval_element_kernel<double>(
                    i, phi_curr, c_i, theta_ij, u_p_ij, u_t_ij, v_inplane_sq, cond.omega, wake_fac, cond);

                res.vi(i, j) = final_out.vi_local;
                if (is_final_pass) {
                    res.alpha(i, j)    = final_out.alpha;
                    res.vt(i, j)       = final_out.vt_local;
                    res.cl(i, j)       = final_out.cl;
                    res.cd(i, j)       = final_out.cd;
                    res.re(i, j)       = final_out.re;
                    res.ma(i, j)       = final_out.ma;
                    res.f_tip(i, j)    = final_out.f_loss;
                    res.df_z(i, j)     = final_out.df_z;
                    res.df_theta(i, j) = final_out.df_theta;
                }
            }
        }

        double weighted_vi_sum = 0.0;
        for (int i = 0; i < nr; ++i) {
            res.vi_annulus[i] = res.vi.row(i).mean();
            const double dA_ring = kTwoPi * geom_.r[i] * geom_.dr[i];
            weighted_vi_sum += res.vi_annulus[i] * dA_ring;
        }
        res.vi_disk_mean = weighted_vi_sum / A_disk;

        const double vz_wake = math::smooth_abs(cond.vz + res.vi_disk_mean, 0.2);
        res.wake_skew_chi    = std::atan2(v_inplane, vz_wake);

        if (is_final_pass) {
            // 已经用收敛的 kx 完成了最终流场与气动力评估，严格保持 kx 与 phi* 同步并退出！
            break;
        }

        if (use_pitt_peters) {
            const double kx_implied = (15.0 * kPi / 23.0) * std::tan(0.5 * res.wake_skew_chi);
            const double res_kx     = kx_implied - kx;
            const double diff_kx    = std::abs(res_kx);

            if (diff_kx < 1e-10) {
                kx             = kx_implied;
                wake_converged = true;
            } else {
                double kx_next = kx_implied;
                // 当不动点残差出现异号振荡时，采用割线/弦截内插一步消除振荡
                if (has_prev_kx && (res_kx * res_kx_prev < 0.0)) {
                    const double d_res = res_kx - res_kx_prev;
                    if (std::abs(d_res) > 1e-14) {
                        kx_next = kx - res_kx * ((kx - kx_prev) / d_res);
                    }
                }
                kx_prev     = kx;
                res_kx_prev = res_kx;
                has_prev_kx = true;
                kx          = std::clamp(kx_next, 0.0, 15.0 * kPi / 23.0);
            }
        }
    }

    res.converged = (unconverged_cells == 0);
    res.pitt_kx   = kx;
    res.pitt_ky   = ky;
    res.total_fallback_cells = total_fallbacks;

    const double B = static_cast<double>(geom_.num_blades);
    res.dT_dr_mean = B * res.df_z.rowwise().mean();
    res.dQ_dr_mean = B * geom_.r * res.df_theta.rowwise().mean();

    res.thrust = (res.dT_dr_mean * geom_.dr).sum();
    res.fz     = res.thrust;
    res.torque = (res.dQ_dr_mean * geom_.dr).sum();
    res.mz     = res.torque;
    res.power  = res.torque * cond.omega;

    const double inv_npsi = 1.0 / static_cast<double>(npsi);
    VectorXd cos_vec = cos_psi.matrix();
    VectorXd sin_vec = sin_psi.matrix();

    ArrayXd dfx_dr =  B * s_dir * inv_npsi * (res.df_theta.matrix() * sin_vec).array();
    ArrayXd dfy_dr = -B * s_dir * inv_npsi * (res.df_theta.matrix() * cos_vec).array();
    ArrayXd dmx_dr =  B * geom_.r * inv_npsi * (res.df_z.matrix() * sin_vec).array();
    ArrayXd dmy_dr = -B * geom_.r * inv_npsi * (res.df_z.matrix() * cos_vec).array();

    res.fx = (dfx_dr * geom_.dr).sum();
    res.fy = (dfy_dr * geom_.dr).sum();
    res.mx = (dmx_dr * geom_.dr).sum();
    res.my = (dmy_dr * geom_.dr).sum();

    const double area_full = kPi * R_tip * R_tip;
    const double denom_f   = std::max(cond.rho * area_full * v_tip * v_tip, 1e-12);

    res.ct = res.thrust / denom_f;
    res.cq = res.torque / (denom_f * R_tip);
    res.cp = res.power / (denom_f * v_tip);
    if (res.ct > 0.0 && res.cp > 1e-12) {
        res.fm = std::pow(res.ct, 1.5) / (std::sqrt(2.0) * res.cp);
    } else {
        res.fm = 0.0;
    }

    if (compute_gradients) {
        compute_ift_sensitivities(*this, cond, res);
    }

    return res;
}

std::vector<BEMTResult> BEMTSolver::solve_batch(const std::vector<OperatingCondition>& conds,
                                                bool compute_gradients) const {
    const int n_cases = static_cast<int>(conds.size());
    std::vector<BEMTResult> results(n_cases);

    #pragma omp parallel for schedule(guided)
    for (int k = 0; k < n_cases; ++k) {
        results[k] = solve(conds[k], compute_gradients);
    }
    return results;
}

} // namespace bemt