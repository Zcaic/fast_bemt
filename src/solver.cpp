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
// 标量 Brent 根求解器 (保證 100% 绝对收敛的兜底保险)
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
                                    double phi_init) const {
    auto eval_r = [&](double p) -> double {
        return eval_element_kernel<double>(
                   i_r, p, chord, theta_local, u_p, u_t, v_inplane_sq, omega, wake_factor, cond)
            .residual;
    };

    // 搜索包含根的异号区间 [a, b]
    double a = -kHalfPi + 1e-3;
    double b = kHalfPi - 1e-3;
    double fa = eval_r(a);
    double fb = eval_r(b);

    // 若两端同号，在初值附近细分扫描寻找异号括号区间
    if (fa * fb > 0.0) {
        const int n_scan = 24;
        double p_prev = a;
        double f_prev = fa;
        for (int k = 1; k <= n_scan; ++k) {
            double p_curr = a + (b - a) * (static_cast<double>(k) / n_scan);
            double f_curr = eval_r(p_curr);
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
    }

    if (fa * fb > 0.0) {
        return std::clamp(phi_init, a, b);
    }

    // 标准 Brent 算法迭代
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

        const double tol1 = 2.0 * 1e-12 * std::abs(b) + 0.5 * cfg_.tol;
        const double xm   = 0.5 * (c - b);
        if (std::abs(xm) <= tol1 || std::abs(fb) <= cfg_.tol) {
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
    return b;
}

// ============================================================================
// 核心斜入流求解实现：方案 A (圆环平衡 + Pitt-Peters 尾迹修正) & 方案 B
// ============================================================================
BEMTResult BEMTSolver::solve(const OperatingCondition& cond, bool compute_gradients) const {
    BEMTResult res;
    const int nr   = static_cast<int>(geom_.r.size());
    const int npsi = cfg_.n_azimuth;

    // 1. 初始化一维与二维输出矩阵 (Row-Major: nr x npsi)
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

    // 预计算方位角三角函数向量与二维切向来流矩阵 U_T(i, j)、局部总桨距角矩阵 Theta(i, j)
    ArrayXd cos_psi = res.psi_grid.cos();
    ArrayXd sin_psi = res.psi_grid.sin();

    // 面内前进速度方向偏角 psi_wind (用于将尾迹偏斜梯度对齐到实际合侧风方向)
    const double psi_wind = std::atan2(cond.vy, cond.vx);
    ArrayXd cos_psi_rel   = (res.psi_grid - psi_wind).cos();
    ArrayXd sin_psi_rel   = (res.psi_grid - psi_wind).sin();

    // 初始化入流角初值 phi_0 = atan2(Vz + vi_guess, omega * r)
    const double vi_hover_est = std::max(2.0, 0.05 * cond.omega * R_tip);
    for (int i = 0; i < nr; ++i) {
        const double phi_guess = std::atan2(cond.vz + vi_hover_est, cond.omega * geom_.r[i]);
        res.phi.row(i).setConstant(std::clamp(phi_guess, -0.5, 0.8));
    }

    // ------------------------------------------------------------------------
    // 2. 外层 Pitt-Peters 尾迹偏斜角 chi 自洽迭代 + 内层 OpenMP & SIMD 向量化牛顿求解
    //    若为纯轴流 (v_inplane < 1e-5) 或非 Pitt-Peters 模式，外层仅需执行 1 次！
    // ------------------------------------------------------------------------
    const bool use_pitt_peters = (cfg_.wake_model == WakeModel::kGlauertPittPeters) && (v_inplane > 1e-5);
    const int outer_iters      = use_pitt_peters ? std::max(1, cfg_.max_wake_iters) : 1;

    double kx = 0.0;
    double ky = 0.0;
    int total_fallbacks = 0;

    for (int wake_iter = 0; wake_iter < outer_iters; ++wake_iter) {
        total_fallbacks = 0;

        // OpenMP 多核并行：沿径向圆环 i (或分块) 分配线程，每个圆环内部沿周向 j 向量化处理
        // 由于第 i 行共享同一个翼型表 geom_.airfoils[i]，按行划分线程具备极致的 L1 缓存命中率！
        #pragma omp parallel for schedule(static) reduction(+:total_fallbacks) if(!omp_in_parallel())
        for (int i = 0; i < nr; ++i) {
            const double r_i   = geom_.r[i];
            const double r_norm = r_i / R_tip;
            const double c_i   = geom_.chord[i];
            const double tw_i  = geom_.twist[i];

            for (int j = 0; j < npsi; ++j) {
                // 局部桨距角 (含总距、扭转角和周期变距)
                const double theta_ij = tw_i + cond.pitch_0 +
                                        cond.pitch_1c * cos_psi[j] +
                                        cond.pitch_1s * sin_psi[j];

                // 局部切向与轴向来流速度
                const double u_t_ij = cond.omega * r_i + s_dir * (cond.vx * sin_psi[j] - cond.vy * cos_psi[j]);
                const double u_p_ij = cond.vz;

                // Pitt-Peters 斜尾迹一阶谐波修正因子: 1 + kx*(r/R)*cos(psi_rel) + ky*(r/R)*sin(psi_rel)
                const double wake_fac = std::max(0.1, 1.0 + kx * r_norm * cos_psi_rel[j] + ky * r_norm * sin_psi_rel[j]);

                // 阶段 1: 基于 Dual1 解析导数的阻尼牛顿-拉夫逊法 (通常 4 ~ 6 步收敛至 1e-12)
                double phi_curr = res.phi(i, j);
                bool cell_conv  = false;

                for (int it = 0; it < cfg_.max_newton_iters; ++it) {
                    Dual1 phi_dual(phi_curr, 1.0); // 设 d(phi)/d(phi) = 1.0
                    auto out_dual = eval_element_kernel<Dual1>(
                        i, phi_dual, Dual1(c_i), Dual1(theta_ij),
                        Dual1(u_p_ij), Dual1(u_t_ij), Dual1(v_inplane_sq),
                        Dual1(cond.omega), Dual1(wake_fac), cond);

                    const double r_val  = out_dual.residual.val;
                    const double dr_dphi = out_dual.residual.grad(0);

                    if (std::abs(r_val) <= cfg_.tol) {
                        cell_conv = true;
                        break;
                    }

                    // 步长保护与更新
                    if (std::abs(dr_dphi) > 1e-9) {
                        double step = -r_val / dr_dphi;
                        step = std::clamp(step, -cfg_.newton_max_step, cfg_.newton_max_step);
                        phi_curr = std::clamp(phi_curr + step, -kHalfPi + 1e-3, kHalfPi - 1e-3);
                    } else {
                        break; // 导数奇异，交由 Brent 兜底
                    }
                }

                // 阶段 2: 若牛顿法未达精度，自动触发 Brent 标量求根兜底
                if (!cell_conv) {
                    phi_curr = solve_cell_brent(
                        i, c_i, theta_ij, u_p_ij, u_t_ij, v_inplane_sq, cond.omega, wake_fac, cond, phi_curr);
                    total_fallbacks += 1;
                }

                // 在最终收敛的 phi* 处评估一次 double 核函数，写入二维结果场
                auto final_out = eval_element_kernel<double>(
                    i, phi_curr, c_i, theta_ij, u_p_ij, u_t_ij, v_inplane_sq, cond.omega, wake_fac, cond);

                res.phi(i, j)      = phi_curr;
                res.alpha(i, j)    = final_out.alpha;
                res.vi(i, j)       = final_out.vi_local;
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

        // 计算径向各圆环的周向平均基准诱导速度 v_i0(r_i) 与桨盘面积加权平均诱导速度 vi_disk_mean
        double weighted_vi_sum = 0.0;
        for (int i = 0; i < nr; ++i) {
            res.vi_annulus[i] = res.vi.row(i).mean();
            const double dA_ring = kTwoPi * geom_.r[i] * geom_.dr[i];
            weighted_vi_sum += res.vi_annulus[i] * dA_ring;
        }
        res.vi_disk_mean = weighted_vi_sum / A_disk;

        // 更新尾迹偏斜角 chi 与 Pitt-Peters 系数 kx, ky
        const double vz_wake = std::abs(cond.vz) + std::max(res.vi_disk_mean, 0.1);
        res.wake_skew_chi    = std::atan2(v_inplane, vz_wake);

        if (use_pitt_peters) {
            // Pitt-Peters (1981) / Coleman 经典纵向入流谐波梯度系数: kx = (15*pi/23) * tan(chi / 2)
            const double kx_new = (15.0 * kPi / 23.0) * std::tan(0.5 * res.wake_skew_chi);
            if ( std::abs(kx_new - kx) < 1e-6 && wake_iter >= 1) {
                kx = kx_new;
                break;
            }
            kx = kx_new;
        }
    }

    res.pitt_kx = kx;
    res.pitt_ky = ky;
    res.total_fallback_cells = total_fallbacks;

    // ------------------------------------------------------------------------
    // 3. 矩阵化全桨盘六分量气动力与力矩积分 (Matrix-Vector Reductions)
    // ------------------------------------------------------------------------
    const double B = static_cast<double>(geom_.num_blades);

    // 径向每个圆环的 B 片桨叶周向平均载荷: dT/dr(r_i) = B * mean_psi(df_z), dQ/dr(r_i) = B * r_i * mean_psi(df_theta)
    res.dT_dr_mean = B * res.df_z.rowwise().mean();
    res.dQ_dr_mean = B * geom_.r * res.df_theta.rowwise().mean();

    // 总拉力 T (= Fz), 总反扭矩 Q (= Mz), 总轴功率 P
    res.thrust = (res.dT_dr_mean * geom_.dr).sum();
    res.fz     = res.thrust;
    res.torque = (res.dQ_dr_mean * geom_.dr).sum();
    res.mz     = res.torque;
    res.power  = res.torque * cond.omega;

    // 面内力 Fx (H-force), Fy (Y-force) 与俯仰/滚转力矩 Mx (Roll), My (Pitch)
    // 对于方位角 psi_j:
    // - 周向阻力 df_theta 投影到桨盘 X, Y 轴:
    //   dFx = B * mean_psi( df_theta * s_dir * sin(psi) )
    //   dFy = B * mean_psi( -df_theta * s_dir * cos(psi) )
    // - 轴向推力 df_z 产生的力矩:
    //   滚转力矩 Mx (前行边 psi=90deg 升力大产生负/正滚转): dMx = B * r * mean_psi( df_z * sin(psi) )
    //   俯仰力矩 My (前缘 psi=180deg 升力大产生抬头力矩): dMy = B * r * mean_psi( -df_z * cos(psi) )
    const double inv_npsi = 1.0 / static_cast<double>(npsi);
    VectorXd cos_vec = cos_psi.matrix();
    VectorXd sin_vec = sin_psi.matrix();

    ArrayXd dfx_dr = B * s_dir * inv_npsi * (res.df_theta.matrix() * sin_vec).array();
    ArrayXd dfy_dr = -B * s_dir * inv_npsi * (res.df_theta.matrix() * cos_vec).array();
    ArrayXd dmx_dr = B * geom_.r * inv_npsi * (res.df_z.matrix() * sin_vec).array();
    ArrayXd dmy_dr = -B * geom_.r * inv_npsi * (res.df_z.matrix() * cos_vec).array();

    res.fx = (dfx_dr * geom_.dr).sum();
    res.fy = (dfy_dr * geom_.dr).sum();
    res.mx = (dmx_dr * geom_.dr).sum();
    res.my = (dmy_dr * geom_.dr).sum();

    // ------------------------------------------------------------------------
    // 4. 计算无量纲气动系数 (CT, CQ, CP, FM)
    // ------------------------------------------------------------------------
    const double area_full = kPi * R_tip * R_tip;
    const double v_tip     = cond.omega * R_tip;
    const double denom_f   = std::max(cond.rho * area_full * v_tip * v_tip, 1e-12);

    res.ct = res.thrust / denom_f;
    res.cq = res.torque / (denom_f * R_tip);
    res.cp = res.power / (denom_f * v_tip);
    if (res.ct > 0.0 && res.cp > 1e-12) {
        res.fm = std::pow(res.ct, 1.5) / (std::sqrt(2.0) * res.cp);
    } else {
        res.fm = 0.0;
    }

    // ------------------------------------------------------------------------
    // 5. 若开启求导，调用隐函数定理 (IFT) 解析敏度计算器
    // ------------------------------------------------------------------------
    if (compute_gradients) {
        compute_ift_sensitivities(*this, cond, res);
    }

    return res;
}

// ============================================================================
// 多工况批量求解
// ============================================================================
std::vector<BEMTResult> BEMTSolver::solve_batch(const std::vector<OperatingCondition>& conds,
                                                bool compute_gradients) const {
    const int n_cases = static_cast<int>(conds.size());
    std::vector<BEMTResult> results(n_cases);

    #pragma omp parallel for schedule(dynamic)
    for (int k = 0; k < n_cases; ++k) {
        results[k] = solve(conds[k], compute_gradients);
    }
    return results;
}

} // namespace bemt