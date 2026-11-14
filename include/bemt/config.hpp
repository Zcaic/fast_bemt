#pragma once

#include "bemt/airfoil.hpp"
#include "bemt/bspline.hpp"
#include "bemt/dual.hpp"
#include "bemt/types.hpp"
#include <algorithm>
#include <cmath>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace bemt {

// ============================================================================
// 1. 旋翼/螺旋桨几何参数结构体 RotorGeometry
// ============================================================================
struct RotorGeometry {
    int num_blades     = 2;     // 桨叶数 B
    double radius      = 1.0;   // 旋翼叶尖半径 R [m]
    double hub_radius  = 0.1;   // 桨毂半径 R_hub [m]
    int spin_dir       = 1;     // 旋向: +1 逆时针(从上往下看), -1 顺时针
    int n_radial       = 20;    // 径向离散叶素截面数 N_r (若 r 为空，自动按 n_radial 等分 [R_hub, R])
    int bspline_degree = 3;     // B 样条曲线阶数 p (与 chord_curve.degree 同步)

    ArrayXd r;                  // 径向离散叶素中心坐标 [m], 长度 N_r (可由 n_radial 自动生成)
    ArrayXd dr;                 // 径向积分微元宽度 [m], 长度 N_r (可自动生成)

    // 独立实例化的弦长与扭转分布 B 样条曲线对象
    BSplineCurve chord_curve;   // 弦长分布二维参数化 B 样条曲线: (r/R(u), chord(u))
    BSplineCurve twist_curve;   // 扭转分布二维参数化 B 样条曲线: (r/R(u), twist(u))

    // 与 chord_curve / twist_curve 同步的控制点字段
    ArrayXd chord_ctrl_r_norm;  // 弦长 B 样条控制点的无量纲径向坐标 (r/R)_k [-], 长度 N_c_chord
    ArrayXd chord_ctrl;         // 弦长 B 样条控制点的弦长值 c_k [m], 长度 N_c_chord
    ArrayXd twist_ctrl_r_norm;  // 扭转角 B 样条控制点的无量纲径向坐标 (r/R)_k [-], 长度 N_c_twist
    ArrayXd twist_ctrl;         // 扭转角 B 样条控制点的扭转角值 theta_k [rad], 长度 N_c_twist

    // 有限截面翼型输入（长度 M_sec，例如 r/R = [0.1, 0.2, 0.5, 0.75, 1.0] 共 5 个截面）
    ArrayXd section_r_norm;     // 翼型控制截面的无量纲径向位置 r/R [-], 长度 M_sec
    std::vector<std::shared_ptr<const Airfoil3D>> section_airfoils; // 长度为 M_sec 的主截面翼型表

    // 由 validate_and_finalize() 自动预计算的离散叶素翼型表、B 样条基函数矩阵与展向分布
    std::vector<std::shared_ptr<const Airfoil3D>> airfoils; // 自动插值到 N_r 个叶素上的翼型表 (长度 N_r)
    ArrayXd chord_knots;        // 弦长均匀夹紧节点向量
    ArrayXd twist_knots;        // 扭转均匀夹紧节点向量
    ArrayXd chord_u_params;     // 各叶素对应的弦长 B 样条参数 u_i* in [0, 1], 长度 N_r
    ArrayXd twist_u_params;     // 各叶素对应的扭转 B 样条参数 u_i* in [0, 1], 长度 N_r
    MatrixXd chord_basis;       // 弦长基函数矩阵 B_c, 维度 N_r x N_c_chord
    MatrixXd twist_basis;       // 扭转基函数矩阵 B_t, 维度 N_r x N_c_twist
    ArrayXd dchord_dr_norm;     // 各叶素处弦长对无量纲半径的导数 dc / d(r/R), 长度 N_r
    ArrayXd dtwist_dr_norm;     // 各叶素处扭转对无量纲半径的导数 d(twist) / d(r/R), 长度 N_r
    ArrayXd chord;              // 各径向站位处计算出的弦长分布 c(r) [m], 长度 N_r
    ArrayXd twist;              // 各径向站位处计算出的扭转角分布 twist(r) [rad], 长度 N_r

    void validate_and_finalize() {
        if (num_blades < 1) {
            throw std::invalid_argument("RotorGeometry: num_blades B must be >= 1.");
        }
        if (spin_dir != 1 && spin_dir != -1) {
            throw std::invalid_argument("RotorGeometry: spin_dir must be +1 (CCW) or -1 (CW).");
        }
        if (radius <= hub_radius || hub_radius < 0.0) {
            throw std::invalid_argument("RotorGeometry: must satisfy R > R_hub >= 0.");
        }

        if (r.size() == 0) {
            if (n_radial < 2) {
                throw std::invalid_argument("RotorGeometry: n_radial N must be >= 2.");
            }
            r.resize(n_radial);
            dr.resize(n_radial);
            const double span = radius - hub_radius;
            for (int i = 0; i < n_radial; ++i) {
                const double e0 = hub_radius + span * (static_cast<double>(i) / static_cast<double>(n_radial));
                const double e1 = hub_radius + span * (static_cast<double>(i + 1) / static_cast<double>(n_radial));
                r[i]  = 0.5 * (e0 + e1);
                dr[i] = e1 - e0;
            }
        } else {
            n_radial = static_cast<int>(r.size());
            if (n_radial < 2) {
                throw std::invalid_argument("RotorGeometry: number of radial stations N_r must be >= 2.");
            }
        }

        const Eigen::Index nr = r.size();
        if (r[0] <= 0.0 || r[0] < hub_radius - 1e-9 || r[nr - 1] > radius + 1e-9) {
            throw std::invalid_argument("RotorGeometry: radial stations r must lie within [hub_radius, radius] and be > 0.");
        }
        for (Eigen::Index i = 0; i < nr - 1; ++i) {
            if (r[i + 1] <= r[i]) {
                throw std::invalid_argument("RotorGeometry: radial stations r must be strictly increasing.");
            }
        }

        if (dr.size() == 0) {
            dr.resize(nr);
            for (Eigen::Index i = 0; i < nr; ++i) {
                double r_left  = (i == 0) ? hub_radius : 0.5 * (r[i - 1] + r[i]);
                double r_right = (i == nr - 1) ? radius : 0.5 * (r[i] + r[i + 1]);
                dr[i] = std::max(r_right - r_left, 1e-6);
            }
        } else if (dr.size() != nr) {
            throw std::invalid_argument("RotorGeometry: dr must either be empty (auto-computed) or have length N_r.");
        } else {
            for (Eigen::Index i = 0; i < nr; ++i) {
                if (dr[i] <= 0.0) {
                    throw std::invalid_argument("RotorGeometry: all elements in dr must be strictly positive.");
                }
            }
        }

        const double x_hub = hub_radius / radius;
        if (!chord_curve.empty()) {
            chord_ctrl_r_norm = chord_curve.ctrl_r_norm();
            chord_ctrl        = chord_curve.ctrl_vals();
            bspline_degree    = chord_curve.degree();
        } else {
            const int nc_chord = static_cast<int>(chord_ctrl.size());
            if (nc_chord < bspline_degree + 1) {
                throw std::invalid_argument("RotorGeometry: must provide a valid chord_curve or chord_ctrl.");
            }
            if (chord_ctrl_r_norm.size() == 0) {
                chord_ctrl_r_norm = bspline::build_greville_abscissae(nc_chord, bspline_degree, x_hub, 1.0);
            }
            chord_curve = BSplineCurve(chord_ctrl_r_norm, chord_ctrl, bspline_degree);
        }

        if (!twist_curve.empty()) {
            twist_ctrl_r_norm = twist_curve.ctrl_r_norm();
            twist_ctrl        = twist_curve.ctrl_vals();
        } else {
            const int nc_twist = static_cast<int>(twist_ctrl.size());
            if (nc_twist < bspline_degree + 1) {
                throw std::invalid_argument("RotorGeometry: must provide a valid twist_curve or twist_ctrl.");
            }
            if (twist_ctrl_r_norm.size() == 0) {
                twist_ctrl_r_norm = bspline::build_greville_abscissae(nc_twist, bspline_degree, x_hub, 1.0);
            }
            twist_curve = BSplineCurve(twist_ctrl_r_norm, twist_ctrl, bspline_degree);
        }

        const ArrayXd r_norm_vec = r / radius;
        chord_knots = chord_curve.knots();
        twist_knots = twist_curve.knots();

        chord_basis = chord_curve.basis_matrix(r_norm_vec, &chord_u_params);
        twist_basis = twist_curve.basis_matrix(r_norm_vec, &twist_u_params);

        chord = (chord_basis * chord_ctrl.matrix()).array();
        twist = (twist_basis * twist_ctrl.matrix()).array();

        dchord_dr_norm = bspline::compute_parametric_slope_dy_dx(
            chord_u_params, chord_ctrl_r_norm, chord_ctrl, chord_knots, chord_curve.degree());
        dtwist_dr_norm = bspline::compute_parametric_slope_dy_dx(
            twist_u_params, twist_ctrl_r_norm, twist_ctrl, twist_knots, twist_curve.degree());

        if (!section_airfoils.empty()) {
            if (section_r_norm.size() == 0) {
                const int m_sec = static_cast<int>(section_airfoils.size());
                section_r_norm.resize(m_sec);
                if (m_sec == 1) {
                    section_r_norm[0] = 0.5 * (x_hub + 1.0);
                } else {
                    for (int m = 0; m < m_sec; ++m) {
                        section_r_norm[m] = x_hub + (1.0 - x_hub) * (static_cast<double>(m) / static_cast<double>(m_sec - 1));
                    }
                }
            }
            airfoils = Airfoil3D::interpolate_spanwise_sections(section_r_norm, section_airfoils, r_norm_vec);
        } else if (!airfoils.empty()) {
            if (section_r_norm.size() > 0) {
                section_airfoils = airfoils;
                airfoils = Airfoil3D::interpolate_spanwise_sections(section_r_norm, section_airfoils, r_norm_vec);
            } else if (static_cast<Eigen::Index>(airfoils.size()) != nr) {
                section_airfoils = airfoils;
                const int m_sec  = static_cast<int>(section_airfoils.size());
                section_r_norm.resize(m_sec);
                if (m_sec == 1) {
                    section_r_norm[0] = 0.5 * (x_hub + 1.0);
                } else {
                    for (int m = 0; m < m_sec; ++m) {
                        section_r_norm[m] = x_hub + (1.0 - x_hub) * (static_cast<double>(m) / static_cast<double>(m_sec - 1));
                    }
                }
                airfoils = Airfoil3D::interpolate_spanwise_sections(section_r_norm, section_airfoils, r_norm_vec);
            } else {
                section_r_norm   = r_norm_vec;
                section_airfoils = airfoils;
            }
        } else {
            throw std::invalid_argument("RotorGeometry: must provide airfoils or section_airfoils.");
        }

        for (Eigen::Index i = 0; i < nr; ++i) {
            if (!airfoils[i]) {
                throw std::invalid_argument("RotorGeometry: interpolated airfoils contains null pointer at index " + std::to_string(i));
            }
        }
    }
};

// ============================================================================
// 2. 运行工况与大气环境 OperatingCondition
// ============================================================================
struct OperatingCondition {
    double vx = 0.0;
    double vy = 0.0;
    double vz = 0.0;
    double omega = 100.0;

    double pitch_0  = 0.0;
    double pitch_1c = 0.0;
    double pitch_1s = 0.0;

    double rho = 1.225;
    double mu  = 1.7894e-5;
    double a_sound = 340.29;

    static OperatingCondition from_freestream(double v_inf,
                                              double alpha_disk_rad,
                                              double omega_rad_s,
                                              double pitch_0_rad = 0.0,
                                              double pitch_1c_rad = 0.0,
                                              double pitch_1s_rad = 0.0,
                                              double beta_rad = 0.0,
                                              double rho_val = 1.225,
                                              double mu_val = 1.7894e-5,
                                              double a_val = 340.29) {
        OperatingCondition cond;
        const double v_inplane = v_inf * std::sin(alpha_disk_rad);
        cond.vx       = v_inplane * std::cos(beta_rad);
        cond.vy       = v_inplane * std::sin(beta_rad);
        cond.vz       = v_inf * std::cos(alpha_disk_rad);
        cond.omega    = omega_rad_s;
        cond.pitch_0  = pitch_0_rad;
        cond.pitch_1c = pitch_1c_rad;
        cond.pitch_1s = pitch_1s_rad;
        cond.rho      = rho_val;
        cond.mu       = mu_val;
        cond.a_sound  = a_val;
        return cond;
    }
};

// ============================================================================
// 3. 求解器与物理修正开关配置 SolverConfig
// ============================================================================
struct SolverConfig {
    int n_azimuth = 36;
    WakeModel wake_model = WakeModel::kGlauertPittPeters;

    bool enable_tip_loss        = true;
    bool enable_vrs_correction  = true;
    bool enable_stall_delay     = true;
    bool enable_tangential_induction = true;

    int max_newton_iters   = 15;
    int max_brent_iters    = 60;
    int max_wake_iters     = 4;
    double tol             = 1e-10;
    double newton_max_step = 0.25;
};

// ============================================================================
// 4. 敏度/解析导数结构体 BEMTSensitivity
// ============================================================================
struct BEMTSensitivity {
    bool valid = false;

    // 1. 集总目标函数对 B 样条控制点纵坐标值 (chord_curve.ctrl_vals, twist_curve.ctrl_vals) 的解析梯度
    ArrayXd dT_dchord_ctrl,  dT_dtwist_ctrl;
    ArrayXd dQ_dchord_ctrl,  dQ_dtwist_ctrl;
    ArrayXd dP_dchord_ctrl,  dP_dtwist_ctrl;
    ArrayXd dFx_dchord_ctrl, dFx_dtwist_ctrl;
    ArrayXd dFy_dchord_ctrl, dFy_dtwist_ctrl;
    ArrayXd dMx_dchord_ctrl, dMx_dtwist_ctrl;
    ArrayXd dMy_dchord_ctrl, dMy_dtwist_ctrl;

    // 2. 集总目标函数对二维 B 样条控制点横坐标 (chord_curve.ctrl_r_norm, twist_curve.ctrl_r_norm) 的解析梯度
    ArrayXd dT_dchord_ctrl_r,  dT_dtwist_ctrl_r;
    ArrayXd dQ_dchord_ctrl_r,  dQ_dtwist_ctrl_r;
    ArrayXd dP_dchord_ctrl_r,  dP_dtwist_ctrl_r;
    ArrayXd dFx_dchord_ctrl_r, dFx_dtwist_ctrl_r;
    ArrayXd dFy_dchord_ctrl_r, dFy_dtwist_ctrl_r;
    ArrayXd dMx_dchord_ctrl_r, dMx_dtwist_ctrl_r;
    ArrayXd dMy_dchord_ctrl_r, dMy_dtwist_ctrl_r;

    // 3. 集总目标函数对各径向站位局部采样值 (chord, twist) 的梯度向量 (长度 N_r)
    ArrayXd dT_dchord,  dT_dtwist;
    ArrayXd dQ_dchord,  dQ_dtwist;
    ArrayXd dP_dchord,  dP_dtwist;
    ArrayXd dFx_dchord, dFx_dtwist;
    ArrayXd dFy_dchord, dFy_dtwist;
    ArrayXd dMx_dchord, dMx_dtwist;
    ArrayXd dMy_dchord, dMy_dtwist;

    // 4. 集总目标函数对各叶素微元二维翼型升/阻力系数 (Cl_2D, Cd_2D) 的解析梯度矩阵 (维度 N_r x N_psi)
    //    专用于通过链式法则耦合 Kulfan (CST) 翼型参数化与 NeuralFoil 的气动导数 d(CL, CD)/d(CST)
    ArrayXXd dT_dcl,  dT_dcd;
    ArrayXXd dQ_dcl,  dQ_dcd;
    ArrayXXd dP_dcl,  dP_dcd;
    ArrayXXd dFx_dcl, dFx_dcd;
    ArrayXXd dFy_dcl, dFy_dcd;
    ArrayXXd dMx_dcl, dMx_dcd;
    ArrayXXd dMy_dcl, dMy_dcd;

    // 5. 集总目标函数 [T, Q, P, Fx, Fy, Mx, My] 对全局工况参数 [omega, pitch_0, vx, vy, vz] 的雅可比矩阵 (7 x 5)
    MatrixXd dForces_dCond;

    // 6. 径向平均拉力分布 (dT/dr, 长度 N_r) 对各径向站位弦长和扭转角的对角雅可比向量 (长度 N_r)
    ArrayXd dTr_dchord_diag;
    ArrayXd dTr_dtwist_diag;
};

// ============================================================================
// 5. BEMT 完整求解结果结构体 BEMTResult
// ============================================================================
struct BEMTResult {
    bool converged = true;
    int total_fallback_cells = 0;

    double thrust = 0.0;
    double torque = 0.0;
    double power  = 0.0;

    double fx = 0.0;
    double fy = 0.0;
    double fz = 0.0;

    double mx = 0.0;
    double my = 0.0;
    double mz = 0.0;

    double ct = 0.0;
    double cq = 0.0;
    double cp = 0.0;
    double fm = 0.0;

    double vi_disk_mean = 0.0;
    double wake_skew_chi = 0.0;
    double pitt_kx = 0.0;
    double pitt_ky = 0.0;

    ArrayXd psi_grid;
    ArrayXd vi_annulus;
    ArrayXd dT_dr_mean;
    ArrayXd dQ_dr_mean;

    ArrayXXd phi;
    ArrayXXd alpha;
    ArrayXXd vi;
    ArrayXXd vt;
    ArrayXXd cl;
    ArrayXXd cd;
    ArrayXXd re;
    ArrayXXd ma;
    ArrayXXd f_tip;
    ArrayXXd df_z;
    ArrayXXd df_theta;

    BEMTSensitivity sensitivity;
};

} // namespace bemt
