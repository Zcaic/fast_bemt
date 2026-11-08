#pragma once

#include "bemt/airfoil.hpp"
#include "bemt/types.hpp"
#include <memory>
#include <stdexcept>
#include <vector>

namespace bemt {

// ============================================================================
// 1. 旋翼/螺旋桨几何参数结构体 RotorGeometry
// 严格匹配你的需求：给定 N_r 个径向叶素站位的 r, chord, twist 以及对应的 N_r 个 Airfoil3D 表
// ============================================================================
struct RotorGeometry {
    int num_blades  = 2;     // 桨叶数 B
    double radius   = 1.0;   // 旋翼叶尖半径 R [m]
    double hub_radius = 0.1; // 桨毂半径 R_hub [m]
    int spin_dir    = 1;     // 旋向: +1 逆时针(从上往下看), -1 顺时针

    ArrayXd r;               // 径向站位坐标 [m], 长度 N_r
    ArrayXd dr;              // 径向积分微元宽度 [m], 长度 N_r (若为空则自动生成梯形/控制体权重)
    ArrayXd chord;           // 弦长分布 [m], 长度 N_r
    ArrayXd twist;           // 几何扭转角分布 [rad], 长度 N_r
    std::vector<std::shared_ptr<const Airfoil3D>> airfoils; // 长度为 N_r 的独立翼型表指针

    // 校验几何合法性并自动补全径向积分权重 dr
    void validate_and_finalize() {
        const Eigen::Index nr = r.size();
        if (nr < 2) {
            throw std::invalid_argument("RotorGeometry: number of radial stations N_r must be >= 2.");
        }
        if (chord.size() != nr || twist.size() != nr) {
            throw std::invalid_argument("RotorGeometry: r, chord, and twist must have the same length N_r.");
        }
        if (static_cast<Eigen::Index>(airfoils.size()) != nr) {
            throw std::invalid_argument("RotorGeometry: airfoils list size must equal N_r (1-to-1 mapping).");
        }
        for (Eigen::Index i = 0; i < nr; ++i) {
            if (!airfoils[i]) {
                throw std::invalid_argument("RotorGeometry: airfoils contains a null pointer at index " + std::to_string(i));
            }
        }
        if (radius <= hub_radius || hub_radius < 0.0) {
            throw std::invalid_argument("RotorGeometry: must satisfy R > R_hub >= 0.");
        }

        // 若用户未手动指定 dr，自动根据节点坐标生成保边界守恒的微元积分宽度
        if (dr.size() != nr) {
            dr.resize(nr);
            for (Eigen::Index i = 0; i < nr; ++i) {
                double r_left  = (i == 0) ? hub_radius : 0.5 * (r[i - 1] + r[i]);
                double r_right = (i == nr - 1) ? radius : 0.5 * (r[i] + r[i + 1]);
                dr[i] = std::max(r_right - r_left, 1e-6);
            }
        }
    }
};

// ============================================================================
// 2. 运行工况与大气环境 OperatingCondition
// 坐标系约定 (Rotor Disk Frame):
// - 原点位于桨毂中心
// - Z 轴沿旋转轴指向推力正方向（向上/向前），Vz > 0 为轴向爬升/正迎面来流，Vz < 0 为下降(进入涡环区)
// - X 轴指向后行边方向 (psi = 0, 下游尾梁方向)，Y 轴指向前行边方向 (psi = 90 deg)
// ============================================================================
struct OperatingCondition {
    double vx = 0.0;          // 桨盘平面内 X 向来流速度 [m/s] (顺风向/后向为正)
    double vy = 0.0;          // 桨盘平面内 Y 向来流速度 [m/s] (侧滑向为正)
    double vz = 0.0;          // 垂直桨盘轴向来流速度 [m/s] (逆推力方向爬升/轴流来流为正)
    double omega = 100.0;     // 旋转角速度 [rad/s] (> 0)

    double pitch_0  = 0.0;    // 总距角 theta_0 [rad]
    double pitch_1c = 0.0;    // 横向周期变距 theta_1c (cos psi 项) [rad]
    double pitch_1s = 0.0;    // 纵向周期变距 theta_1s (sin psi 项) [rad]

    double rho = 1.225;       // 空气密度 [kg/m^3]
    double mu  = 1.7894e-5;   // 空气动力粘度 [Pa·s]
    double a_sound = 340.29;  // 声速 [m/s]

    // 辅助构造函数：由远前方合速度 V_inf [m/s]、桨盘倾角 alpha_disk [rad] (来流与旋转轴夹角: 0为纯轴流, 90度为纯边飞)
    // 及侧滑角 beta [rad] 生成工况
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
    int n_azimuth = 36;                    // 周向方位角网格数 N_psi (纯轴流时可设为 1，斜入流推荐 36 ~ 72)
    WakeModel wake_model = WakeModel::kGlauertPittPeters; // 默认采用方案 A (圆环平衡 + Pitt-Peters 斜尾迹修正)

    // 三大物理修正模型独立开关
    bool enable_tip_loss        = true;    // 开关 1: Prandtl 叶尖与叶根损失修正
    bool enable_vrs_correction  = true;    // 开关 2: Buhl / Glauert 高诱导速度区(涡环/风车态)经验修正
    bool enable_stall_delay     = true;    // 开关 3: Du-Selig 三维旋转失速延迟修正
    bool enable_tangential_induction = true; // 是否包含切向诱导速度 vt (尾流旋转效应)

    // 数值迭代控制参数
    int max_newton_iters   = 15;           // 内层向量化阻尼牛顿法最大迭代步数
    int max_brent_iters    = 60;           // 标量 Brent 兜底算法最大迭代步数
    int max_wake_iters     = 4;            // 外层 Pitt-Peters 尾迹偏斜角 chi 自洽迭代步数
    double tol             = 1e-10;        // 残差收敛容差
    double newton_max_step = 0.25;         // 牛顿法单步最大角度更新限幅 [rad]
};

// ============================================================================
// 4. 敏度/解析导数结构体 BEMTSensitivity (由 Step 4 隐函数求导模块填充)
// ============================================================================
struct BEMTSensitivity {
    bool valid = false;

    // 集总目标函数 (T, Q, P, Fx, Fy, Mx, My) 对径向分布设计变量 (chord, twist) 的梯度向量 (长度 N_r)
    ArrayXd dT_dchord, dT_dtwist;
    ArrayXd dQ_dchord, dQ_dtwist;
    ArrayXd dP_dchord, dP_dtwist;
    ArrayXd dMx_dchord, dMx_dtwist;
    ArrayXd dMy_dchord, dMy_dtwist;

    // 集总目标函数 [T, Q, P, Fx, Fy, Mx, My] (7维) 对全局工况/操纵参数 [omega, pitch_0, vx, vy, vz] (5维) 的雅可比矩阵 (7 x 5)
    MatrixXd dForces_dCond; // 行: 0:T(Fz), 1:Q(Mz), 2:P, 3:Fx, 4:Fy, 5:Mx, 6:My; 列: 0:omega, 1:pitch_0, 2:vx, 3:vy, 4:vz

    // 径向平均拉力分布 (dT/dr, 长度 N_r) 对弦长和扭转角的对角雅可比向量 (长度 N_r)
    ArrayXd dTr_dchord_diag;
    ArrayXd dTr_dtwist_diag;
};

// ============================================================================
// 5. BEMT 完整求解结果结构体 BEMTResult
// ============================================================================
struct BEMTResult {
    bool converged = true;
    int total_fallback_cells = 0; // 触发 Brent 兜底的微元数量统计

    // 全局集总力与力矩 [SI: N, N·m, W]
    double thrust = 0.0; // 轴向总拉力 T (= Fz) [N]
    double torque = 0.0; // 旋转轴反扭矩 Q [N·m]
    double power  = 0.0; // 消耗轴功率 P = Q * omega [W]

    double fx = 0.0;     // 桨盘面内后向力 Fx (H-force) [N]
    double fy = 0.0;     // 桨盘面内侧向力 Fy (Y-force) [N]
    double fz = 0.0;     // 桨盘轴向力 Fz (= T) [N]

    double mx = 0.0;     // 桨盘滚转力矩 Mx (Roll moment) [N·m]
    double my = 0.0;     // 桨盘俯仰力矩 My (Pitch moment) [N·m]
    double mz = 0.0;     // 桨盘偏航/扭矩力矩 Mz (= Q) [N·m]

    // 无量纲气动系数 (采用 NASA 旋翼惯例: 基于 rho, A = pi*R^2, V_tip = omega*R)
    double ct = 0.0;     // 拉力系数 C_T = T / (rho * A * V_tip^2)
    double cq = 0.0;     // 扭矩系数 C_Q = Q / (rho * A * V_tip^2 * R)
    double cp = 0.0;     // 功率系数 C_P = P / (rho * A * V_tip^3)
    double fm = 0.0;     // 悬停品质因数 Figure of Merit = C_T^(3/2) / (sqrt(2) * C_P)

    // 全局尾迹偏斜参数 (方案 A 输出)
    double vi_disk_mean = 0.0; // 桨盘面积加权平均诱导速度 [m/s]
    double wake_skew_chi = 0.0; // 尾迹偏斜角 chi [rad]
    double pitt_kx = 0.0;      // 纵向一阶谐波入流梯度系数 kx
    double pitt_ky = 0.0;      // 横向一阶谐波入流梯度系数 ky

    // 一维网格与周向平均径向分布 (长度 N_r 或 N_psi)
    ArrayXd psi_grid;    // 方位角网格 [rad], 长度 N_psi
    ArrayXd vi_annulus;  // 各径向环基准诱导速度 v_i0(r) [m/s], 长度 N_r
    ArrayXd dT_dr_mean;  // 周向平均径向拉力分布 dT/dr [N/m], 长度 N_r
    ArrayXd dQ_dr_mean;  // 周向平均径向扭矩分布 dQ/dr [N·m/m], 长度 N_r

    // 二维全桨盘分布场 (Row-Major 矩阵: 维度 N_r x N_psi)
    ArrayXXd phi;        // 局部入流角分布 [rad]
    ArrayXXd alpha;      // 局部有效迎角分布 [rad]
    ArrayXXd vi;         // 局部轴向诱导速度分布 [m/s]
    ArrayXXd vt;         // 局部切向诱导速度分布 [m/s]
    ArrayXXd cl;         // 局部升力系数分布(含3D修正) [-]
    ArrayXXd cd;         // 局部阻力系数分布(含3D修正) [-]
    ArrayXXd re;         // 局部雷诺数分布 [-]
    ArrayXXd ma;         // 局部马赫数分布 [-]
    ArrayXXd f_tip;      // 局部 Prandtl 损失系数 F [-]
    ArrayXXd df_z;       // 单片桨叶在 (r_i, psi_j) 处的单位展长轴向力 dFz/dr [N/m]
    ArrayXXd df_theta;   // 单片桨叶在 (r_i, psi_j) 处的单位展长周向阻力 dF_theta/dr [N/m]

    // 解析梯度输出
    BEMTSensitivity sensitivity;
};

} // namespace bemt