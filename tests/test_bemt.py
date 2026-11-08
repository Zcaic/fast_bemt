import os
import sys
import time
import numpy as np

# 将 python 目录加入路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "python")))
import fast_bemt as fb


def build_20_station_rotor() -> fb.RotorGeometry:
    """构建包含 20 个径向站位及 20 个独立三维翼型表 (alpha, Re, Ma) 的测试旋翼"""
    n_r = 20
    radius = 1.25      # 旋翼半径 1.25 m
    hub_radius = 0.15  # 桨毂半径 0.15 m

    # 20 个径向控制体中心坐标与微元宽度 dr
    edges = np.linspace(hub_radius, radius, n_r + 1)
    r = 0.5 * (edges[:-1] + edges[1:])
    dr = np.diff(edges)

    # 弦长与非线性扭转角分布
    r_norm = r / radius
    chord = 0.10 * (1.0 - 0.35 * r_norm**2)
    twist = np.deg2rad(28.0 * (1.0 - r_norm) + 4.0)

    # 三维翼型网格: N_alpha = 73 (-18 deg ~ +18 deg), N_Re = 4, N_Ma = 4
    alpha_grid = np.deg2rad(np.linspace(-18.0, 18.0, 73))
    re_grid = np.array([1.0e5, 5.0e5, 1.5e6, 5.0e6])
    ma_grid = np.array([0.0, 0.3, 0.5, 0.7])

    A, RE, MA = np.meshgrid(alpha_grid, re_grid, ma_grid, indexing="ij")

    # 为 20 个径向站位生成各自独立的 3D 翼型极曲线表 (模拟从根部到尖部弯度与阻力系数变化)
    airfoils = []
    for i in range(n_r):
        span_frac = i / (n_r - 1)
        alpha_0 = np.deg2rad(-2.0 * (1.0 - 0.5 * span_frac))  # 零升迎角随展向变化
        pg_factor = 1.0 / np.sqrt(1.0 - (0.8 * MA) ** 2)       # Prandtl-Glauert 压缩性放大
        re_scale = (RE / 1.0e6) ** (-0.08)                     # 雷诺数对摩擦阻力的影响

        # 构造带失速软化的光滑 Cl 和 Cd 三维张量
        a_eff = A - alpha_0
        cl_linear = 2.0 * np.pi * a_eff * pg_factor
        stall_blend = 1.0 / (1.0 + (a_eff / np.deg2rad(14.0)) ** 6)
        cl_3d = cl_linear * stall_blend + 1.05 * np.sin(2.0 * a_eff) * (1.0 - stall_blend)

        cd_min = (0.0085 - 0.002 * span_frac) * re_scale
        cd_3d = cd_min + 0.025 * (a_eff / np.deg2rad(10.0)) ** 2 + 0.05 * (MA**3)

        af = fb.create_3d_airfoil(alpha_grid, re_grid, ma_grid, cl_3d, cd_3d, degrees=False)
        airfoils.append(af)

    geom = fb.RotorGeometry(
        num_blades=3,
        radius=radius,
        hub_radius=hub_radius,
        r=r,
        chord=chord,
        twist=twist,
        airfoils=airfoils,
        spin_dir=1,
        dr=dr,
    )
    return geom


def test_oblique_solver_and_gradients():
    print("=" * 76)
    print("1. 构建 20 径向站位 + 20 独立 3D (alpha, Re, Ma) 翼型表旋翼模型...")
    geom = build_20_station_rotor()

    cfg = fb.SolverConfig(
        n_azimuth=36,
        wake_model=fb.WakeModel.GlauertPittPeters,
        enable_tip_loss=True,
        enable_vrs_correction=True,
        enable_stall_delay=True,
        enable_tangential_induction=True,
    )
    solver = fb.BEMTSolver(geom, cfg)

    # 设定斜入流倾转过渡工况: V_inf = 25 m/s, 桨盘倾角 45 deg, 侧滑角 10 deg, 转速 180 rad/s
    cond = fb.OperatingCondition.from_freestream(
        v_inf=25.0,
        alpha_disk_rad=np.deg2rad(45.0),
        omega_rad_s=180.0,
        pitch_0_rad=np.deg2rad(6.0),
        beta_rad=np.deg2rad(10.0),
    )

    # 预热并测试求解 + 求导速度
    _ = solver.solve(cond, compute_gradients=True)
    n_runs = 200
    t0 = time.perf_counter()
    for _ in range(n_runs):
        res_fast = solver.solve(cond, compute_gradients=False)
    t_solve_us = (time.perf_counter() - t0) / n_runs * 1e6

    t1 = time.perf_counter()
    for _ in range(n_runs):
        res = solver.solve(cond, compute_gradients=True)
    t_grad_us = (time.perf_counter() - t1) / n_runs * 1e6

    print(f"   [求解成功] Converged: {res.converged}, Brent兜底微元数: {res.total_fallback_cells}")
    print(f"   [计算耗时] 纯求解 (20x36=720微元): {t_solve_us:.1f} us | 求解+全解析雅可比梯度: {t_grad_us:.1f} us")
    print(f"   [气动性能] T = {res.thrust:.2f} N, Q = {res.torque:.2f} N·m, P = {res.power/1e3:.2f} kW")
    print(f"   [三轴力矩] Mx(Roll) = {res.mx:.2f} N·m, My(Pitch) = {res.my:.2f} N·m, Fx = {res.fx:.2f} N")
    print(f"   [尾迹偏斜] 平均诱导速度 vi = {res.vi_disk_mean:.2f} m/s, 尾迹偏斜角 chi = {np.rad2deg(res.wake_skew_chi):.2f} deg, kx = {res.pitt_kx:.4f}")

    # ========================================================================
    # 2. 严格对比隐函数定理 (IFT) 解析梯度 vs 中心有限差分 (Central FD)
    # ========================================================================
    print("\n" + "=" * 76)
    print("2. 校验解析梯度 (IFT-AD) 与中心有限差分 (FD) 精度...")
    eps = 1e-6
    n_r = geom.r.size

    fd_dT_dchord = np.zeros(n_r)
    fd_dQ_dchord = np.zeros(n_r)
    fd_dMy_dchord = np.zeros(n_r)
    fd_dT_dtwist = np.zeros(n_r)
    fd_dQ_dtwist = np.zeros(n_r)
    fd_dMx_dtwist = np.zeros(n_r)

    for i in range(n_r):
        # 扰动第 i 个站位的弦长 chord[i]
        c_plus = geom.chord.copy()
        c_minus = geom.chord.copy()
        c_plus[i] += eps
        c_minus[i] -= eps

        g_p = fb.RotorGeometry(geom.num_blades, geom.radius, geom.hub_radius, geom.r, c_plus, geom.twist, geom.airfoils, geom.spin_dir, geom.dr)
        g_m = fb.RotorGeometry(geom.num_blades, geom.radius, geom.hub_radius, geom.r, c_minus, geom.twist, geom.airfoils, geom.spin_dir, geom.dr)
        r_p = fb.BEMTSolver(g_p, cfg).solve(cond, False)
        r_m = fb.BEMTSolver(g_m, cfg).solve(cond, False)

        fd_dT_dchord[i] = (r_p.thrust - r_m.thrust) / (2.0 * eps)
        fd_dQ_dchord[i] = (r_p.torque - r_m.torque) / (2.0 * eps)
        fd_dMy_dchord[i] = (r_p.my - r_m.my) / (2.0 * eps)

        # 扰动第 i 个站位的扭转角 twist[i]
        tw_plus = geom.twist.copy()
        tw_minus = geom.twist.copy()
        tw_plus[i] += eps
        tw_minus[i] -= eps

        g_p = fb.RotorGeometry(geom.num_blades, geom.radius, geom.hub_radius, geom.r, geom.chord, tw_plus, geom.airfoils, geom.spin_dir, geom.dr)
        g_m = fb.RotorGeometry(geom.num_blades, geom.radius, geom.hub_radius, geom.r, geom.chord, tw_minus, geom.airfoils, geom.spin_dir, geom.dr)
        r_p = fb.BEMTSolver(g_p, cfg).solve(cond, False)
        r_m = fb.BEMTSolver(g_m, cfg).solve(cond, False)

        fd_dT_dtwist[i] = (r_p.thrust - r_m.thrust) / (2.0 * eps)
        fd_dQ_dtwist[i] = (r_p.torque - r_m.torque) / (2.0 * eps)
        fd_dMx_dtwist[i] = (r_p.mx - r_m.mx) / (2.0 * eps)

    def rel_err(ad_val, fd_val):
        return np.linalg.norm(ad_val - fd_val) / max(np.linalg.norm(fd_val), 1e-12)

    sens = res.sensitivity
    print(f"   - dT / d(chord)   相对 L2 误差: {rel_err(sens.dT_dchord, fd_dT_dchord):.3e}")
    print(f"   - dQ / d(chord)   相对 L2 误差: {rel_err(sens.dQ_dchord, fd_dQ_dchord):.3e}")
    print(f"   - dMy / d(chord)  相对 L2 误差: {rel_err(sens.dMy_dchord, fd_dMy_dchord):.3e}")
    print(f"   - dT / d(twist)   相对 L2 误差: {rel_err(sens.dT_dtwist, fd_dT_dtwist):.3e}")
    print(f"   - dQ / d(twist)   相对 L2 误差: {rel_err(sens.dQ_dtwist, fd_dQ_dtwist):.3e}")
    print(f"   - dMx / d(twist)  相对 L2 误差: {rel_err(sens.dMx_dtwist, fd_dMx_dtwist):.3e}")

    # 验证全局工况参数雅可比矩阵 dForces_dCond (7 x 5)
    cond_attrs = ["omega", "pitch_0", "vx", "vy", "vz"]
    fd_cond_jac = np.zeros((7, 5))
    for c_idx, attr in enumerate(cond_attrs):
        c_p = fb.OperatingCondition(cond.vx, cond.vy, cond.vz, cond.omega, cond.pitch_0, cond.pitch_1c, cond.pitch_1s, cond.rho, cond.mu, cond.a_sound)
        c_m = fb.OperatingCondition(cond.vx, cond.vy, cond.vz, cond.omega, cond.pitch_0, cond.pitch_1c, cond.pitch_1s, cond.rho, cond.mu, cond.a_sound)
        setattr(c_p, attr, getattr(cond, attr) + eps)
        setattr(c_m, attr, getattr(cond, attr) - eps)
        rp = solver.solve(c_p, False)
        rm = solver.solve(c_m, False)
        vec_p = np.array([rp.thrust, rp.torque, rp.power, rp.fx, rp.fy, rp.mx, rp.my])
        vec_m = np.array([rm.thrust, rm.torque, rm.power, rm.fx, rm.fy, rm.mx, rm.my])
        fd_cond_jac[:, c_idx] = (vec_p - vec_m) / (2.0 * eps)

    print(f"   - dForces / dCond (7x5 矩阵) 相对 L2 误差: {rel_err(sens.dForces_dCond, fd_cond_jac):.3e}")
    print("=" * 76)


if __name__ == "__main__":
    test_oblique_solver_and_gradients()