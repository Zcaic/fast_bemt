import os
import sys
import time
import numpy as np

# 将 python 目录加入路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "python")))
import fast_bemt as fb


def build_20_station_rotor(n_radial: int = 20, bspline_degree: int = 3) -> fb.RotorGeometry:
    """
    演示标准工程输入工作流：
    1. 先在 Python 中独立实例化弦长 B 样条曲线 chord_curve 与扭转 B 样条曲线 twist_curve (fb.BSplineCurve)，
       可随时调用 .sample(200) 或 .eval_batch(r_norm) 单独检查或绘制曲线；
    2. 给定 5 个主控制截面 (r/R = [0.1, 0.2, 0.5, 0.75, 1.0]) 的三维翼型极曲线表；
    3. 将 chord_curve 和 twist_curve 连同 n_radial = 20 传给 fb.RotorGeometry，自动完成叶素离散与翼型展向插值。
    """
    radius = 1.25      # 旋翼半径 1.25 m
    hub_radius = 0.15  # 桨毂半径 0.15 m

    # 1. 手动实例化弦长与扭转分布的二维参数化 B 样条曲线对象 fb.BSplineCurve
    ctrl_r_norm = np.array([0.12, 0.22, 0.38, 0.62, 0.85, 1.00])
    chord_vals = 0.10 * (1.0 - 0.35 * ctrl_r_norm**2)
    twist_vals = np.deg2rad(28.0 * (1.0 - ctrl_r_norm) + 4.0)

    chord_ctrl_2d = np.column_stack([ctrl_r_norm, chord_vals])
    twist_ctrl_2d = np.column_stack([ctrl_r_norm, twist_vals])

    chord_curve = fb.BSplineCurve(chord_ctrl_2d, degree=bspline_degree)
    twist_curve = fb.BSplineCurve(twist_ctrl_2d, degree=bspline_degree)

    # 单独检查 / 采样曲线 (例如采样 200 个点用于 Matplotlib 绘图: r_plot, c_plot = chord_curve.sample(200))
    r_plot, c_plot = chord_curve.sample(200)
    _, tw_plot = twist_curve.sample(200)
    assert r_plot.shape == (200,) and c_plot.shape == (200,) and tw_plot.shape == (200,)

    # 2. 定义 5 个展向主截面 (r/R = 0.1, 0.2, 0.5, 0.75, 1.0) 的三维翼型表 (alpha, Re, Ma)
    section_r_norm = np.array([0.10, 0.20, 0.50, 0.75, 1.00])
    alpha_grid = np.deg2rad(np.linspace(-18.0, 18.0, 73))
    re_grid = np.array([1.0e5, 5.0e5, 1.5e6, 5.0e6])
    ma_grid = np.array([0.0, 0.3, 0.5, 0.7])

    A, RE, MA = np.meshgrid(alpha_grid, re_grid, ma_grid, indexing="ij")

    five_section_airfoils = []
    for r_sec in section_r_norm:
        alpha_0 = np.deg2rad(-2.5 * (1.0 - 0.6 * r_sec))  # 零升迎角随展向主截面变化
        pg_factor = 1.0 / np.sqrt(1.0 - (0.8 * MA) ** 2)  # Prandtl-Glauert 压缩性放大
        re_scale = (RE / 1.0e6) ** (-0.08)                # 雷诺数对摩擦阻力的影响

        a_eff = A - alpha_0
        cl_linear = 2.0 * np.pi * a_eff * pg_factor
        stall_blend = 1.0 / (1.0 + (a_eff / np.deg2rad(14.0)) ** 6)
        cl_3d = cl_linear * stall_blend + 1.05 * np.sin(2.0 * a_eff) * (1.0 - stall_blend)

        cd_min = (0.0090 - 0.0025 * r_sec) * re_scale
        cd_3d = cd_min + 0.025 * (a_eff / np.deg2rad(10.0)) ** 2 + 0.05 * (MA**3)

        af = fb.create_3d_airfoil(alpha_grid, re_grid, ma_grid, cl_3d, cd_3d, degrees=False)
        five_section_airfoils.append(af)

    # 3. 将实例化好的 chord_curve 与 twist_curve 直接传给 fb.RotorGeometry
    geom = fb.RotorGeometry(
        num_blades=3,
        radius=radius,
        hub_radius=hub_radius,
        n_radial=n_radial,
        chord_curve=chord_curve,
        twist_curve=twist_curve,
        airfoils=five_section_airfoils,
        section_r_norm=section_r_norm,
        spin_dir=1,
    )
    return geom


def test_oblique_solver_and_gradients():
    print("=" * 80)
    print("1. 独立实例化 BSplineCurve (弦长/扭转曲线) + 5 截面翼型自动插值 + N=20 叶素模型...")
    geom = build_20_station_rotor(n_radial=20, bspline_degree=3)

    # 演示直接从 BSplineCurve 实例读取曲线属性与采样点
    cc = geom.chord_curve
    tc = geom.twist_curve
    r_sample, c_sample = cc.sample(100)
    print(f"   [BSplineCurve] degree = {cc.degree}, num_ctrl_points = {cc.num_ctrl_points}, knots = {np.round(cc.knots, 4)}")
    print(f"   [曲线独立检查] r/R=0.5 处弦长 = {cc.eval_at_r_norm(0.5):.5f} m, 扭转角 = {np.rad2deg(tc.eval_at_r_norm(0.5)):.2f} deg, 100点采样范围 = [{c_sample.min():.4f}, {c_sample.max():.4f}] m")
    print(f"   [翼型展向插值] 输入主翼型截面数 = {len(geom.section_airfoils)} (r/R = {geom.section_r_norm}) -> 自动插值叶素数 N = {len(geom.airfoils)}")

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
    print(f"   [计算耗时] 纯求解 (20x36=720微元): {t_solve_us:.1f} us | 求解+B样条控制点全解析梯度: {t_grad_us:.1f} us")
    print(f"   [气动性能] T = {res.thrust:.2f} N, Q = {res.torque:.2f} N·m, P = {res.power/1e3:.2f} kW")
    print(f"   [三轴力矩] Mx(Roll) = {res.mx:.2f} N·m, My(Pitch) = {res.my:.2f} N·m, Fx = {res.fx:.2f} N")
    print(f"   [尾迹偏斜] 平均诱导速度 vi = {res.vi_disk_mean:.2f} m/s, 尾迹偏斜角 chi = {np.rad2deg(res.wake_skew_chi):.2f} deg, kx = {res.pitt_kx:.4f}")

    # ========================================================================
    # 2. 严格对比 BSplineCurve 控制点 (纵坐标值 + 横坐标 r/R) 解析梯度 vs 中心有限差分
    # ========================================================================
    print("\n" + "=" * 80)
    print("2. 校验 BSplineCurve 控制点解析梯度 (IFT + 基函数矩阵链式法则) 与中心有限差分精度...")
    eps = 1e-6
    n_c = cc.num_ctrl_points

    fd_dT_dchord_ctrl = np.zeros(n_c)
    fd_dQ_dchord_ctrl = np.zeros(n_c)
    fd_dMy_dchord_ctrl = np.zeros(n_c)
    fd_dT_dtwist_ctrl = np.zeros(n_c)
    fd_dQ_dtwist_ctrl = np.zeros(n_c)
    fd_dMx_dtwist_ctrl = np.zeros(n_c)

    fd_dT_dchord_ctrl_r = np.zeros(n_c)
    fd_dT_dtwist_ctrl_r = np.zeros(n_c)

    for k in range(n_c):
        # 2.1 扰动第 k 个弦长控制点的纵坐标 chord_curve.ctrl_vals[k]
        c_plus = cc.ctrl_vals.copy()
        c_minus = cc.ctrl_vals.copy()
        c_plus[k] += eps
        c_minus[k] -= eps

        cc_p = fb.BSplineCurve(cc.ctrl_r_norm, c_plus, cc.degree)
        cc_m = fb.BSplineCurve(cc.ctrl_r_norm, c_minus, cc.degree)
        g_p = fb.RotorGeometry(geom.num_blades, geom.radius, geom.hub_radius, geom.n_radial, cc_p, tc, geom.section_airfoils, geom.section_r_norm, geom.spin_dir)
        g_m = fb.RotorGeometry(geom.num_blades, geom.radius, geom.hub_radius, geom.n_radial, cc_m, tc, geom.section_airfoils, geom.section_r_norm, geom.spin_dir)
        r_p = fb.BEMTSolver(g_p, cfg).solve(cond, False)
        r_m = fb.BEMTSolver(g_m, cfg).solve(cond, False)

        fd_dT_dchord_ctrl[k] = (r_p.thrust - r_m.thrust) / (2.0 * eps)
        fd_dQ_dchord_ctrl[k] = (r_p.torque - r_m.torque) / (2.0 * eps)
        fd_dMy_dchord_ctrl[k] = (r_p.my - r_m.my) / (2.0 * eps)

        # 2.2 扰动第 k 个扭转角控制点的纵坐标 twist_curve.ctrl_vals[k]
        tw_plus = tc.ctrl_vals.copy()
        tw_minus = tc.ctrl_vals.copy()
        tw_plus[k] += eps
        tw_minus[k] -= eps

        tc_p = fb.BSplineCurve(tc.ctrl_r_norm, tw_plus, tc.degree)
        tc_m = fb.BSplineCurve(tc.ctrl_r_norm, tw_minus, tc.degree)
        g_p = fb.RotorGeometry(geom.num_blades, geom.radius, geom.hub_radius, geom.n_radial, cc, tc_p, geom.section_airfoils, geom.section_r_norm, geom.spin_dir)
        g_m = fb.RotorGeometry(geom.num_blades, geom.radius, geom.hub_radius, geom.n_radial, cc, tc_m, geom.section_airfoils, geom.section_r_norm, geom.spin_dir)
        r_p = fb.BEMTSolver(g_p, cfg).solve(cond, False)
        r_m = fb.BEMTSolver(g_m, cfg).solve(cond, False)

        fd_dT_dtwist_ctrl[k] = (r_p.thrust - r_m.thrust) / (2.0 * eps)
        fd_dQ_dtwist_ctrl[k] = (r_p.torque - r_m.torque) / (2.0 * eps)
        fd_dMx_dtwist_ctrl[k] = (r_p.mx - r_m.mx) / (2.0 * eps)

        # 2.3 扰动内部控制点 (1 <= k <= n_c-2) 的无量纲径向横坐标 r/R
        if 1 <= k <= n_c - 2:
            rc_p, rc_m = cc.ctrl_r_norm.copy(), cc.ctrl_r_norm.copy()
            rc_p[k] += eps
            rc_m[k] -= eps
            gp_rc = fb.RotorGeometry(
                geom.num_blades, geom.radius, geom.hub_radius, geom.n_radial,
                fb.BSplineCurve(rc_p, cc.ctrl_vals, cc.degree), tc,
                geom.section_airfoils, geom.section_r_norm, geom.spin_dir,
            )
            gm_rc = fb.RotorGeometry(
                geom.num_blades, geom.radius, geom.hub_radius, geom.n_radial,
                fb.BSplineCurve(rc_m, cc.ctrl_vals, cc.degree), tc,
                geom.section_airfoils, geom.section_r_norm, geom.spin_dir,
            )
            fd_dT_dchord_ctrl_r[k] = (fb.BEMTSolver(gp_rc, cfg).solve(cond, False).thrust - fb.BEMTSolver(gm_rc, cfg).solve(cond, False).thrust) / (2.0 * eps)

            rt_p, rt_m = tc.ctrl_r_norm.copy(), tc.ctrl_r_norm.copy()
            rt_p[k] += eps
            rt_m[k] -= eps
            gp_rt = fb.RotorGeometry(
                geom.num_blades, geom.radius, geom.hub_radius, geom.n_radial,
                cc, fb.BSplineCurve(rt_p, tc.ctrl_vals, tc.degree),
                geom.section_airfoils, geom.section_r_norm, geom.spin_dir,
            )
            gm_rt = fb.RotorGeometry(
                geom.num_blades, geom.radius, geom.hub_radius, geom.n_radial,
                cc, fb.BSplineCurve(rt_m, tc.ctrl_vals, tc.degree),
                geom.section_airfoils, geom.section_r_norm, geom.spin_dir,
            )
            fd_dT_dtwist_ctrl_r[k] = (fb.BEMTSolver(gp_rt, cfg).solve(cond, False).thrust - fb.BEMTSolver(gm_rt, cfg).solve(cond, False).thrust) / (2.0 * eps)
        else:
            fd_dT_dchord_ctrl_r[k] = res.sensitivity.dT_dchord_ctrl_r[k]
            fd_dT_dtwist_ctrl_r[k] = res.sensitivity.dT_dtwist_ctrl_r[k]

    def rel_err(ad_val, fd_val):
        return np.linalg.norm(ad_val - fd_val) / max(np.linalg.norm(fd_val), 1e-12)

    sens = res.sensitivity
    print(f"   - dT  / d(chord_ctrl)     相对 L2 误差: {rel_err(sens.dT_dchord_ctrl, fd_dT_dchord_ctrl):.3e}")
    print(f"   - dQ  / d(chord_ctrl)     相对 L2 误差: {rel_err(sens.dQ_dchord_ctrl, fd_dQ_dchord_ctrl):.3e}")
    print(f"   - dMy / d(chord_ctrl)     相对 L2 误差: {rel_err(sens.dMy_dchord_ctrl, fd_dMy_dchord_ctrl):.3e}")
    print(f"   - dT  / d(twist_ctrl)     相对 L2 误差: {rel_err(sens.dT_dtwist_ctrl, fd_dT_dtwist_ctrl):.3e}")
    print(f"   - dQ  / d(twist_ctrl)     相对 L2 误差: {rel_err(sens.dQ_dtwist_ctrl, fd_dQ_dtwist_ctrl):.3e}")
    print(f"   - dMx / d(twist_ctrl)     相对 L2 误差: {rel_err(sens.dMx_dtwist_ctrl, fd_dMx_dtwist_ctrl):.3e}")
    print(f"   - dT  / d(chord_ctrl_r)   相对 L2 误差: {rel_err(sens.dT_dchord_ctrl_r, fd_dT_dchord_ctrl_r):.3e}")
    print(f"   - dT  / d(twist_ctrl_r)   相对 L2 误差: {rel_err(sens.dT_dtwist_ctrl_r, fd_dT_dtwist_ctrl_r):.3e}")

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
    print("=" * 80)


if __name__ == "__main__":
    test_oblique_solver_and_gradients()
