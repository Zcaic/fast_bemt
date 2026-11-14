# -*- coding: utf-8 -*-
"""
专门针对 fb.BSplineCurve 类的单元测试与独立可视化检查脚本
涵盖：
1. 多种构造方式（(N,2) 二维控制点矩阵、双一维数组、from_1d_values Greville 横坐标）与自定义阶数 (p=1,2,3,4)
2. 均匀 Clamped 节点向量、端点精确插值性质、参数域采样 sample() 与反求参数 invert_r_norm() 精度验证
3. 线性函数精确重构性、基函数矩阵单位分解性 (Partition of Unity) 与曲线斜率 slope_dy_dx() 有限差分验证
4. 控制点横/纵坐标扰动对叶素分布及 BEMT 气动力解析梯度 (链式法则) 的有限差分验证
5. 异常输入拦截测试（非单调控制点横坐标、阶数越界等）

运行方式：
- 直接作为脚本运行：python tests/test_bspline.py
- 直接运行并弹出曲线对比图：python tests/test_bspline.py --plot
- 使用 pytest 运行：pytest tests/test_bspline.py -v
"""

import os
import sys
import numpy as np
import pytest

# 确保优先从当前仓库的 python 目录导入 fast_bemt
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "python")))
import fast_bemt as fb


# ==============================================================================
# 1. 构造器、属性与自定义阶数测试
# ==============================================================================
def test_bspline_constructors_and_properties():
    """测试 BSplineCurve 的三种构造方式及基本属性访问"""
    ctrl_r = np.array([0.10, 0.25, 0.50, 0.75, 0.90, 1.00], dtype=np.float64)
    ctrl_v = np.array([0.08, 0.11, 0.10, 0.08, 0.06, 0.04], dtype=np.float64)
    ctrl_2d = np.column_stack([ctrl_r, ctrl_v])

    # 方式 1: (N, 2) 二维矩阵构造
    c1 = fb.BSplineCurve(ctrl_2d, degree=3)
    assert c1.degree == 3
    assert c1.num_ctrl_points == 6
    np.testing.assert_allclose(c1.ctrl_r_norm, ctrl_r, atol=1e-14)
    np.testing.assert_allclose(c1.ctrl_vals, ctrl_v, atol=1e-14)
    np.testing.assert_allclose(c1.ctrl_points, ctrl_2d, atol=1e-14)

    # 均匀 Clamped 节点向量长度应为 n_ctrl + degree + 1 = 6 + 3 + 1 = 10
    expected_knots = np.array([0.0, 0.0, 0.0, 0.0, 1.0 / 3.0, 2.0 / 3.0, 1.0, 1.0, 1.0, 1.0])
    np.testing.assert_allclose(c1.knots, expected_knots, atol=1e-14)

    # 方式 2: (ctrl_r_norm, ctrl_vals) 双一维数组构造
    c2 = fb.BSplineCurve(ctrl_r, ctrl_v, degree=3)
    r_query = np.linspace(0.10, 1.00, 50)
    np.testing.assert_allclose(c1(r_query), c2(r_query), atol=1e-14)

    # 方式 3: from_1d_values 自动计算 Greville 横坐标
    c3 = fb.BSplineCurve.from_1d_values(ctrl_v, degree=3, r_norm_min=0.10, r_norm_max=1.00)
    assert c3.num_ctrl_points == 6
    assert np.isclose(c3.ctrl_r_norm[0], 0.10, atol=1e-14)
    assert np.isclose(c3.ctrl_r_norm[-1], 1.00, atol=1e-14)
    assert np.all(np.diff(c3.ctrl_r_norm) > 0.0)


@pytest.mark.parametrize("degree", [1, 2, 3, 4])
def test_bspline_custom_degrees_and_clamped_endpoints(degree: int):
    """测试不同阶数 (p=1,2,3,4) 下的 Clamped 端点插值与参数域采样"""
    ctrl_2d = np.array([
        [0.12, 0.090],
        [0.30, 0.115],
        [0.55, 0.100],
        [0.80, 0.075],
        [1.00, 0.045],
    ], dtype=np.float64)

    curve = fb.BSplineCurve(ctrl_2d, degree=degree)
    assert curve.degree == degree
    assert len(curve.knots) == curve.num_ctrl_points + degree + 1

    # Clamped B 样条在 u=0 和 u=1 处必须精确经过首尾控制点
    r0, v0 = curve.eval_u(0.0)
    r1, v1 = curve.eval_u(1.0)
    assert np.isclose(r0, ctrl_2d[0, 0], atol=1e-14)
    assert np.isclose(v0, ctrl_2d[0, 1], atol=1e-14)
    assert np.isclose(r1, ctrl_2d[-1, 0], atol=1e-14)
    assert np.isclose(v1, ctrl_2d[-1, 1], atol=1e-14)

    # 单点按径向坐标查询首尾端点也必须精确一致
    assert np.isclose(curve.eval_at_r_norm(0.12), ctrl_2d[0, 1], atol=1e-13)
    assert np.isclose(curve.eval_at_r_norm(1.00), ctrl_2d[-1, 1], atol=1e-13)

    # 测试 sample(n_points) 返回的 (r_norm_array, val_array) 维度及横坐标单调性
    r_sample, v_sample = curve.sample(150)
    assert r_sample.shape == (150,)
    assert v_sample.shape == (150,)
    assert np.all(np.diff(r_sample) > 0.0), "采样点的径向坐标 r/R(u) 必须严格单调递增"


# ==============================================================================
# 2. 参数反求 invert_r_norm、线性重构与基函数矩阵/斜率验证
# ==============================================================================
def test_bspline_inversion_and_linear_precision():
    """验证参数反求精度以及 B 样条对仿射线性分布的精确重现性质"""
    ctrl_r = np.array([0.10, 0.22, 0.48, 0.77, 1.00], dtype=np.float64)
    # 构造一条线性函数 y = -0.18 * (r/R) + 0.24
    slope_exact = -0.18
    intercept_exact = 0.24
    ctrl_v = slope_exact * ctrl_r + intercept_exact

    curve = fb.BSplineCurve(ctrl_r, ctrl_v, degree=3)

    r_test = np.linspace(0.10, 1.00, 80)
    u_stars = curve.invert_r_norm(r_test)
    r_recovered, v_recovered = curve.eval_u_batch(u_stars)

    np.testing.assert_allclose(r_recovered, r_test, atol=1e-13)
    np.testing.assert_allclose(v_recovered, slope_exact * r_test + intercept_exact, atol=1e-13)

    # 斜率与批量求值验证
    slopes = curve.slope_dy_dx(r_test)
    np.testing.assert_allclose(slopes, np.full_like(r_test, slope_exact), atol=1e-11)

    v_batch = curve(r_test)
    np.testing.assert_allclose(v_batch, slope_exact * r_test + intercept_exact, atol=1e-13)


def test_bspline_basis_matrix_and_derivatives():
    """验证基函数矩阵 B 的单位分解性与控制点横/纵坐标解析导数"""
    ctrl_2d = np.array([
        [0.10, 0.080],
        [0.28, 0.115],
        [0.52, 0.098],
        [0.78, 0.072],
        [1.00, 0.042],
    ], dtype=np.float64)

    curve = fb.BSplineCurve(ctrl_2d, degree=3)
    r_grid = np.linspace(0.12, 0.98, 25)

    # 1. 基函数矩阵单位分解性: 每行之和恒等于 1，且 B @ ctrl_vals == curve(r_grid)
    B = curve.basis_matrix(r_grid)
    assert B.shape == (len(r_grid), curve.num_ctrl_points)
    np.testing.assert_allclose(np.sum(B, axis=1), np.ones(len(r_grid)), atol=1e-14)
    np.testing.assert_allclose(B @ curve.ctrl_vals, curve(r_grid), atol=1e-13)

    # 2. 曲线斜率 dy/d(r/R) 与中心有限差分对比
    eps_r = 1e-6
    dy_dx_ana = curve.slope_dy_dx(r_grid)
    dy_dx_fd = (curve(r_grid + eps_r) - curve(r_grid - eps_r)) / (2.0 * eps_r)
    np.testing.assert_allclose(dy_dx_ana, dy_dx_fd, rtol=1e-7, atol=1e-9)

    # 3. 验证隐式参数化下曲线值对控制点横坐标 P_x 的解析导数:
    #    dc(r_i)/dP_{x,j} = - (dy/dx)|_{r_i} * B_{ij}
    eps_px = 1e-6
    for j in range(curve.num_ctrl_points):
        ctrl_plus = ctrl_2d.copy()
        ctrl_minus = ctrl_2d.copy()
        ctrl_plus[j, 0] += eps_px
        ctrl_minus[j, 0] -= eps_px
        c_plus = fb.BSplineCurve(ctrl_plus, degree=3)
        c_minus = fb.BSplineCurve(ctrl_minus, degree=3)

        dc_dpx_fd = (c_plus(r_grid) - c_minus(r_grid)) / (2.0 * eps_px)
        dc_dpx_ana = -dy_dx_ana * B[:, j]
        np.testing.assert_allclose(dc_dpx_ana, dc_dpx_fd, rtol=1e-5, atol=1e-7)


# ==============================================================================
# 3. 传入 RotorGeometry 及 BEMT 控制点解析梯度端到端验证
# ==============================================================================
def test_bspline_integration_with_rotor_geometry_and_bemt():
    """测试手动实例化 chord_curve 和 twist_curve 后传入 RotorGeometry 并校验解析梯度"""
    chord_ctrl = np.array([
        [0.10, 0.085],
        [0.30, 0.110],
        [0.60, 0.095],
        [0.85, 0.072],
        [1.00, 0.048],
    ], dtype=np.float64)

    twist_ctrl = np.column_stack([
        chord_ctrl[:, 0],
        np.deg2rad([22.0, 17.0, 11.5, 7.5, 4.5]),
    ])

    chord_curve = fb.BSplineCurve(chord_ctrl, degree=3)
    twist_curve = fb.BSplineCurve(twist_ctrl, degree=3)

    # 构建 5 个控制截面的翼型表 (r/R = 0.10, 0.20, 0.50, 0.75, 1.00)
    alpha_grid = np.linspace(-0.5, 0.5, 101)
    section_r_norm = np.array([0.10, 0.20, 0.50, 0.75, 1.00], dtype=np.float64)
    section_airfoils = [
        fb.Airfoil3D.from_1d_polar(
            alpha_grid,
            (5.8 + 0.3 * k) * alpha_grid,
            0.010 + 0.02 * alpha_grid**2,
        )
        for k in range(len(section_r_norm))
    ]

    geom = fb.RotorGeometry(
        num_blades=3,
        radius=1.2,
        hub_radius=0.12,
        n_radial=24,
        chord_curve=chord_curve,
        twist_curve=twist_curve,
        airfoils=section_airfoils,
        section_r_norm=section_r_norm,
        spin_dir=1,
    )

    # 确认 RotorGeometry 中的叶素弦长/扭转角与独立 BSplineCurve 求值完全一致
    r_norm_elems = geom.r / geom.radius
    np.testing.assert_allclose(geom.chord, chord_curve(r_norm_elems), atol=1e-14)
    np.testing.assert_allclose(geom.twist, twist_curve(r_norm_elems), atol=1e-14)

    # 求解 BEMT 并验证对 BSplineCurve 控制点的解析灵敏度
    cfg = fb.SolverConfig(n_azimuth=18)
    cond = fb.OperatingCondition(vx=8.0, vy=0.0, vz=2.0, omega=180.0, pitch_0=np.deg2rad(6.0))
    solver = fb.BEMTSolver(geom, cfg)
    res = solver.solve(cond, compute_gradients=True)

    assert res.converged
    assert res.sensitivity.valid

    # 有限差分校验第 2 个弦长控制点纵坐标与横坐标的梯度
    eps = 1e-6
    idx = 2

    # (a) 扰动弦长控制点纵坐标 ctrl_vals[idx]
    c_ctrl_p = chord_ctrl.copy()
    c_ctrl_m = chord_ctrl.copy()
    c_ctrl_p[idx, 1] += eps
    c_ctrl_m[idx, 1] -= eps
    res_p = fb.BEMTSolver(
        fb.RotorGeometry(3, 1.2, 0.12, 24, fb.BSplineCurve(c_ctrl_p, 3), twist_curve, section_airfoils, section_r_norm),
        cfg,
    ).solve(cond)
    res_m = fb.BEMTSolver(
        fb.RotorGeometry(3, 1.2, 0.12, 24, fb.BSplineCurve(c_ctrl_m, 3), twist_curve, section_airfoils, section_r_norm),
        cfg,
    ).solve(cond)
    dT_dc_fd = (res_p.thrust - res_m.thrust) / (2.0 * eps)
    assert np.isclose(res.sensitivity.dT_dchord_ctrl[idx], dT_dc_fd, rtol=1e-5)

    # (b) 扰动弦长控制点横坐标 ctrl_r_norm[idx]
    c_ctrl_rp = chord_ctrl.copy()
    c_ctrl_rm = chord_ctrl.copy()
    c_ctrl_rp[idx, 0] += eps
    c_ctrl_rm[idx, 0] -= eps
    res_rp = fb.BEMTSolver(
        fb.RotorGeometry(3, 1.2, 0.12, 24, fb.BSplineCurve(c_ctrl_rp, 3), twist_curve, section_airfoils, section_r_norm),
        cfg,
    ).solve(cond)
    res_rm = fb.BEMTSolver(
        fb.RotorGeometry(3, 1.2, 0.12, 24, fb.BSplineCurve(c_ctrl_rm, 3), twist_curve, section_airfoils, section_r_norm),
        cfg,
    ).solve(cond)
    dT_dcr_fd = (res_rp.thrust - res_rm.thrust) / (2.0 * eps)
    assert np.isclose(res.sensitivity.dT_dchord_ctrl_r[idx], dT_dcr_fd, rtol=1e-4)


# ==============================================================================
# 4. 非法输入异常拦截测试
# ==============================================================================
def test_bspline_invalid_inputs():
    """测试非单调控制点横坐标或非法阶数时是否正确抛出 ValueError"""
    # 横坐标非严格递增
    bad_ctrl = np.array([
        [0.10, 0.08],
        [0.50, 0.10],
        [0.45, 0.09],
        [1.00, 0.05],
    ], dtype=np.float64)
    with pytest.raises(ValueError):
        fb.BSplineCurve(bad_ctrl, degree=2)

    # 阶数大于等于控制点数
    valid_ctrl = np.array([
        [0.10, 0.08],
        [0.50, 0.10],
        [1.00, 0.05],
    ], dtype=np.float64)
    with pytest.raises(ValueError):
        fb.BSplineCurve(valid_ctrl, degree=3)


# ==============================================================================
# 5. 命令行直接运行入口（支持 --plot 可视化检查）
# ==============================================================================
if __name__ == "__main__":
    print("=== Running BSplineCurve Unit Tests ===")
    test_bspline_constructors_and_properties()
    print("[PASS] test_bspline_constructors_and_properties")

    for deg in (1, 2, 3, 4):
        test_bspline_custom_degrees_and_clamped_endpoints(deg)
    print("[PASS] test_bspline_custom_degrees_and_clamped_endpoints (degree=1,2,3,4)")

    test_bspline_inversion_and_linear_precision()
    print("[PASS] test_bspline_inversion_and_linear_precision")

    test_bspline_basis_matrix_and_derivatives()
    print("[PASS] test_bspline_basis_matrix_and_derivatives")

    test_bspline_integration_with_rotor_geometry_and_bemt()
    print("[PASS] test_bspline_integration_with_rotor_geometry_and_bemt")

    test_bspline_invalid_inputs()
    print("[PASS] test_bspline_invalid_inputs")
    print("=== All BSplineCurve tests passed! ===")

    if "--plot" in sys.argv:
        import matplotlib.pyplot as plt

        chord_ctrl = np.array([
            [0.10, 0.080],
            [0.30, 0.115],
            [0.60, 0.095],
            [0.85, 0.070],
            [1.00, 0.045],
        ])
        twist_ctrl = np.column_stack([
            [0.10, 0.30, 0.60, 0.85, 1.00],
            np.deg2rad([24.0, 18.0, 11.5, 7.0, 4.0]),
        ])

        chord_curve = fb.BSplineCurve(chord_ctrl, degree=3)
        twist_curve = fb.BSplineCurve(twist_ctrl, degree=3)

        r_c, val_c = chord_curve.sample(200)
        r_t, val_t = twist_curve.sample(200)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
        ax1.plot(r_c, val_c, lw=2, label="Chord B-Spline (degree=3)")
        ax1.plot(chord_curve.ctrl_r_norm, chord_curve.ctrl_vals, "o--", label="Control Polygon")
        ax1.set_xlabel("r/R [-]")
        ax1.set_ylabel("Chord [m]")
        ax1.set_title("Chord Distribution Curve")
        ax1.grid(True, alpha=0.3)
        ax1.legend()

        ax2.plot(r_t, np.rad2deg(val_t), lw=2, color="tab:orange", label="Twist B-Spline (degree=3)")
        ax2.plot(twist_curve.ctrl_r_norm, np.rad2deg(twist_curve.ctrl_vals), "s--", color="tab:red", label="Control Polygon")
        ax2.set_xlabel("r/R [-]")
        ax2.set_ylabel("Twist [deg]")
        ax2.set_title("Twist Distribution Curve")
        ax2.grid(True, alpha=0.3)
        ax2.legend()

        plt.tight_layout()
        plt.show()