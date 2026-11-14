# -*- coding: utf-8 -*-
"""
针对 AeroSandbox (KulfanAirfoil CST + NeuralFoil) 翼型极曲线与导数接入、
三级链式法则 CST 气动梯度校验、SciPy SLSQP 桨叶平面形状+翼型 CST 联合优化器 (RotorBladeOptimizer)
以及 OpenMDAO 显式组件 (BEMTRotorComp) 的完整测试脚本。

运行方式：
- 直接运行：python tests/test_optimization.py
- 使用 pytest 运行：pytest tests/test_optimization.py -v
"""

import os
import sys
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "python")))
import fast_bemt as fb


def test_kulfan_cst_airfoil_and_thickness_jacobian():
    """测试 KulfanCSTAirfoil 的构造、NACA 4 位数 CST 拟合与厚度对 CST 权重的线性解析雅可比"""
    kaf = fb.KulfanCSTAirfoil.from_naca4("4412")
    assert kaf.lower_weights.shape == (8,)
    assert kaf.upper_weights.shape == (8,)

    x_chk = np.linspace(0.05, 0.95, 19)
    t0 = kaf.thickness_at(x_chk)
    assert np.max(t0) > 0.10 and np.max(t0) < 0.14  # NACA 4412 最大相对厚度约为 12%

    # 校验厚度分布对 CST 参数向量 (17 维) 的解析雅可比与中心差分一致性
    # 中心差分步长 eps=1e-6 下 IEEE-754 双精度舍入误差地板约为 ~1e-10，故设 rtol=1e-6, atol=1e-9
    jac_ana = fb.KulfanCSTAirfoil.thickness_jacobian_wrt_vector(x_chk, include_te=False)
    v0 = kaf.to_vector(include_te=False)
    eps = 1e-6
    for j in range(v0.size):
        vp = v0.copy()
        vm = v0.copy()
        vp[j] += eps
        vm[j] -= eps
        tp = kaf.with_vector(vp).thickness_at(x_chk)
        tm = kaf.with_vector(vm).thickness_at(x_chk)
        jac_fd = (tp - tm) / (2.0 * eps)
        np.testing.assert_allclose(jac_ana[:, j], jac_fd, rtol=1e-6, atol=1e-9)


def test_cst_chain_rule_gradients_and_coupled_optimization():
    """
    测试三级链式法则计算的 d(T, Q, P)/d(CST) 梯度以及
    【弦长 B 样条控制点 + 扭转 B 样条控制点 + 控制截面翼型 CST 权重】联合气动优化
    """
    ctrl_r = np.array([0.12, 0.32, 0.58, 0.82, 1.00], dtype=np.float64)
    # 初始给定未优化的等弦长 + 平直扭转 + 对称无弯度 NACA 0015/0012 翼型
    init_chord = fb.BSplineCurve(ctrl_r, np.full_like(ctrl_r, 0.085), degree=3)
    init_twist = fb.BSplineCurve(ctrl_r, np.deg2rad([16.0, 16.0, 16.0, 16.0, 16.0]), degree=3)

    section_r_norm = np.array([0.12, 0.55, 1.00], dtype=np.float64)
    section_kulfans = [
        fb.KulfanCSTAirfoil.from_naca4("0015"),
        fb.KulfanCSTAirfoil.from_naca4("0012"),
        fb.KulfanCSTAirfoil.from_naca4("0012"),
    ]

    cond = fb.OperatingCondition(vx=0.0, vy=0.0, vz=2.0, omega=220.0, pitch_0=0.0)
    cfg = fb.SolverConfig(n_azimuth=1, enable_tip_loss=True)

    nf_kwargs = {
        "alpha_deg_grid": np.linspace(-14.0, 16.0, 61),
        "re_grid": np.array([1.0e5, 5.0e5, 2.0e6]),
        "ma_grid": np.array([0.0, 0.3, 0.6]),
        "model_size": "medium",
    }

    optimizer = fb.RotorBladeOptimizer(
        num_blades=3,
        radius=1.0,
        hub_radius=0.12,
        n_radial=18,
        init_chord_curve=init_chord,
        init_twist_curve=init_twist,
        section_airfoils=section_kulfans,
        section_r_norm=section_r_norm,
        cond=cond,
        solver_cfg=cfg,
        neuralfoil_kwargs=nf_kwargs,
    )

    init_geom = fb.RotorGeometry(
        3, 1.0, 0.12, 18, init_chord, init_twist, optimizer.airfoils, section_r_norm
    )
    init_res = fb.BEMTSolver(init_geom, cfg).solve(cond, compute_gradients=True)

    # 1. 验证三级链式法则 CST 气动梯度输出维度与有限性
    cst_grads = fb.compute_cst_aerodynamic_gradients(
        geom=init_geom,
        cond=cond,
        cfg=cfg,
        res=init_res,
        section_kulfan_airfoils=section_kulfans,
        model_size=nf_kwargs["model_size"],
    )
    assert cst_grads["dT_dcst"].shape == (3, 17)
    assert cst_grads["dP_dcst"].shape == (3, 17)
    assert np.all(np.isfinite(cst_grads["dT_dcst"]))
    assert np.all(np.isfinite(cst_grads["dP_dcst"]))

    # 2. 执行【弦长控制点 + 扭转控制点 + 翼型 CST 权重】联合优化
    opt_out = optimizer.optimize(
        target_thrust=init_res.thrust,
        max_solidity=0.10,
        objective="min_power",
        optimize_chord=True,
        optimize_twist=True,
        optimize_cst=True,
        cst_weight_delta_bound=0.06,
        min_thickness_ratio_scale=0.85,
        chord_bounds=(0.025, 0.14),
        twist_deg_bounds=(2.0, 32.0),
        smoothness_weight=5e-3,
        cst_reg_weight=5e-3,
        maxiter=35,
    )

    assert np.isclose(opt_out.bemt_result.thrust, init_res.thrust, rtol=5e-3)
    assert opt_out.bemt_result.power < init_res.power * 0.96
    print(
        f"   [平面形状 + Kulfan CST 联合优化] 推力保持 T = {opt_out.bemt_result.thrust:.2f} N "
        f"(初始 {init_res.thrust:.2f} N) | "
        f"功率从 {init_res.power / 1e3:.3f} kW 降至 {opt_out.bemt_result.power / 1e3:.3f} kW "
        f"(节省 {(1.0 - opt_out.bemt_result.power / init_res.power) * 100:.2f}%, 迭代 {opt_out.nit} 步)"
    )


def test_openmdao_component_with_cst_if_available():
    """若环境中安装了 openmdao，则验证开启 optimize_cst=True 时 BEMTRotorComp 的运行与偏导计算"""
    try:
        import openmdao.api as om
    except ImportError:
        print("   [Info] openmdao 未安装，跳过 BEMTRotorComp CST 组件测试。")
        return

    prob = om.Problem()
    section_r_norm = [0.12, 0.55, 1.00]
    section_kulfans = [
        fb.KulfanCSTAirfoil.from_naca4("0015"),
        fb.KulfanCSTAirfoil.from_naca4("2412"),
        fb.KulfanCSTAirfoil.from_naca4("2410"),
    ]

    comp = fb.BEMTRotorComp(
        num_blades=3,
        radius=1.0,
        hub_radius=0.12,
        n_radial=12,
        n_chord_ctrl=4,
        n_twist_ctrl=4,
        bspline_degree=3,
        optimize_cst=True,
        section_r_norm=section_r_norm,
        section_airfoils=section_kulfans,
        neuralfoil_kwargs={
            "alpha_deg_grid": np.linspace(-14.0, 16.0, 31),
            "re_grid": np.array([1.0e5, 1.0e6]),
            "ma_grid": np.array([0.0, 0.5]),
            "model_size": "medium",
        },
        solver_config=fb.SolverConfig(n_azimuth=1),
    )
    prob.model.add_subsystem("rotor", comp, promotes=["*"])
    prob.setup(force_alloc_complex=False)
    prob.run_model()
    assert float(prob.get_val("thrust").item()) > 10.0
    totals = prob.compute_totals(of=["thrust", "power"], wrt=["chord_ctrl", "twist_ctrl", "kulfan_cst_params"])
    assert np.all(np.isfinite(totals["thrust", "kulfan_cst_params"]))
    assert np.all(np.isfinite(totals["power", "kulfan_cst_params"]))
    print("   [OpenMDAO BEMTRotorComp + CST] 全导数矩阵计算通过验证！")


if __name__ == "__main__":
    print("=== Running Kulfan CST & Coupled Rotor Blade Optimization Tests ===")
    test_kulfan_cst_airfoil_and_thickness_jacobian()
    print("[PASS] test_kulfan_cst_airfoil_and_thickness_jacobian")

    test_cst_chain_rule_gradients_and_coupled_optimization()
    print("[PASS] test_cst_chain_rule_gradients_and_coupled_optimization")

    test_openmdao_component_with_cst_if_available()
    print("[PASS] test_openmdao_component_with_cst_if_available")
    print("=== All Kulfan CST & optimization tests passed! ===")
