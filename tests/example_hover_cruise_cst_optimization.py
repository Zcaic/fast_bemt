# -*- coding: utf-8 -*-
r"""
通用多工况桨叶平面形状（B-Spline 弦长/扭转）+ 截面翼型 Kulfan (CST) + 转速/总距角联合优化案例模板
===============================================================================================
本脚本将五个典型优化场景分别封装为完全独立的案例函数，在底部的 `if __name__ == "__main__":` 中
只需注释掉任意一行函数调用，即可自由选择运行单个或多个案例：

1. `run_case_a_hover_cruise_min_cruise_power(show_plot=False)`
   - 【案例 A】：使用 `HoverCruiseBladeOptimizer` 一键切换目标函数（最小化巡航功率 `min_cruise_power`）与推力不等式约束。
2. `run_case_b_multipoint_min_hover_power(show_plot=False)`
   - 【案例 B】：使用声明式框架 `MultiPointBladeOptimizer` 进行悬停+巡航双工况优化（最小化悬停功率）。
3. `run_case_c_three_point_oblique_transition(show_plot=False)`
   - 【案例 C】：使用 `MultiPointBladeOptimizer` 进行【悬停 + 45° 斜入流倾转过渡 + 巡航】三工况联合优化，并约束过渡态俯仰力矩 |My|。
4. `run_case_d_large_rotor_2p1m_five_section(show_plot=False)`
   - 【案例 D】：R = 2.1 m 大直径三叶倾转旋翼 5 截面翼型 CST (厚度 >= [0.28, 0.26, 0.18, 0.12, 0.08]) +
     根/尖弦长约束 (c_root = 0.318 m, c_tip >= 0.15 m) + 4.5 km 高空 111.11 m/s 巡航与悬停变总距优化。
5. `run_case_e_blade_count_comparison(blade_counts=(3, 4, 5), mode="both", show_plot=False)`
   - 【案例 E】：基于案例 D (R = 2.1 m)，对比桨叶数 B = 3, 4, 5 在【保持总实度不变 (弦长按 3/B 缩放)】与
     【保持单叶弦长约束不变】两种方案下的理论最高悬停效率 FM_max(DL, B) 与 fast_bemt 优化结果。

运行方式：
- 直接运行：python tests/example_hover_cruise_cst_optimization.py
- 弹出可视化对比图：python tests/example_hover_cruise_cst_optimization.py --plot
"""

import os
import sys
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "python")))
import fast_bemt as fb


# ==============================================================================
# 公共辅助函数：基准几何构建、大气解算、理论极限 FM 估算与绘图
# ==============================================================================
def build_baseline_1m_rotor_setup():
    """构建案例 A/B/C 共用的 R = 1.0 m 基准三叶旋翼几何、初始 B 样条与 3 截面 Kulfan CST 翼型"""
    num_blades = 3
    radius = 1.00      # 旋翼半径 R = 1.0 m
    hub_radius = 0.12  # 桨毂半径 R_hub = 0.12 m
    n_radial = 24      # 径向离散为 24 个叶素

    ctrl_r = np.array([0.12, 0.30, 0.55, 0.80, 1.00], dtype=np.float64)
    init_chord_vals = np.array([0.085, 0.095, 0.090, 0.075, 0.058], dtype=np.float64)
    init_twist_vals = np.deg2rad([24.0, 20.0, 16.0, 12.0, 8.0])

    init_chord_curve = fb.BSplineCurve(ctrl_r, init_chord_vals, degree=3)
    init_twist_curve = fb.BSplineCurve(ctrl_r, init_twist_vals, degree=3)

    section_r_norm = np.array([0.12, 0.55, 1.00], dtype=np.float64)
    init_section_kulfans = [
        fb.KulfanCSTAirfoil.from_naca4("2415"),  # 叶根控制截面 (r/R = 0.12): 15% 厚度
        fb.KulfanCSTAirfoil.from_naca4("2412"),  # 中展控制截面 (r/R = 0.55): 12% 厚度
        fb.KulfanCSTAirfoil.from_naca4("2410"),  # 叶尖控制截面 (r/R = 1.00): 10% 厚度
    ]

    hover_cond_init = fb.OperatingCondition(vx=0.0, vy=0.0, vz=0.0, omega=215.0, pitch_0=0.0)
    v_cruise = 26.0
    cruise_cond_init = fb.OperatingCondition(vx=0.0, vy=0.0, vz=v_cruise, omega=185.0, pitch_0=0.0)

    nf_kwargs = {
        "alpha_deg_grid": np.linspace(-16.0, 18.0, 69),
        "re_grid": np.array([1.0e5, 5.0e5, 1.5e6, 4.0e6]),
        "ma_grid": np.array([0.0, 0.25, 0.50, 0.75]),
        "model_size": "large",
    }

    common_geom_constraints = dict(
        optimize_chord=True,
        chord_bounds=(0.050, 0.120),
        min_tip_chord=0.055,                  # 桨尖弦长 c(R) >= 0.055 m
        min_root_chord=0.075,                 # 桨根弦长 c(R_hub) >= 0.075 m
        min_tip_to_max_chord_ratio=0.50,      # 尖峰比 c_tip / c_max >= 0.50
        chord_taper_from_index=1,             # 从第 1 个控制点向外单调收缩
        max_chord_step=0.025,                 # 相邻弦长控制点最大步长 <= 0.025 m
        optimize_twist=True,
        twist_deg_bounds=(5.0, 30.0),
        min_tip_twist_deg=5.0,
        max_root_twist_deg=28.0,
        monotonic_twist=True,                 # 从根到尖单调递减洗出
        min_twist_step_deg=1.5,               # 相邻控制点至少递减 1.5°
        max_twist_step_deg=7.5,               # 相邻控制点最多递减 7.5°
        optimize_cst=True,
        cst_weight_delta_bound=0.05,
        min_thickness_ratio_scale=0.88,       # 保持翼型厚度 >= 88% 初始厚度
        max_solidity=0.095,
        smoothness_weight=6e-3,
        cst_reg_weight=2e-3,
        maxiter=70,
        ftol=5e-4,
    )

    return (
        num_blades,
        radius,
        hub_radius,
        n_radial,
        init_chord_curve,
        init_twist_curve,
        section_r_norm,
        init_section_kulfans,
        hover_cond_init,
        cruise_cond_init,
        nf_kwargs,
        common_geom_constraints,
    )


def compute_atmosphere_from_p_rho(pressure_pa: float, rho_kg_m3: float) -> tuple[float, float, float]:
    """
    根据给定的大气静压 p [Pa] 与空气密度 rho [kg/m^3]，由理想气体状态方程与 Sutherland 公式精确计算：
    - 静温 T_kelvin = p / (rho * R_gas) [K]
    - 当地声速 a_sound = sqrt(gamma * p / rho) [m/s]
    - 空气动力粘度 mu = mu_0 * (T / T_0)^1.5 * (T_0 + S) / (T + S) [Pa*s]
    """
    gamma = 1.4
    r_gas = 287.05287
    t_kelvin = pressure_pa / (rho_kg_m3 * r_gas)
    a_sound = float(np.sqrt(gamma * pressure_pa / rho_kg_m3))

    t0 = 288.15
    mu0 = 1.7894e-5
    s_const = 110.4
    mu = float(mu0 * ((t_kelvin / t0) ** 1.5) * ((t0 + s_const) / (t_kelvin + s_const)))
    return float(t_kelvin), a_sound, mu


def estimate_theoretical_hover_fm_limit(
    thrust_n: float,
    radius_m: float,
    hub_radius_m: float,
    num_blades: int,
    omega_rad_s: float,
    rho_kg_m3: float = 1.225,
    ld_eff_ref: float = 98.0,
    ref_blades: int = 3,
    scale_chord_for_constant_solidity: bool = True,
) -> dict[str, float]:
    r"""
    根据指定桨盘载荷 DL = T / (pi * R^2) 与桨叶数 B，使用叶素动量闭式积分公式估算理论最高悬停效率 FM_max：

        FM_max(DL, B) = 1 / [ kappa_eff(DL, B) + 2 * V_tip * sqrt(2 * rho) / (3 * sqrt(DL) * (L/D)_eff) ]

    其中综合诱导功率修正因子 kappa_eff 包含四项不可消除的物理机制：
        kappa_eff = 1.0 + dk_tip + dk_cutout + dk_swirl
        - dk_tip    = ((1.386 + 0.693 * r_h_bar) / (B * V_tip)) * sqrt(DL / rho)   [Prandtl 叶尖+叶根涡损失]
        - dk_cutout = r_h_bar^2 / (1 - r_h_bar^2)                                  [桨毂根切面积损失]
        - dk_swirl  = (DL / (rho * V_tip^2)) * ln(1 / r_h_bar)                     [尾迹切向旋转动能损失]
    当保持总实度不变 (scale_chord_for_constant_solidity=True) 时，单叶弦长按 c ~ 1/B 缩窄，
    考虑湍流边界层摩擦阻力随雷诺数的幂律关系 Cd0 ~ Re^(-0.2)，有效升阻比按 (ref_blades / B)^0.2 修正。
    """
    area = float(np.pi * (radius_m ** 2))
    dl = thrust_n / area
    v_tip = omega_rad_s * radius_m
    r_h_bar = hub_radius_m / radius_m

    dk_tip = ((1.386 + 0.693 * r_h_bar) / (num_blades * v_tip)) * float(np.sqrt(dl / rho_kg_m3))
    dk_cutout = (r_h_bar ** 2) / (1.0 - r_h_bar ** 2)
    dk_swirl = (dl / (rho_kg_m3 * (v_tip ** 2))) * float(np.log(1.0 / r_h_bar))
    kappa_eff = 1.0 + dk_tip + dk_cutout + dk_swirl

    if scale_chord_for_constant_solidity:
        ld_eff = ld_eff_ref * ((ref_blades / float(num_blades)) ** 0.20)
    else:
        ld_eff = ld_eff_ref

    profile_ratio = (2.0 * v_tip * float(np.sqrt(2.0 * rho_kg_m3))) / (3.0 * float(np.sqrt(dl)) * ld_eff)
    fm_max = 1.0 / (kappa_eff + profile_ratio)
    p_ideal = thrust_n * float(np.sqrt(dl / (2.0 * rho_kg_m3)))
    p_min_shaft = p_ideal / fm_max

    return {
        "dl": dl,
        "dk_tip": dk_tip,
        "dk_cutout": dk_cutout,
        "dk_swirl": dk_swirl,
        "kappa_eff": kappa_eff,
        "ld_eff": ld_eff,
        "profile_ratio": profile_ratio,
        "fm_max": fm_max,
        "p_ideal_kw": p_ideal / 1e3,
        "p_min_shaft_kw": p_min_shaft / 1e3,
    }


def _plot_optimization_summary(
    case_title: str,
    init_chord_curve: fb.BSplineCurve,
    opt_chord_curve: fb.BSplineCurve,
    init_twist_curve: fb.BSplineCurve,
    opt_twist_curve: fb.BSplineCurve,
    section_r_norm: np.ndarray,
    init_kulfans: list[fb.KulfanCSTAirfoil],
    opt_kulfans: list[fb.KulfanCSTAirfoil],
    hover_power_hist: list[float],
    cruise_power_hist: list[float],
    min_tip_chord_val: float,
) -> None:
    """绘制单个优化案例的四合一对比图（弦长分布、扭转分布、各截面 CST 翼型对比、功率收敛历程）"""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(12, 8.5))
    fig.suptitle(case_title, fontsize=13, fontweight="bold")

    r_c0, val_c0 = init_chord_curve.sample(200)
    r_c1, val_c1 = opt_chord_curve.sample(200)
    ax = axes[0, 0]
    ax.plot(r_c0, val_c0, "--", color="gray", label="Initial Chord")
    ax.plot(r_c1, val_c1, "-", lw=2, color="tab:blue", label="Optimized Chord")
    ax.plot(opt_chord_curve.ctrl_r_norm, opt_chord_curve.ctrl_vals, "o", color="tab:blue")
    ax.axhline(min_tip_chord_val, ls=":", color="tab:red", label=f"Min Tip Chord ({min_tip_chord_val:.3f} m)")
    ax.set_xlabel("r/R [-]")
    ax.set_ylabel("Chord [m]")
    ax.set_title("Spanwise Chord Distribution")
    ax.grid(True, alpha=0.3)
    ax.legend()

    r_t0, val_t0 = init_twist_curve.sample(200)
    r_t1, val_t1 = opt_twist_curve.sample(200)
    ax = axes[0, 1]
    ax.plot(r_t0, np.rad2deg(val_t0), "--", color="gray", label="Initial Twist")
    ax.plot(r_t1, np.rad2deg(val_t1), "-", lw=2, color="tab:orange", label="Optimized Twist")
    ax.plot(opt_twist_curve.ctrl_r_norm, np.rad2deg(opt_twist_curve.ctrl_vals), "s", color="tab:orange")
    ax.set_xlabel("r/R [-]")
    ax.set_ylabel("Twist [deg]")
    ax.set_title("Spanwise Twist Distribution")
    ax.grid(True, alpha=0.3)
    ax.legend()

    ax = axes[1, 0]
    for m, (k0, k1) in enumerate(zip(init_kulfans, opt_kulfans)):
        x_af, yu0, yl0 = k0.surface_coordinates()
        _, yu1, yl1 = k1.surface_coordinates()
        offset = -0.26 * m
        line_opt = ax.plot(x_af, yu1 + offset, "-", lw=1.8, label=f"Sec #{m+1} (r/R={section_r_norm[m]:.2f})")[0]
        col = line_opt.get_color()
        ax.plot(x_af, yl1 + offset, "-", lw=1.8, color=col)
        ax.plot(x_af, yu0 + offset, "--", color=col, alpha=0.45)
        ax.plot(x_af, yl0 + offset, "--", color=col, alpha=0.45)
    ax.set_xlabel("x/c [-]")
    ax.set_ylabel("y/c (vertically shifted)")
    ax.set_title("Kulfan (CST) Airfoils: Dashed=Init, Solid=Opt")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=8)

    ax = axes[1, 1]
    iters = np.arange(1, len(hover_power_hist) + 1)
    ax.plot(iters, np.array(hover_power_hist) / 1e3, "-o", ms=3, color="tab:red", label="Hover Power [kW]")
    ax.plot(iters, np.array(cruise_power_hist) / 1e3, "-s", ms=3, color="tab:blue", label="Cruise Power [kW]")
    ax.set_xlabel("Function Evaluations")
    ax.set_ylabel("Shaft Power [kW]")
    ax.set_title("Optimization Power Convergence History")
    ax.grid(True, alpha=0.3)
    ax.legend()

    plt.tight_layout()
    plt.show()


# ==============================================================================
# 独立案例函数 1：【案例 A】HoverCruiseBladeOptimizer 最小化巡航功率
# ==============================================================================
def run_case_a_hover_cruise_min_cruise_power(show_plot: bool = False) -> fb.HoverCruiseOptimizationResult:
    """
    【案例 A】使用 HoverCruiseBladeOptimizer，将优化目标设为最小化巡航功率 (min_cruise_power)，
    同时将悬停拉力与巡航拉力设为不等式下限约束 (T_hover >= 1.02 * T_h0, T_cruise >= T_c0)。
    """
    (
        num_blades,
        radius,
        hub_radius,
        n_radial,
        init_chord_curve,
        init_twist_curve,
        section_r_norm,
        init_section_kulfans,
        hover_cond_init,
        cruise_cond_init,
        nf_kwargs,
        common_geom_constraints,
    ) = build_baseline_1m_rotor_setup()

    optimizer = fb.HoverCruiseBladeOptimizer(
        num_blades=num_blades,
        radius=radius,
        hub_radius=hub_radius,
        n_radial=n_radial,
        init_chord_curve=init_chord_curve,
        init_twist_curve=init_twist_curve,
        section_airfoils=init_section_kulfans,
        section_r_norm=section_r_norm,
        hover_cond=hover_cond_init,
        cruise_cond=cruise_cond_init,
        neuralfoil_kwargs=nf_kwargs,
    )

    init_geom = fb.RotorGeometry(
        num_blades, radius, hub_radius, n_radial,
        init_chord_curve, init_twist_curve, optimizer.airfoils, section_r_norm
    )
    init_hover_res = fb.BEMTSolver(init_geom, optimizer.hover_cfg).solve(hover_cond_init)
    init_cruise_res = fb.BEMTSolver(init_geom, optimizer.cruise_cfg).solve(cruise_cond_init)
    init_eta_cruise = (init_cruise_res.thrust * cruise_cond_init.vz) / init_cruise_res.power

    min_hover_thrust_req = init_hover_res.thrust * 1.02
    min_cruise_thrust_req = init_cruise_res.thrust
    min_hover_fm_req = max(init_hover_res.fm + 0.008, 0.70)

    print("=" * 102)
    print("【案例 A】HoverCruiseBladeOptimizer: 目标 = 最小化巡航功率 (min_cruise_power)")
    print(f"  -> [初始状态] 悬停: T = {init_hover_res.thrust:7.2f} N, P = {init_hover_res.power/1e3:6.3f} kW, FM = {init_hover_res.fm:.4f} | "
          f"巡航: T = {init_cruise_res.thrust:7.2f} N, P = {init_cruise_res.power/1e3:6.3f} kW, eta = {init_eta_cruise*100:5.2f}%")

    opt_result = optimizer.optimize(
        objective="min_cruise_power",
        min_hover_fm=min_hover_fm_req,
        min_hover_thrust=min_hover_thrust_req,
        min_cruise_thrust=min_cruise_thrust_req,
        omega_hover_bounds=(160.0, 260.0),
        omega_cruise_bounds=(120.0, 220.0),
        optimize_omega_hover=True,
        optimize_omega_cruise=True,
        optimize_pitch_0=False,
        **common_geom_constraints,
    )

    print(f"  -> [收敛状态] Success = {opt_result.success} (迭代 {opt_result.nit} 步)")
    print(f"  -> [悬停结果] T = {opt_result.hover_result.thrust:7.2f} N (要求 >= {min_hover_thrust_req:.1f} N), "
          f"P = {opt_result.hover_result.power/1e3:6.3f} kW, FM = {opt_result.hover_fm:.4f}, RPM = {opt_result.rpm_hover:.0f}")
    print(f"  -> [巡航结果] T = {opt_result.cruise_result.thrust:7.2f} N (要求 >= {min_cruise_thrust_req:.1f} N), "
          f"P = {opt_result.cruise_result.power/1e3:6.3f} kW (初始 {init_cruise_res.power/1e3:6.3f} kW, "
          f"降低 {(1.0 - opt_result.cruise_result.power/init_cruise_res.power)*100:.2f}%), "
          f"eta = {opt_result.cruise_efficiency*100:5.2f}%, RPM = {opt_result.rpm_cruise:.0f}")
    print("=" * 102)

    assert opt_result.hover_result.thrust >= min_hover_thrust_req - 1.0
    assert opt_result.cruise_result.thrust >= min_cruise_thrust_req - 1.0
    assert opt_result.hover_fm >= min_hover_fm_req - 1e-3
    assert opt_result.cruise_result.power < init_cruise_res.power

    if show_plot:
        _plot_optimization_summary(
            case_title="Case A: Minimize Cruise Power (R=1.0m, 3 Sections)",
            init_chord_curve=init_chord_curve,
            opt_chord_curve=opt_result.chord_curve,
            init_twist_curve=init_twist_curve,
            opt_twist_curve=opt_result.twist_curve,
            section_r_norm=section_r_norm,
            init_kulfans=init_section_kulfans,
            opt_kulfans=opt_result.section_kulfan_airfoils,
            hover_power_hist=opt_result.history_power_hover,
            cruise_power_hist=opt_result.history_power_cruise,
            min_tip_chord_val=0.055,
        )

    return opt_result


# ==============================================================================
# 独立案例函数 2：【案例 B】MultiPointBladeOptimizer 最小化悬停功率
# ==============================================================================
def run_case_b_multipoint_min_hover_power(show_plot: bool = False) -> fb.MultiPointOptimizationResult:
    """
    【案例 B】使用声明式通用框架 MultiPointBladeOptimizer 进行双工况优化：
    目标为最小化悬停功率 (point="hover", metric="power", sense="min")，
    同时施加悬停拉力下限、巡航拉力下限与巡航效率下限约束。
    """
    (
        num_blades,
        radius,
        hub_radius,
        n_radial,
        init_chord_curve,
        init_twist_curve,
        section_r_norm,
        init_section_kulfans,
        hover_cond_init,
        cruise_cond_init,
        nf_kwargs,
        common_geom_constraints,
    ) = build_baseline_1m_rotor_setup()

    mp_opt = fb.MultiPointBladeOptimizer(
        num_blades=num_blades,
        radius=radius,
        hub_radius=hub_radius,
        n_radial=n_radial,
        init_chord_curve=init_chord_curve,
        init_twist_curve=init_twist_curve,
        section_airfoils=init_section_kulfans,
        section_r_norm=section_r_norm,
        neuralfoil_kwargs=nf_kwargs,
    )

    init_geom = fb.RotorGeometry(
        num_blades, radius, hub_radius, n_radial,
        init_chord_curve, init_twist_curve, mp_opt.airfoils, section_r_norm
    )
    cfg_ax = fb.SolverConfig(n_azimuth=1, enable_tip_loss=True, enable_vrs_correction=False)
    init_hover_res = fb.BEMTSolver(init_geom, cfg_ax).solve(hover_cond_init)
    init_cruise_res = fb.BEMTSolver(init_geom, cfg_ax).solve(cruise_cond_init)
    init_eta_cruise = (init_cruise_res.thrust * cruise_cond_init.vz) / init_cruise_res.power

    hover_thrust_floor = max(init_hover_res.thrust * 1.03, 1950.0)
    cruise_eta_floor = max(init_eta_cruise + 0.015, 0.72)

    print("=" * 102)
    print("【案例 B】MultiPointBladeOptimizer (双工况): 目标 = 最小化悬停功率 (point='hover', metric='power', sense='min')")
    print(f"  -> [约束设置] T_hover >= {hover_thrust_floor:.1f} N, T_cruise >= {init_cruise_res.thrust:.1f} N, eta_cruise >= {cruise_eta_floor*100:.1f}%")

    mp_opt.add_operating_point(
        name="hover",
        cond=hover_cond_init,
        optimize_omega=True,
        omega_bounds=(160.0, 260.0),
    ).add_operating_point(
        name="cruise",
        cond=cruise_cond_init,
        optimize_omega=True,
        omega_bounds=(120.0, 220.0),
    )

    mp_opt.set_objective(point="hover", metric="power", sense="min")
    mp_opt.add_constraint(point="hover", metric="thrust", lower=hover_thrust_floor)
    mp_opt.add_constraint(point="cruise", metric="thrust", lower=init_cruise_res.thrust)
    mp_opt.add_constraint(point="cruise", metric="eta", lower=cruise_eta_floor)

    mp_res = mp_opt.optimize(**common_geom_constraints)

    h_m = mp_res.point_metrics["hover"]
    c_m = mp_res.point_metrics["cruise"]
    print(f"  -> [收敛状态] Success = {mp_res.success} (迭代 {mp_res.nit} 步)")
    print(f"  -> [悬停结果] T = {h_m['thrust']:7.2f} N (要求 >= {hover_thrust_floor:.1f} N), "
          f"P = {h_m['power']/1e3:6.3f} kW, FM = {h_m['fm']:.4f} (初始 {init_hover_res.fm:.4f}), RPM = {h_m['rpm']:.0f}")
    print(f"  -> [巡航结果] T = {c_m['thrust']:7.2f} N (要求 >= {init_cruise_res.thrust:.1f} N), "
          f"P = {c_m['power']/1e3:6.3f} kW, eta = {c_m['eta']*100:5.2f}% (要求 >= {cruise_eta_floor*100:.1f}%), RPM = {c_m['rpm']:.0f}")
    print("=" * 102)

    assert h_m["thrust"] >= hover_thrust_floor - 1.0
    assert c_m["thrust"] >= init_cruise_res.thrust - 1.0
    assert c_m["eta"] >= cruise_eta_floor - 2e-3
    assert h_m["fm"] > init_hover_res.fm

    if show_plot:
        _plot_optimization_summary(
            case_title="Case B: MultiPoint Minimize Hover Power (R=1.0m, 3 Sections)",
            init_chord_curve=init_chord_curve,
            opt_chord_curve=mp_res.chord_curve,
            init_twist_curve=init_twist_curve,
            opt_twist_curve=mp_res.twist_curve,
            section_r_norm=section_r_norm,
            init_kulfans=init_section_kulfans,
            opt_kulfans=mp_res.section_kulfan_airfoils,
            hover_power_hist=mp_res.history_point_metrics["hover"]["power"],
            cruise_power_hist=mp_res.history_point_metrics["cruise"]["power"],
            min_tip_chord_val=0.055,
        )

    return mp_res


# ==============================================================================
# 独立案例函数 3：【案例 C】悬停 + 45° 斜入流倾转过渡 + 巡航 三工况联合优化
# ==============================================================================
def run_case_c_three_point_oblique_transition(show_plot: bool = False) -> fb.MultiPointOptimizationResult:
    """
    【案例 C】使用 MultiPointBladeOptimizer 进行【悬停 + 45° 斜入流倾转过渡 + 巡航】三工况联合优化，
    将 45° 过渡态的俯仰力矩 |My| 削减至初始值的 85% 以内，同时最小化悬停+巡航加权总功率。
    """
    (
        num_blades,
        radius,
        hub_radius,
        n_radial,
        init_chord_curve,
        init_twist_curve,
        section_r_norm,
        init_section_kulfans,
        hover_cond_init,
        cruise_cond_init,
        nf_kwargs,
        common_geom_constraints,
    ) = build_baseline_1m_rotor_setup()

    mp3_opt = fb.MultiPointBladeOptimizer(
        num_blades=num_blades,
        radius=radius,
        hub_radius=hub_radius,
        n_radial=n_radial,
        init_chord_curve=init_chord_curve,
        init_twist_curve=init_twist_curve,
        section_airfoils=init_section_kulfans,
        section_r_norm=section_r_norm,
        neuralfoil_kwargs=nf_kwargs,
    )

    init_geom = fb.RotorGeometry(
        num_blades, radius, hub_radius, n_radial,
        init_chord_curve, init_twist_curve, mp3_opt.airfoils, section_r_norm
    )
    cfg_ax = fb.SolverConfig(n_azimuth=1, enable_tip_loss=True, enable_vrs_correction=False)
    init_hover_res = fb.BEMTSolver(init_geom, cfg_ax).solve(hover_cond_init)
    init_cruise_res = fb.BEMTSolver(init_geom, cfg_ax).solve(cruise_cond_init)

    trans_cond_init = fb.OperatingCondition.from_freestream(
        v_inf=15.0,
        alpha_disk_rad=np.deg2rad(45.0),
        omega_rad_s=200.0,
        pitch_0_rad=0.0,
    )
    trans_cfg = fb.SolverConfig(
        n_azimuth=36,
        wake_model=fb.WakeModel.GlauertPittPeters,
        enable_tip_loss=True,
        enable_vrs_correction=False,
        enable_stall_delay=True,
        enable_tangential_induction=True,
    )
    init_trans_res = fb.BEMTSolver(init_geom, trans_cfg).solve(trans_cond_init)

    my_init_abs = abs(init_trans_res.my)
    my_limit = 0.85 * my_init_abs
    trans_thrust_floor = 0.95 * init_trans_res.thrust

    print("=" * 102)
    print("【案例 C】MultiPointBladeOptimizer (三工况含 45° 斜流过渡): 目标 = 悬停+巡航加权功率最小化")
    print(f"  -> [过渡态约束] T_trans >= {trans_thrust_floor:.1f} N 且俯仰力矩 |My_trans| <= {my_limit:.2f} N·m (初始 {init_trans_res.my:+.2f} N·m)")

    mp3_opt.add_operating_point(
        name="hover",
        cond=hover_cond_init,
        optimize_omega=True,
        omega_bounds=(160.0, 260.0),
    ).add_operating_point(
        name="transition_45deg",
        cond=trans_cond_init,
        solver_cfg=trans_cfg,
        optimize_omega=True,
        omega_bounds=(150.0, 240.0),
    ).add_operating_point(
        name="cruise",
        cond=cruise_cond_init,
        optimize_omega=True,
        omega_bounds=(120.0, 220.0),
    )

    mp3_opt.add_objective_term(point="hover", metric="power", sense="min", weight=0.4)
    mp3_opt.add_objective_term(point="cruise", metric="power", sense="min", weight=0.6)

    mp3_opt.add_constraint(point="hover", metric="thrust", lower=init_hover_res.thrust)
    mp3_opt.add_constraint(point="hover", metric="fm", lower=init_hover_res.fm)
    mp3_opt.add_constraint(point="transition_45deg", metric="thrust", lower=trans_thrust_floor)
    mp3_opt.add_constraint(point="transition_45deg", metric="my", lower=-my_limit, upper=my_limit)
    mp3_opt.add_constraint(point="cruise", metric="thrust", lower=init_cruise_res.thrust)

    mp3_res = mp3_opt.optimize(**common_geom_constraints)

    h3 = mp3_res.point_metrics["hover"]
    t3 = mp3_res.point_metrics["transition_45deg"]
    c3 = mp3_res.point_metrics["cruise"]
    print(f"  -> [收敛状态] Success = {mp3_res.success} (迭代 {mp3_res.nit} 步)")
    print(f"  -> [悬停结果] T = {h3['thrust']:7.2f} N, P = {h3['power']/1e3:6.3f} kW, FM = {h3['fm']:.4f}, RPM = {h3['rpm']:.0f}")
    print(f"  -> [45°过渡态] T = {t3['thrust']:7.2f} N (要求 >= {trans_thrust_floor:.1f} N), "
          f"My = {t3['my']:+6.2f} N·m (约束区间 [{-my_limit:+.2f}, {+my_limit:+.2f}] N·m), Fx = {t3['fx']:5.2f} N, RPM = {t3['rpm']:.0f}")
    print(f"  -> [巡航结果] T = {c3['thrust']:7.2f} N, P = {c3['power']/1e3:6.3f} kW, eta = {c3['eta']*100:5.2f}%, RPM = {c3['rpm']:.0f}")
    print("=" * 102)

    assert h3["thrust"] >= init_hover_res.thrust - 1.0
    assert t3["thrust"] >= trans_thrust_floor - 1.0
    assert abs(t3["my"]) <= my_limit + 0.05
    assert c3["thrust"] >= init_cruise_res.thrust - 1.0

    if show_plot:
        _plot_optimization_summary(
            case_title="Case C: 3-Point Optimization with 45° Oblique Transition |My| Bound",
            init_chord_curve=init_chord_curve,
            opt_chord_curve=mp3_res.chord_curve,
            init_twist_curve=init_twist_curve,
            opt_twist_curve=mp3_res.twist_curve,
            section_r_norm=section_r_norm,
            init_kulfans=init_section_kulfans,
            opt_kulfans=mp3_res.section_kulfan_airfoils,
            hover_power_hist=mp3_res.history_point_metrics["hover"]["power"],
            cruise_power_hist=mp3_res.history_point_metrics["cruise"]["power"],
            min_tip_chord_val=0.055,
        )

    return mp3_res


# ==============================================================================
# 独立案例函数 4：【案例 D】R = 2.1 m 大直径三叶倾转旋翼 5 截面翼型 + 4.5km 巡航变总距优化
# ==============================================================================
def run_case_d_large_rotor_2p1m_five_section(
    show_plot: bool = False,
    v_cruise: float = 111.11,
    cruise_pressure_pa: float = 57752.6,
    cruise_rho: float = 0.777,
    hover_rho: float = 1.225,
    optimize_hover_pitch: bool = True,
    optimize_cruise_pitch: bool = True,
) -> fb.MultiPointOptimizationResult:
    """
    【案例 D】R = 2.1 m 大直径三叶倾转旋翼 5 截面翼型 CST + 弦长/扭转 B 样条 + 悬停/巡航总距角联合优化：
    - 旋翼半径 R = 2.10 m, 桨叶数 B = 3, 桨根半径 R_hub = 0.1 R = 0.21 m
    - 5 个控制截面翼型站位: r/R = [0.10, 0.20, 0.50, 0.75, 1.00]
    - 5 个截面翼型最大相对厚度下限: min_section_thickness = [0.28, 0.26, 0.18, 0.12, 0.08]
    - 桨根弦长约束: c(0.1R) == 0.318 m (min_root_chord = max_root_chord = 0.318 m)
    - 桨尖弦长约束: c(1.0R) >= 0.150 m (min_tip_chord = 0.150 m)
    - 悬停工况: Omega_hover = 110.0 rad/s (固定)，优化悬停总距角 pitch_0_hover，约束 T_hover >= 15000 N
    - 巡航工况: 海拔 4.5 km (p = 57752.6 Pa, rho = 0.777 kg/m^3 -> T = 258.93 K, a = 322.58 m/s, mu = 1.645e-5 Pa*s)，
      前飞速度 V_cruise = 111.11 m/s，Omega_cruise = 95.7 rad/s (固定)，优化巡航总距角 pitch_0_cruise，
      约束 T_cruise >= 850 N 且 P_cruise <= 110 kW (110000 W)
    - 优化目标: 最小化悬停功率 (min P_hover)
    """
    radius = 2.10
    num_blades = 3
    hub_radius = 0.10 * radius  # 0.21 m
    n_radial = 24

    ctrl_r = np.array([0.10, 0.25, 0.50, 0.75, 1.00], dtype=np.float64)
    init_chord_vals = np.array([0.318, 0.300, 0.265, 0.215, 0.155], dtype=np.float64)
    init_twist_vals = np.deg2rad([34.0, 26.0, 18.5, 11.5, 4.5])

    init_chord_curve = fb.BSplineCurve(ctrl_r, init_chord_vals, degree=3)
    init_twist_curve = fb.BSplineCurve(ctrl_r, init_twist_vals, degree=3)

    section_r_norm = np.array([0.10, 0.20, 0.50, 0.75, 1.00], dtype=np.float64)
    min_sec_thick = [0.28, 0.26, 0.18, 0.12, 0.08]
    init_section_kulfans = [
        fb.KulfanCSTAirfoil.from_naca4("2428"),  # r/R = 0.10: 初始 28% 厚度
        fb.KulfanCSTAirfoil.from_naca4("2426"),  # r/R = 0.20: 初始 26% 厚度
        fb.KulfanCSTAirfoil.from_naca4("2418"),  # r/R = 0.50: 初始 18% 厚度
        fb.KulfanCSTAirfoil.from_naca4("2412"),  # r/R = 0.75: 初始 12% 厚度
        fb.KulfanCSTAirfoil.from_naca4("2408"),  # r/R = 1.00: 初始  8% 厚度
    ]

    nf_kwargs = {
        "alpha_deg_grid": np.linspace(-16.0, 18.0, 69),
        "re_grid": np.array([2.0e5, 8.0e5, 2.5e6, 6.0e6]),
        "ma_grid": np.array([0.0, 0.25, 0.50, 0.75]),
        "model_size": "large",
    }

    t_cruise_k, a_sound_cruise, mu_cruise = compute_atmosphere_from_p_rho(cruise_pressure_pa, cruise_rho)

    hover_cond = fb.OperatingCondition(
        vx=0.0,
        vy=0.0,
        vz=0.0,
        omega=110.0,
        pitch_0=np.deg2rad(4.5),
        rho=hover_rho,
        mu=1.7894e-5,
        a_sound=340.29,
    )
    cruise_cond = fb.OperatingCondition(
        vx=0.0,
        vy=0.0,
        vz=v_cruise,
        omega=95.7,
        pitch_0=np.deg2rad(24.5),
        rho=cruise_rho,
        mu=mu_cruise,
        a_sound=a_sound_cruise,
    )

    mp_2p1 = fb.MultiPointBladeOptimizer(
        num_blades=num_blades,
        radius=radius,
        hub_radius=hub_radius,
        n_radial=n_radial,
        init_chord_curve=init_chord_curve,
        init_twist_curve=init_twist_curve,
        section_airfoils=init_section_kulfans,
        section_r_norm=section_r_norm,
        neuralfoil_kwargs=nf_kwargs,
    )

    init_geom = fb.RotorGeometry(
        num_blades, radius, hub_radius, n_radial,
        init_chord_curve, init_twist_curve, mp_2p1.airfoils, section_r_norm
    )
    cfg_ax = fb.SolverConfig(n_azimuth=1, enable_tip_loss=True, enable_vrs_correction=False)
    init_h = fb.BEMTSolver(init_geom, cfg_ax).solve(hover_cond)
    init_c = fb.BEMTSolver(init_geom, cfg_ax).solve(cruise_cond)

    mp_2p1.add_operating_point(
        name="hover",
        cond=hover_cond,
        solver_cfg=cfg_ax,
        optimize_omega=False,
        optimize_pitch_0=optimize_hover_pitch,
        pitch_0_deg_bounds=(-10.0, 20.0),
    ).add_operating_point(
        name="cruise",
        cond=cruise_cond,
        solver_cfg=cfg_ax,
        optimize_omega=False,
        optimize_pitch_0=optimize_cruise_pitch,
        pitch_0_deg_bounds=(18.0, 45.0),
    )

    mp_2p1.set_objective(point="hover", metric="power", sense="min")

    mp_2p1.add_constraint(point="hover", metric="thrust", lower=15000.0)
    mp_2p1.add_constraint(point="cruise", metric="thrust", lower=850.0)
    mp_2p1.add_constraint(point="cruise", metric="power", upper=110000.0)

    res_2p1 = mp_2p1.optimize(
        optimize_chord=True,
        chord_bounds=(0.150, 0.420),
        min_root_chord=0.318,                 # 锁定桨根弦长 c(0.1R) = 0.318 m
        max_root_chord=0.318,
        min_tip_chord=0.150,                  # 约束桨尖弦长 c(1.0R) >= 0.150 m
        min_tip_to_max_chord_ratio=0.42,
        chord_taper_from_index=1,             # 从第 1 控制点向桨尖单调收缩
        max_chord_step=0.090,
        optimize_twist=True,
        twist_deg_bounds=(0.0, 50.0),
        monotonic_twist=True,                 # 从根到尖单调递减洗出
        min_twist_step_deg=2.0,
        max_twist_step_deg=15.0,
        optimize_cst=True,
        cst_weight_delta_bound=0.04,
        min_thickness_ratio_scale=0.88,
        min_section_thickness=min_sec_thick,  # 5 个截面各自独立的最小相对厚度 [0.28, 0.26, 0.18, 0.12, 0.08]
        max_solidity=0.12,
        smoothness_weight=6e-3,
        cst_reg_weight=2e-2,
        maxiter=60,
        ftol=1e-4,
    )

    h_d = res_2p1.point_metrics["hover"]
    c_d = res_2p1.point_metrics["cruise"]
    print("=" * 102)
    print("【案例 D】R=2.1m 三叶倾转旋翼 5 截面翼型 + 变总距优化 (4.5km 巡航 Vz=111.11 m/s, p=57752.6 Pa, rho=0.777 kg/m^3):")
    print(f"  -> [高空大气参数] T_static = {t_cruise_k:.2f} K ({t_cruise_k-273.15:.2f} °C), "
          f"声速 a = {a_sound_cruise:.2f} m/s, 动力粘度 mu = {mu_cruise:.4e} Pa·s")
    print(f"  -> [初始状态基准] 悬停: T = {init_h.thrust:8.1f} N, P = {init_h.power/1e3:6.2f} kW, FM = {init_h.fm:.4f}, pitch_0 = +4.50° | "
          f"巡航: T = {init_c.thrust:7.1f} N, P = {init_c.power/1e3:6.2f} kW, pitch_0 = +24.50°")
    print(f"  -> [收敛状态] Success = {res_2p1.success} (迭代 {res_2p1.nit} 步)")
    print(f"  -> [优化悬停] T = {h_d['thrust']:8.1f} N (要求 >= 15000.0 N), P = {h_d['power']/1e3:6.2f} kW, "
          f"FM = {h_d['fm']:.4f}, Omega = {h_d['omega']:.1f} rad/s, 总距 pitch_0 = {h_d['pitch_0_deg']:+.2f}°")
    print(f"  -> [优化巡航] T = {c_d['thrust']:8.1f} N (要求 >= 850.0 N), P = {c_d['power']/1e3:6.2f} kW "
          f"(要求 <= 110.00 kW), eta = {c_d['eta']*100:5.2f}%, Omega = {c_d['omega']:.1f} rad/s, 总距 pitch_0 = {c_d['pitch_0_deg']:+.2f}°")
    print(f"  -> [弦长结果] 控制点 [m] = {np.round(res_2p1.chord_curve.ctrl_vals, 4)} "
          f"(c_root = {res_2p1.root_chord:.3f} m == 0.318 m, c_tip = {res_2p1.tip_chord:.3f} m >= 0.150 m)")
    print(f"  -> [扭转结果] 控制点 [deg] = {np.round(np.rad2deg(res_2p1.twist_curve.ctrl_vals), 2)}")

    x_chk = np.linspace(0.05, 0.95, 61)
    thick_summary = []
    for m, (kaf, t_req) in enumerate(zip(res_2p1.section_kulfan_airfoils, min_sec_thick)):
        t_max = float(np.max(kaf.thickness_at(x_chk)))
        thick_summary.append(f"Sec#{m+1}(r/R={section_r_norm[m]:.2f}): {t_max:.3f}>={t_req:.2f}")
        assert t_max >= t_req - 1e-3, f"截面 #{m+1} 最大厚度 {t_max:.4f} 低于下限 {t_req:.2f}"
    print("  -> [5截面厚度] " + " | ".join(thick_summary))
    print("=" * 102)

    assert np.isclose(res_2p1.root_chord, 0.318, atol=1e-4)
    assert res_2p1.tip_chord >= 0.150 - 1e-4
    assert h_d["thrust"] >= 15000.0 - 5.0
    assert c_d["thrust"] >= 850.0 - 2.0
    assert c_d["power"] <= 110000.0 + 50.0

    if show_plot:
        _plot_optimization_summary(
            case_title="Case D: R=2.1m 3-Blade Tiltrotor 5-Section Optimization (4.5km Cruise)",
            init_chord_curve=init_chord_curve,
            opt_chord_curve=res_2p1.chord_curve,
            init_twist_curve=init_twist_curve,
            opt_twist_curve=res_2p1.twist_curve,
            section_r_norm=section_r_norm,
            init_kulfans=init_section_kulfans,
            opt_kulfans=res_2p1.section_kulfan_airfoils,
            hover_power_hist=res_2p1.history_point_metrics["hover"]["power"],
            cruise_power_hist=res_2p1.history_point_metrics["cruise"]["power"],
            min_tip_chord_val=0.150,
        )

    return res_2p1


# 保留兼容旧接口的函数名
run_large_rotor_2p1m_five_section_case = run_case_d_large_rotor_2p1m_five_section


# ==============================================================================
# 独立案例函数 5：【案例 E】R = 2.1 m 变桨叶数 B = 3, 4, 5 理论与优化对比
# ==============================================================================
def _solve_single_blade_count_2p1m(
    num_blades: int,
    scale_chord_for_constant_solidity: bool,
    enforce_cruise_power_cap: bool,
    v_cruise: float = 111.11,
    cruise_pressure_pa: float = 57752.6,
    cruise_rho: float = 0.777,
    hover_rho: float = 1.225,
) -> fb.MultiPointOptimizationResult:
    """内部辅助函数：针对指定桨叶数 B 与弦长缩放策略运行 R = 2.1 m 5 截面优化"""
    radius = 2.10
    hub_radius = 0.10 * radius
    n_radial = 24

    scale = (3.0 / float(num_blades)) if scale_chord_for_constant_solidity else 1.0
    c_root_target = 0.318 * scale
    c_tip_min = 0.150 * scale

    ctrl_r = np.array([0.10, 0.25, 0.50, 0.75, 1.00], dtype=np.float64)
    init_chord_vals = np.array([0.318, 0.300, 0.265, 0.215, 0.155], dtype=np.float64) * scale
    init_twist_vals = np.deg2rad([34.0, 26.0, 18.5, 11.5, 4.5])

    init_chord_curve = fb.BSplineCurve(ctrl_r, init_chord_vals, degree=3)
    init_twist_curve = fb.BSplineCurve(ctrl_r, init_twist_vals, degree=3)

    section_r_norm = np.array([0.10, 0.20, 0.50, 0.75, 1.00], dtype=np.float64)
    min_sec_thick = [0.28, 0.26, 0.18, 0.12, 0.08]
    init_section_kulfans = [
        fb.KulfanCSTAirfoil.from_naca4("2428"),
        fb.KulfanCSTAirfoil.from_naca4("2426"),
        fb.KulfanCSTAirfoil.from_naca4("2418"),
        fb.KulfanCSTAirfoil.from_naca4("2412"),
        fb.KulfanCSTAirfoil.from_naca4("2408"),
    ]

    nf_kwargs = {
        "alpha_deg_grid": np.linspace(-16.0, 18.0, 69),
        "re_grid": np.array([1.5e5, 8.0e5, 2.5e6, 6.0e6]),
        "ma_grid": np.array([0.0, 0.25, 0.50, 0.75]),
        "model_size": "large",
    }

    _, a_sound_cruise, mu_cruise = compute_atmosphere_from_p_rho(cruise_pressure_pa, cruise_rho)

    hover_cond = fb.OperatingCondition(
        vx=0.0, vy=0.0, vz=0.0, omega=110.0, pitch_0=np.deg2rad(4.5),
        rho=hover_rho, mu=1.7894e-5, a_sound=340.29,
    )
    cruise_cond = fb.OperatingCondition(
        vx=0.0, vy=0.0, vz=v_cruise, omega=95.7, pitch_0=np.deg2rad(24.5),
        rho=cruise_rho, mu=mu_cruise, a_sound=a_sound_cruise,
    )

    cfg_ax = fb.SolverConfig(n_azimuth=1, enable_tip_loss=True, enable_vrs_correction=False)

    mp = fb.MultiPointBladeOptimizer(
        num_blades=num_blades,
        radius=radius,
        hub_radius=hub_radius,
        n_radial=n_radial,
        init_chord_curve=init_chord_curve,
        init_twist_curve=init_twist_curve,
        section_airfoils=init_section_kulfans,
        section_r_norm=section_r_norm,
        neuralfoil_kwargs=nf_kwargs,
    )
    mp.add_operating_point(
        name="hover",
        cond=hover_cond,
        solver_cfg=cfg_ax,
        optimize_omega=False,
        optimize_pitch_0=True,
        pitch_0_deg_bounds=(-10.0, 20.0),
    ).add_operating_point(
        name="cruise",
        cond=cruise_cond,
        solver_cfg=cfg_ax,
        optimize_omega=False,
        optimize_pitch_0=True,
        pitch_0_deg_bounds=(18.0, 45.0),
    )

    mp.add_objective_term(point="hover", metric="power", sense="min", weight=1.0)
    mp.add_constraint(point="hover", metric="thrust", lower=15000.0)
    if enforce_cruise_power_cap:
        mp.add_constraint(point="cruise", metric="thrust", lower=850.0)
        mp.add_constraint(point="cruise", metric="power", upper=110000.0)
    else:
        # 当总实度随 B 成倍增大时，为准确评估 T_cruise = 850 N 配平点的最低巡航功率，
        # 对巡航功率施加微弱次要权重使巡航总距角自动收敛于 T_cruise = 850 N 边界
        mp.add_objective_term(point="cruise", metric="power", sense="min", weight=1e-2)
        mp.add_constraint(point="cruise", metric="thrust", lower=850.0, upper=855.0)

    max_sol = 0.125 if scale_chord_for_constant_solidity else (0.125 * num_blades / 3.0)

    return mp.optimize(
        optimize_chord=True,
        chord_bounds=(c_tip_min, 0.420 * scale),
        min_root_chord=c_root_target,
        max_root_chord=c_root_target,
        min_tip_chord=c_tip_min,
        min_tip_to_max_chord_ratio=0.42,
        chord_taper_from_index=1,
        max_chord_step=0.090 * scale,
        optimize_twist=True,
        twist_deg_bounds=(0.0, 50.0),
        monotonic_twist=True,
        min_twist_step_deg=2.0,
        max_twist_step_deg=15.0,
        optimize_cst=True,
        cst_weight_delta_bound=0.04,
        min_thickness_ratio_scale=0.88,
        min_section_thickness=min_sec_thick,
        max_solidity=max_sol,
        smoothness_weight=6e-3,
        cst_reg_weight=2e-2,
        maxiter=80,
        ftol=5e-5,
    )


def run_case_e_blade_count_comparison(
    blade_counts: tuple[int, ...] = (3, 4, 5),
    mode: str = "both",
    show_plot: bool = False,
) -> dict[str, dict[int, fb.MultiPointOptimizationResult]]:
    """
    【案例 E】基于案例 D (R = 2.1 m, T_hover >= 15000 N, 4.5 km 巡航 Vz = 111.11 m/s)，
    定量对比桨叶数 B = 3, 4, 5 对悬停最高效率 FM、悬停轴功率以及高空巡航功率的影响：
    - mode="constant_solidity"：方案一（保持总实度 sigma 基本不变，单叶弦长约束按 3/B 缩放，严格满足巡航功率 <= 110 kW）
    - mode="fixed_chord"      ：方案二（保持单叶弦长约束 c_root = 0.318 m, c_tip >= 0.150 m 不变，总实度随 B 增加）
    - mode="both"             ：同时运行方案一与方案二并输出完整对比表
    """
    if mode not in ("constant_solidity", "fixed_chord", "both"):
        raise ValueError(f"Unsupported mode: {mode}. Choose from 'constant_solidity', 'fixed_chord', or 'both'.")

    results: dict[str, dict[int, fb.MultiPointOptimizationResult]] = {}
    theory_const_sol: dict[int, dict[str, float]] = {}

    print("=" * 108)
    print(f"【案例 E】R=2.1m 倾转旋翼变桨叶数对比分析 (B = {list(blade_counts)}, DL = 1082.7 N/m^2, T_hover = 15000 N)")
    print("-" * 108)
    print("  [1] 闭式理论上限公式预测 (保持总实度不变，计入 c ~ 1/B 引起的雷诺数摩擦阻力修正):")
    for b in blade_counts:
        th = estimate_theoretical_hover_fm_limit(
            thrust_n=15000.0,
            radius_m=2.10,
            hub_radius_m=0.21,
            num_blades=b,
            omega_rad_s=110.0,
            rho_kg_m3=1.225,
            ld_eff_ref=98.0,
            ref_blades=blade_counts[0],
            scale_chord_for_constant_solidity=True,
        )
        theory_const_sol[b] = th
        dfm_th = th["fm_max"] - theory_const_sol[blade_counts[0]]["fm_max"]
        dp_th = th["p_min_shaft_kw"] - theory_const_sol[blade_counts[0]]["p_min_shaft_kw"]
        print(
            f"      B = {b} 叶 | dk_tip = {th['dk_tip']:.4f}, kappa_eff = {th['kappa_eff']:.4f}, "
            f"(L/D)_eff = {th['ld_eff']:5.1f} -> 理论 FM_max = {th['fm_max']:.4f} ({dfm_th:+.4f}), "
            f"P_min = {th['p_min_shaft_kw']:6.2f} kW ({dp_th:+.2f} kW)"
        )

    if mode in ("constant_solidity", "both"):
        print("-" * 108)
        print("  [2] 方案一 (constant_solidity): 保持总实度 sigma 不变 (单叶弦长按 3/B 缩放，满足 P_cruise <= 110 kW):")
        res_const: dict[int, fb.MultiPointOptimizationResult] = {}
        for b in blade_counts:
            res_b = _solve_single_blade_count_2p1m(
                num_blades=b,
                scale_chord_for_constant_solidity=True,
                enforce_cruise_power_cap=True,
            )
            res_const[b] = res_b
            hb = res_b.point_metrics["hover"]
            cb = res_b.point_metrics["cruise"]
            h_ref = res_const[blade_counts[0]].point_metrics["hover"]
            dfm = hb["fm"] - h_ref["fm"]
            dp_kw = (hb["power"] - h_ref["power"]) / 1e3
            print(
                f"      B = {b} 叶 | sigma = {res_b.solidity:.4f}, c_root = {res_b.root_chord:.3f} m, c_tip = {res_b.tip_chord:.3f} m | "
                f"悬停 T = {hb['thrust']:7.1f} N, FM = {hb['fm']:.4f} ({dfm:+.4f}), P_hover = {hb['power']/1e3:6.2f} kW ({dp_kw:+.2f} kW) | "
                f"巡航 T = {cb['thrust']:5.1f} N, P_cruise = {cb['power']/1e3:6.2f} kW, eta = {cb['eta']*100:5.2f}%"
            )
            assert hb["thrust"] >= 15000.0 - 5.0
            assert cb["thrust"] >= 850.0 - 2.0
            assert cb["power"] <= 110000.0 + 250.0
        results["constant_solidity"] = res_const

        if len(blade_counts) >= 2:
            b0, b_last = blade_counts[0], blade_counts[-1]
            assert res_const[b_last].point_metrics["hover"]["fm"] > res_const[b0].point_metrics["hover"]["fm"]

    if mode in ("fixed_chord", "both"):
        print("-" * 108)
        print("  [3] 方案二 (fixed_chord): 保持单叶弦长边界不变 (c_root = 0.318 m, c_tip >= 0.150 m, 总实度随 B 增加):")
        res_fixed: dict[int, fb.MultiPointOptimizationResult] = {}
        for b in blade_counts:
            res_b = _solve_single_blade_count_2p1m(
                num_blades=b,
                scale_chord_for_constant_solidity=False,
                enforce_cruise_power_cap=(b == 3),
            )
            res_fixed[b] = res_b
            hb = res_b.point_metrics["hover"]
            cb = res_b.point_metrics["cruise"]
            h_ref = res_fixed[blade_counts[0]].point_metrics["hover"]
            dfm = hb["fm"] - h_ref["fm"]
            dp_kw = (hb["power"] - h_ref["power"]) / 1e3
            print(
                f"      B = {b} 叶 | sigma = {res_b.solidity:.4f}, c_root = {res_b.root_chord:.3f} m, c_tip = {res_b.tip_chord:.3f} m | "
                f"悬停 T = {hb['thrust']:7.1f} N, FM = {hb['fm']:.4f} ({dfm:+.4f}), P_hover = {hb['power']/1e3:6.2f} kW ({dp_kw:+.2f} kW) | "
                f"巡航 T = {cb['thrust']:5.1f} N, P_cruise = {cb['power']/1e3:6.2f} kW, eta = {cb['eta']*100:5.2f}%"
            )
            assert hb["thrust"] >= 15000.0 - 5.0
            assert cb["thrust"] >= 850.0 - 2.0
        results["fixed_chord"] = res_fixed

        if len(blade_counts) >= 2:
            b0, b_last = blade_counts[0], blade_counts[-1]
            assert res_fixed[b_last].point_metrics["hover"]["fm"] > res_fixed[b0].point_metrics["hover"]["fm"]

    print("=" * 108)

    if show_plot and "constant_solidity" in results:
        import matplotlib.pyplot as plt

        res_c = results["constant_solidity"]
        fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
        fig.suptitle("Case E: R=2.1m Blade Count Comparison (B = 3, 4, 5 at Constant Total Solidity)", fontsize=12, fontweight="bold")

        b_arr = np.array(blade_counts, dtype=int)
        fm_th_arr = [theory_const_sol[b]["fm_max"] for b in blade_counts]
        fm_opt_arr = [res_c[b].point_metrics["hover"]["fm"] for b in blade_counts]
        p_opt_arr = [res_c[b].point_metrics["hover"]["power"] / 1e3 for b in blade_counts]

        ax = axes[0]
        ax.plot(b_arr, fm_th_arr, "--o", color="tab:gray", lw=1.8, label="Theoretical FM_max(DL, B)")
        ax.plot(b_arr, fm_opt_arr, "-s", color="tab:blue", lw=2.2, label="fast_bemt 5-Sec Optimized FM")
        ax.set_xticks(b_arr)
        ax.set_xlabel("Number of Blades B [-]")
        ax.set_ylabel("Hover Figure of Merit FM [-]")
        ax.set_title("Hover FM vs. Blade Count B")
        ax.grid(True, alpha=0.3)
        ax.legend()

        ax = axes[1]
        ax.bar(b_arr, p_opt_arr, width=0.45, color="tab:orange", edgecolor="black", alpha=0.85)
        ax.set_xticks(b_arr)
        ax.set_xlabel("Number of Blades B [-]")
        ax.set_ylabel("Hover Shaft Power [kW]")
        ax.set_ylim(min(p_opt_arr) - 5.0, max(p_opt_arr) + 5.0)
        ax.set_title("Optimized Hover Power [kW]")
        ax.grid(True, alpha=0.3)

        ax = axes[2]
        for b in blade_counts:
            r_s, c_s = res_c[b].chord_curve.sample(200)
            ax.plot(r_s, c_s, "-", lw=2.0, label=f"B = {b} (sigma={res_c[b].solidity:.3f})")
        ax.set_xlabel("r/R [-]")
        ax.set_ylabel("Single-Blade Chord [m]")
        ax.set_title("Optimized Chord Distribution (Constant Solidity)")
        ax.grid(True, alpha=0.3)
        ax.legend()

        plt.tight_layout()
        plt.show()

    return results


def run_hover_cruise_joint_optimization_case(show_plot: bool = False):
    """一次性顺序运行全部 5 个案例"""
    res_a = run_case_a_hover_cruise_min_cruise_power(show_plot=False)
    res_b = run_case_b_multipoint_min_hover_power(show_plot=False)
    res_c = run_case_c_three_point_oblique_transition(show_plot=False)
    res_d = run_case_d_large_rotor_2p1m_five_section(show_plot=show_plot)
    res_e = run_case_e_blade_count_comparison(show_plot=show_plot)
    return res_a, res_b, res_c, res_d, res_e


if __name__ == "__main__":
    enable_plot = "--plot" in sys.argv

    # ==========================================================================
    # 在下方直接注释掉不需要运行的案例函数即可单独执行目标案例：
    # ==========================================================================
    # run_case_a_hover_cruise_min_cruise_power(show_plot=False)
    # run_case_b_multipoint_min_hover_power(show_plot=False)
    # run_case_c_three_point_oblique_transition(show_plot=False)
    # run_case_d_large_rotor_2p1m_five_section(show_plot=enable_plot)
    run_case_e_blade_count_comparison(blade_counts=(3, 4, 5), mode="both", show_plot=enable_plot)
