import os
import sys
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "python")))

import fast_bemt as fb


def create_ideal_linear_airfoil(cl_alpha: float = 2.0 * np.pi, cd0: float = 0.01) -> fb.Airfoil3D:
    """构造无失速的理想线性翼型: Cl(alpha) = cl_alpha * alpha, Cd(alpha) = cd0"""
    alpha_grid = np.deg2rad(np.linspace(-45.0, 45.0, 91))
    cl_1d = cl_alpha * alpha_grid
    cd_1d = np.full_like(alpha_grid, cd0)
    return fb.Airfoil3D.from_1d_polar(
        alpha_grid=alpha_grid,
        cl_1d=cl_1d,
        cd_1d=cd_1d,
        ref_re=1.0e6,
        aspect_ratio=10.0,
        enable_extrapolation=False,
    )


def build_fitted_bspline_curve(
    r: np.ndarray,
    target_vals: np.ndarray,
    hub_radius: float,
    radius: float,
    n_ctrl: int = 18,
    degree: int = 3,
) -> fb.BSplineCurve:
    """利用 B 样条基函数矩阵最小二乘反求控制点，并构造独立的 fb.BSplineCurve 实例"""
    x_hub = hub_radius / radius
    x_ctrl = fb.build_greville_abscissae(n_ctrl, degree, x_hub, 1.0)
    basis = fb.build_parametric_bspline_basis_matrix(r / radius, x_ctrl, degree)
    ctrl_vals, _, _, _ = np.linalg.lstsq(basis, target_vals, rcond=None)
    return fb.BSplineCurve(x_ctrl, ctrl_vals, degree=degree)


def leishman_small_angle_hyperbolic(
    sigma_0: float, cl_alpha: float, cd0: float, theta_tip_rad: float, x_hub: float
):
    sc = sigma_0 * cl_alpha
    lambda_i = (sc / 16.0) * (np.sqrt(1.0 + (32.0 * theta_tip_rad) / sc) - 1.0)
    ct = 2.0 * (lambda_i**2) * (1.0 - x_hub**2)
    cq_induced = lambda_i * ct
    cq_profile = (1.0 / 8.0) * sigma_0 * cd0 * (1.0 - x_hub**4)
    cq = cq_induced + cq_profile
    fm = (ct**1.5) / (np.sqrt(2.0) * cq) if cq > 0 else 0.0
    return ct, cq, fm, lambda_i


def leishman_small_angle_station_dist(
    x: np.ndarray, sigma_0: float, cl_alpha: float, cd0: float, theta_x: np.ndarray
):
    sc_0 = sigma_0 * cl_alpha
    lam_x = (sc_0 / 16.0) * (np.sqrt(1.0 + (32.0 / sc_0) * theta_x * x) - 1.0)
    dct_dx = 4.0 * (lam_x**2) * x
    dcq_dx = 4.0 * (lam_x**3) * x + 0.5 * sigma_0 * cd0 * (x**3)
    return lam_x, dct_dx, dcq_dx


def run_hover_analytical_validation():
    print("=" * 86)
    print("FAST_BEMT 物理正确性验证 (Step 1): 悬停经典解析解对标 (Leishman Analytical Benchmark)")
    print("=" * 86)

    num_blades = 2
    radius = 1.0
    hub_radius = 0.10
    x_hub = hub_radius / radius
    chord_const = 0.08
    sigma_0 = (num_blades * chord_const) / (np.pi * radius)

    cl_alpha = 2.0 * np.pi
    cd0 = 0.010
    omega = 150.0
    rho = 1.225
    v_tip = omega * radius
    area = np.pi * radius**2
    f_ref = rho * area * (v_tip**2)

    cfg_ideal = fb.SolverConfig(
        n_azimuth=1,
        wake_model=fb.WakeModel.GlauertPittPeters,
        enable_tip_loss=False,
        enable_vrs_correction=False,
        enable_stall_delay=False,
        enable_tangential_induction=False,
        tol=1e-12,
    )

    n_r = 80
    edges = np.linspace(hub_radius, radius, n_r + 1)
    r = 0.5 * (edges[:-1] + edges[1:])
    dr = np.diff(edges)
    x = r / radius
    section_r_norm = np.array([0.10, 0.50, 1.00])
    section_airfoils = [create_ideal_linear_airfoil(cl_alpha, cd0) for _ in range(3)]

    # 独立实例化等弦长 BSplineCurve 对象
    chord_curve_const = fb.BSplineCurve.from_1d_values(
        np.full(6, chord_const), degree=3, r_norm_min=x_hub, r_norm_max=1.0
    )

    # =========================================================================
    # 测试 1：理想双曲扭转旋翼 (Ideal Hyperbolic Twist: theta(x) = theta_tip / x)
    # =========================================================================
    print("\n[验证 1] 理想双曲扭转悬停解析解对标 (theta(r) = theta_tip / (r/R), N_r = 80, BSplineCurve N_ctrl = 18):")
    print("  " + "-" * 80)
    print("  叶尖桨距 |   C_T (BEMT vs 解析)   相对误差 |   C_Q (BEMT vs 解析)   相对误差 | 诱导速度均匀性变异系数")
    print("  " + "-" * 80)

    for theta_tip_deg in [4.0, 6.0, 8.0, 10.0]:
        theta_tip = np.deg2rad(theta_tip_deg)
        twist_curve_hyp = build_fitted_bspline_curve(r, theta_tip / x, hub_radius, radius, n_ctrl=18, degree=3)

        geom = fb.RotorGeometry(
            num_blades, radius, hub_radius, n_r, chord_curve_const, twist_curve_hyp, section_airfoils, section_r_norm, 1
        )
        solver = fb.BEMTSolver(geom, cfg_ideal)
        cond = fb.OperatingCondition(vx=0.0, vy=0.0, vz=0.0, omega=omega, pitch_0=0.0, rho=rho)
        res = solver.solve(cond, compute_gradients=False)

        ct_ana, cq_ana, fm_ana, lam_ana = leishman_small_angle_hyperbolic(
            sigma_0, cl_alpha, cd0, theta_tip, x_hub
        )

        err_ct = abs(res.ct - ct_ana) / ct_ana * 100.0
        err_cq = abs(res.cq - cq_ana) / cq_ana * 100.0
        vi_cov = np.std(res.vi_annulus) / np.mean(res.vi_annulus) * 100.0

        print(
            f"   {theta_tip_deg:5.1f}°  | {res.ct:.6f} vs {ct_ana:.6f} ({err_ct:5.2f}%) | "
            f"{res.cq:.7f} vs {cq_ana:.7f} ({err_cq:5.2f}%) |   CoV = {vi_cov:.4f}%"
        )

    theta_tip_6 = np.deg2rad(6.0)
    twist_curve_6 = build_fitted_bspline_curve(r, theta_tip_6 / x, hub_radius, radius, n_ctrl=18, degree=3)
    geom_6 = fb.RotorGeometry(
        num_blades, radius, hub_radius, n_r, chord_curve_const, twist_curve_6, section_airfoils, section_r_norm, 1
    )
    res_6 = fb.BEMTSolver(geom_6, cfg_ideal).solve(
        fb.OperatingCondition(vx=0.0, vy=0.0, vz=0.0, omega=omega, pitch_0=0.0, rho=rho)
    )
    lam_x, dct_dx_ana, dcq_dx_ana = leishman_small_angle_station_dist(x, sigma_0, cl_alpha, cd0, theta_tip_6 / x)
    dT_dr_ana = dct_dx_ana * f_ref / radius
    dQ_dr_ana = dcq_dx_ana * (f_ref * radius) / radius

    l2_err_dT = np.linalg.norm(res_6.dT_dr_mean - dT_dr_ana) / np.linalg.norm(dT_dr_ana) * 100.0
    l2_err_dQ = np.linalg.norm(res_6.dQ_dr_mean - dQ_dr_ana) / np.linalg.norm(dQ_dr_ana) * 100.0
    print(f"  -> [展向载荷分布校验 @ 6.0°] dT/dr 全展向 L2 相对偏差: {l2_err_dT:.3f}% | dQ/dr 全展向 L2 相对偏差: {l2_err_dQ:.3f}%")

    # =========================================================================
    # 测试 2：无扭转矩形桨叶 (Untwisted Rectangular Blade) 总距扫掠对标
    # =========================================================================
    print("\n[验证 2] 无扭转矩形桨叶 (twist_curve = 0) 悬停总距扫掠对标:")
    print("  " + "-" * 80)
    print("  总距角 pitch_0 |   C_T (BEMT vs 解析)   相对误差 |   C_Q (BEMT vs 解析)   相对误差 | 品质因数 FM (BEMT vs 解析)")
    print("  " + "-" * 80)

    twist_curve_zero = fb.BSplineCurve.from_1d_values(np.zeros(6), degree=3, r_norm_min=x_hub, r_norm_max=1.0)
    geom_rect = fb.RotorGeometry(
        num_blades, radius, hub_radius, n_r, chord_curve_const, twist_curve_zero, section_airfoils, section_r_norm, 1
    )
    solver_rect = fb.BEMTSolver(geom_rect, cfg_ideal)

    for p0_deg in [2.0, 4.0, 6.0, 8.0, 10.0]:
        p0_rad = np.deg2rad(p0_deg)
        cond = fb.OperatingCondition(vx=0.0, vy=0.0, vz=0.0, omega=omega, pitch_0=p0_rad, rho=rho)
        res = solver_rect.solve(cond, compute_gradients=False)

        _, dct_dx_rect, dcq_dx_rect = leishman_small_angle_station_dist(
            x, sigma_0, cl_alpha, cd0, np.full_like(x, p0_rad)
        )
        ct_rect_ana = float(np.sum(dct_dx_rect * (dr / radius)))
        cq_rect_ana = float(np.sum(dcq_dx_rect * (dr / radius)))
        fm_rect_ana = (ct_rect_ana**1.5) / (np.sqrt(2.0) * cq_rect_ana)

        err_ct = abs(res.ct - ct_rect_ana) / ct_rect_ana * 100.0
        err_cq = abs(res.cq - cq_rect_ana) / cq_rect_ana * 100.0
        print(
            f"      {p0_deg:5.1f}°     | {res.ct:.6f} vs {ct_rect_ana:.6f} ({err_ct:5.2f}%) | "
            f"{res.cq:.7f} vs {cq_rect_ana:.7f} ({err_cq:5.2f}%) |    {res.fm:.4f} vs {fm_rect_ana:.4f}"
        )

    # =========================================================================
    # 测试 3：精确非线性 BEMT 代数方程零误差验证 & 径向网格二阶收敛率验证
    # =========================================================================
    print("\n[验证 3] 精确大角度单微元代数方程残差校验 & 径向积分网格收敛阶 (Grid Convergence):")
    phi_v = res_6.phi[:, 0]
    vi_v = res_6.vi_annulus
    alpha_v = geom_6.twist - phi_v
    cl_v = cl_alpha * alpha_v
    cd_v = np.full_like(phi_v, cd0)
    cn_v = cl_v * np.cos(phi_v) - cd_v * np.sin(phi_v)
    w_sq_v = (omega * r) ** 2 + vi_v**2

    dT_dr_exact_be = 0.5 * rho * w_sq_v * geom_6.chord * cn_v * num_blades
    max_station_err = np.max(np.abs(res_6.dT_dr_mean - dT_dr_exact_be))
    print(f"  - 单站位叶素载荷公式与 C++ 内部输出最大绝对偏差: {max_station_err:.3e} N/m")

    def solve_rect_nr(n_stations: int):
        g_s = fb.RotorGeometry(
            num_blades, radius, hub_radius, n_stations,
            chord_curve_const, twist_curve_zero, section_airfoils, section_r_norm, 1
        )
        c_s = fb.OperatingCondition(vx=0.0, vy=0.0, vz=0.0, omega=omega, pitch_0=np.deg2rad(8.0), rho=rho)
        return fb.BEMTSolver(g_s, cfg_ideal).solve(c_s, False).thrust

    t_ref = solve_rect_nr(640)
    grid_list = [10, 20, 40, 80, 160]
    errs = []
    for n_s in grid_list:
        t_val = solve_rect_nr(n_s)
        errs.append(abs(t_val - t_ref) / t_ref)

    print("  - 径向网格加密收敛性 (对比 N_r = 640 基准解):")
    for idx, n_s in enumerate(grid_list):
        order_str = "---"
        if idx > 0:
            order = np.log2(errs[idx - 1] / errs[idx])
            order_str = f"{order:.2f} 阶 (理论二阶 2.00)"
        print(f"    N_r = {n_s:3d} | 拉力相对离散误差 = {errs[idx]*100:.5f}% | 收敛阶 = {order_str}")
    print("=" * 86)


if __name__ == "__main__":
    run_hover_analytical_validation()
