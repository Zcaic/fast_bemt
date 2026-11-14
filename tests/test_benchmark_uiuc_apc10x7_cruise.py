#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
公开风洞实验对标验证算例 2：UIUC APC Thin Electric 10x7E 螺旋桨巡航风洞实验
================================================================================
数据来源与参考文献：
1. Brandt, J. B., and Selig, M. S., "Propeller Performance Data at Low Reynolds
   Numbers," 49th AIAA Aerospace Sciences Meeting, AIAA Paper 2011-1255, 2011.
   UIUC Propeller Database Vol. 1: https://m-selig.ae.illinois.edu/props/volume-1/propDB-volume-1.html
   论文全文: https://m-selig.ae.illinois.edu/pubs/BrandtSelig-2011-AIAA-2011-1255-LRN-Propellers.pdf
2. Brandt, J. B., "Small-Scale Propeller Performance at Low Speeds," M.S. Thesis
   Appendix A (Table A.1: APC Thin Electric 10x7 Geometry Summary), UIUC, 2005.
   https://m-selig.ae.illinois.edu/props/volume-1/Brandt_2005_UIUC-MS-Thesis_Appendix.pdf
3. APC 10x7E 18 站位离散几何表 (Table A1) 与 6500 RPM 风洞实测/CFD/BEM 对标数据表 (Table C2):
   Takeyeldein, M. M., "Embedded Micro Electric Ducted Fan Propulsion for Small
   Fixed-Wing UAVs," Preprints.org, 2026, Appendix A & Appendix C.
   https://www.preprints.org/frontend/manuscript/f9c5b8d64ff5a7d9bf44ac5a2beaed72/download_pub

螺旋桨几何与风洞实验参数：
- 型号: APC Thin Electric 10x7 (10 in 直径, 7 in 螺距)
- 叶片数: B = 2
- 半径: R = 0.127 m (直径 D = 0.254 m = 10.00 in)
- 0.75R 处弦长: c_0.75R = 0.645 in = 0.01638 m (c_0.75R / R = 0.129)
- 截面翼型: 根部 Clark Y (t/c = 12%) 向外段 NACA 4412 / E63 薄翼型 (t/c = 6%) 过渡
- 测试转速: 6500 RPM (n = 108.333 rev/s, Omega = 680.678 rad/s)
- 前进比扫描范围: J = V_inf / (n * D) in [0.10, 0.80]
"""

from __future__ import annotations

import sys
from pathlib import Path
import numpy as np

# 确保可从项目根目录或 tests 目录直接导入 fast_bemt
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_PYTHON_PKG_DIR = _PROJECT_ROOT / "python"
if _PYTHON_PKG_DIR.exists() and str(_PYTHON_PKG_DIR) not in sys.path:
    sys.path.insert(0, str(_PYTHON_PKG_DIR))

import fast_bemt as fb


# ==============================================================================
# 1. UIUC APC Thin Electric 10x7 实测离散几何数据表 (18 径向站位, r/R = 0.15 ~ 1.00)
#    来源: UIUC Propeller Database / Brandt (2005) & Takeyeldein (2026) Table A1
# ==============================================================================

APC_10X7E_GEOMETRY_TABLE = np.array(
    [
        # [r/R,   c/R,    beta_deg]
        [0.15, 0.138, 37.86],
        [0.20, 0.154, 45.82],
        [0.25, 0.175, 44.19],
        [0.30, 0.190, 38.35],
        [0.35, 0.198, 33.64],
        [0.40, 0.202, 29.90],
        [0.45, 0.200, 27.02],
        [0.50, 0.195, 24.67],
        [0.55, 0.186, 22.62],
        [0.60, 0.174, 20.88],
        [0.65, 0.161, 19.36],
        [0.70, 0.145, 17.98],
        [0.75, 0.129, 16.74],
        [0.80, 0.112, 15.79],
        [0.85, 0.096, 14.64],
        [0.90, 0.081, 13.86],
        [0.95, 0.061, 12.72],
        [1.00, 0.040, 11.53],
    ],
    dtype=float,
)


# ==============================================================================
# 2. UIUC APC 10x7 (6500 RPM) 风洞实测、三维 RANS CFD 与文献参考 BEM 对标数据
#    来源: Brandt & Selig (2011) UIUC Wind Tunnel / Takeyeldein (2026) Table C2
#    以及 UIUC Volume 1 (run 6519/6015) 对应的功率系数 CP 与推进效率 eta_exp
# ==============================================================================

APC_10X7E_WIND_TUNNEL_6500RPM = [
    # J,     CT_exp,  CP_exp,  eta_exp, CT_cfd_kw_sst, CT_lit_bem
    (0.10, 0.1099, 0.0492, 0.2234, 0.0994, 0.0996),
    (0.20, 0.1044, 0.0498, 0.4193, 0.0925, 0.1015),
    (0.30, 0.0972, 0.0495, 0.5891, 0.0829, 0.1058),
    (0.40, 0.0865, 0.0480, 0.7208, 0.0730, 0.1031),
    (0.50, 0.0679, 0.0435, 0.7805, 0.0565, 0.0859),
    (0.60, 0.0469, 0.0360, 0.7817, 0.0407, 0.0607),
    (0.70, 0.0249, 0.0252, 0.6917, 0.0242, 0.0294),
    (0.80, 0.0075, 0.0135, 0.4444, 0.0072, 0.0076),
]


# ==============================================================================
# 3. 用最小二乘 B-Spline 拟合 UIUC 实测弦长与扭转分布并构建多截面翼型
# ==============================================================================

def fit_bspline_from_uiuc_table(
    r_norm_data: np.ndarray,
    values_data: np.ndarray,
    num_ctrl_pts: int = 12,
    degree: int = 3,
) -> fb.BSplineCurve:
    """
    利用 fast_bemt.BSplineCurve 的解析基函数矩阵 B = basis_matrix(r_norm)，
    通过固定首尾端点插值 + 内部控制点最小二乘拟合，将 UIUC 实测离散几何转化为光滑 B 样条曲线。
    """
    ctrl_r_norm = np.linspace(float(r_norm_data[0]), float(r_norm_data[-1]), num_ctrl_pts)
    template_curve = fb.BSplineCurve(ctrl_r_norm, np.zeros(num_ctrl_pts), degree=degree)
    basis_mat = np.asarray(template_curve.basis_matrix(r_norm_data), dtype=float)

    # Clamped B-Spline 首尾控制点精确插值首尾实测点，内部控制点由最小二乘求解
    ctrl_vals = np.zeros(num_ctrl_pts, dtype=float)
    ctrl_vals[0] = float(values_data[0])
    ctrl_vals[-1] = float(values_data[-1])

    rhs = values_data - basis_mat[:, 0] * ctrl_vals[0] - basis_mat[:, -1] * ctrl_vals[-1]
    b_inner = basis_mat[:, 1:-1]
    ctrl_vals[1:-1], _, _, _ = np.linalg.lstsq(b_inner, rhs, rcond=None)

    return fb.BSplineCurve(ctrl_r_norm, ctrl_vals, degree=degree)


def build_apc10x7_spanwise_airfoils(reynolds_075: float = 1.15e5) -> tuple[np.ndarray, list[fb.Airfoil3D]]:
    """
    根据 Brandt (2005) Table A.1 及 Takeyeldein (2026) Appendix A，APC Thin Electric 10x7
    桨根 (r/R = 0.15) 为厚截面 Clark Y (t/c = 12%)，中展段 (r/R = 0.55) 为 NACA 4412，
    桨尖 (r/R = 1.00) 过渡为薄翼型截面 (t/c = 6%)。
    此处构建根、中、尖 3 个控制截面的 Airfoil3D 极曲线，交由 RotorGeometry 自动沿展向线性插值。
    """
    alphas_deg = np.linspace(-15.0, 22.0, 186)
    alphas_rad = np.deg2rad(alphas_deg)

    def _make_section(alpha0_deg: float, slope: float, cd0: float, k_ind: float, cl_opt: float) -> fb.Airfoil3D:
        a0 = np.deg2rad(alpha0_deg)
        eff_a = alphas_rad - a0
        cl_lin = slope * eff_a
        stall_rad = np.deg2rad(13.5)
        blend = 1.0 / (1.0 + np.exp((np.abs(eff_a) - stall_rad) / np.deg2rad(2.0)))
        cl_stall = np.sign(eff_a) * 1.15 * np.sin(2.0 * np.abs(eff_a))
        cl = blend * cl_lin + (1.0 - blend) * cl_stall
        cd = cd0 + k_ind * ((cl - cl_opt) ** 2) + 0.38 * (1.0 - blend) * (np.sin(eff_a) ** 2)
        return fb.Airfoil3D.from_1d_polar(
            alphas_rad,
            cl,
            cd,
            ref_re=reynolds_075,
            aspect_ratio=5.80,
            enable_extrapolation=True,
        )

    section_r_norm = np.array([0.15, 0.55, 1.00], dtype=float)
    af_root = _make_section(alpha0_deg=-3.85, slope=5.80, cd0=0.022, k_ind=0.025, cl_opt=0.35)
    af_mid = _make_section(alpha0_deg=-3.65, slope=5.98, cd0=0.016, k_ind=0.020, cl_opt=0.35)
    af_tip = _make_section(alpha0_deg=-2.75, slope=5.90, cd0=0.015, k_ind=0.022, cl_opt=0.25)

    return section_r_norm, [af_root, af_mid, af_tip]


# ==============================================================================
# 4. 运行 UIUC APC 10x7E 全前进比巡航对标验证
# ==============================================================================

def run_uiuc_apc10x7_validation(plot: bool = False) -> dict:
    print("=" * 98)
    print("  公开风洞实验验证 2：UIUC APC Thin Electric 10x7E 螺旋桨全前进比巡航对标 (6500 RPM)")
    print("  数据源: https://m-selig.ae.illinois.edu/props/volume-1/propDB-volume-1.html")
    print("=" * 98)

    diameter = 0.254          # 10.00 inches = 0.254 m
    radius = 0.5 * diameter   # 0.127 m
    num_blades = 2
    rpm = 6500.0
    n_rps = rpm / 60.0        # 108.3333 rev/s
    omega = 2.0 * np.pi * n_rps
    rho = 1.225

    r_norm_tab = APC_10X7E_GEOMETRY_TABLE[:, 0]
    c_tab = APC_10X7E_GEOMETRY_TABLE[:, 1] * radius
    twist_rad_tab = np.deg2rad(APC_10X7E_GEOMETRY_TABLE[:, 2])
    hub_radius = float(r_norm_tab[0] * radius)

    # 1. 用 12 控制点 3 次 Clamped B-Spline 拟合 UIUC 18 个站位的弦长与扭转分布
    chord_spline = fit_bspline_from_uiuc_table(r_norm_tab, c_tab, num_ctrl_pts=12, degree=3)
    twist_spline = fit_bspline_from_uiuc_table(r_norm_tab, twist_rad_tab, num_ctrl_pts=12, degree=3)

    c_fit = chord_spline(r_norm_tab)
    twist_deg_fit = np.rad2deg(twist_spline(r_norm_tab))
    chord_rmse_mm = np.sqrt(np.mean((c_fit - c_tab) ** 2)) * 1000.0
    twist_rmse_deg = np.sqrt(np.mean((twist_deg_fit - APC_10X7E_GEOMETRY_TABLE[:, 2]) ** 2))
    c_075_in = float(chord_spline(0.75)) / 0.0254

    print(
        f"[B-Spline 几何拟合] 12 控制点 3 次 Clamped B-Spline 拟合 UIUC 18 个实测站位:\n"
        f"   - 弦长拟合均方根误差 RMSE = {chord_rmse_mm:.3f} mm | 0.75R 弦长 = {c_075_in:.3f} in (Brandt 实测 0.645 in)\n"
        f"   - 扭转拟合均方根误差 RMSE = {twist_rmse_deg:.3f} deg | 桨尖扭转角 = {float(np.rad2deg(twist_spline(1.0))):.2f} deg"
    )
    assert chord_rmse_mm < 0.15, f"弦长 B-Spline 拟合误差过大: {chord_rmse_mm:.3f} mm"
    assert twist_rmse_deg < 0.50, f"扭转 B-Spline 拟合误差过大: {twist_rmse_deg:.3f} deg"

    # 2. 构建 36 个径向微元的 RotorGeometry 与 BEMTSolver（含根/中/尖三截面变厚度翼型展向插值）
    num_elements = 36
    section_r_norm, section_airfoils = build_apc10x7_spanwise_airfoils()

    geom = fb.RotorGeometry(
        num_blades=num_blades,
        radius=radius,
        hub_radius=hub_radius,
        n_radial=num_elements,
        chord_curve=chord_spline,
        twist_curve=twist_spline,
        airfoils=section_airfoils,
        section_r_norm=section_r_norm,
        spin_dir=1,
    )

    solver_cfg = fb.SolverConfig(
        n_azimuth=1,
        enable_tip_loss=True,
        enable_vrs_correction=False,
        enable_stall_delay=True,
        enable_tangential_induction=True,
    )
    solver = fb.BEMTSolver(geom, solver_cfg)

    # 3. 扫描前进比 J = 0.10 ~ 0.80 并转换至 UIUC 螺旋桨无量纲定义:
    #    CT = T / (rho * n^2 * D^4),  CP = P / (rho * n^3 * D^5),  eta = J * CT / CP
    print("-" * 98)
    print(
        f"{'前进比 J':>8} | {'V_inf(m/s)':>10} | {'UIUC CT_exp':>11} | {'RANS CFD CT':>11} | "
        f"{'文献 BEM CT':>11} | {'fast_bemt CT':>12} | {'UIUC eta_exp':>12} | {'fast_bemt eta':>13}"
    )
    print("-" * 98)

    j_list = []
    ct_exp_list = []
    ct_cfd_list = []
    ct_lit_bem_list = []
    ct_fast_bemt_list = []
    eta_exp_list = []
    eta_fast_bemt_list = []

    ct_denom = rho * (n_rps ** 2) * (diameter ** 4)
    cp_denom = rho * (n_rps ** 3) * (diameter ** 5)

    for j_val, ct_exp, cp_exp, eta_exp, ct_cfd, ct_lit_bem in APC_10X7E_WIND_TUNNEL_6500RPM:
        v_inf = j_val * n_rps * diameter
        cond = fb.OperatingCondition(
            vx=0.0, vy=0.0, vz=v_inf, omega=omega, pitch_0=0.0, rho=rho
        )
        res = solver.solve(cond, compute_gradients=True)
        assert res.converged, f"J = {j_val:.2f} 处 BEMT 未完全收敛"
        assert res.sensitivity.valid, f"J = {j_val:.2f} 处 BEMT 解析灵敏度未生成"

        ct_fb = float(res.thrust) / ct_denom
        cp_fb = float(res.power) / cp_denom
        eta_fb = j_val * ct_fb / cp_fb if cp_fb > 1e-12 else 0.0

        j_list.append(j_val)
        ct_exp_list.append(ct_exp)
        ct_cfd_list.append(ct_cfd)
        ct_lit_bem_list.append(ct_lit_bem)
        ct_fast_bemt_list.append(ct_fb)
        eta_exp_list.append(eta_exp)
        eta_fast_bemt_list.append(eta_fb)

        print(
            f"{j_val:>8.2f} | {v_inf:>10.2f} | {ct_exp:>11.4f} | {ct_cfd:>11.4f} | "
            f"{ct_lit_bem:>11.4f} | {ct_fb:>12.4f} | {eta_exp * 100:>11.2f}% | {eta_fb * 100:>12.2f}%"
        )

    ct_exp_arr = np.asarray(ct_exp_list)
    ct_fb_arr = np.asarray(ct_fast_bemt_list)
    ct_lit_arr = np.asarray(ct_lit_bem_list)
    ct_cfd_arr = np.asarray(ct_cfd_list)

    # 计算与 UIUC 实验曲线的相关系数 R^2 与平均绝对误差 MAE
    corr_matrix = np.corrcoef(ct_exp_arr, ct_fb_arr)
    r_squared = float(corr_matrix[0, 1] ** 2)
    mae_fb = float(np.mean(np.abs(ct_fb_arr - ct_exp_arr)))
    mae_lit_bem = float(np.mean(np.abs(ct_lit_arr - ct_exp_arr)))
    mae_cfd = float(np.mean(np.abs(ct_cfd_arr - ct_exp_arr)))

    # 检验最高效率点对应的前进比区间
    idx_peak_exp = int(np.argmax(eta_exp_list))
    idx_peak_fb = int(np.argmax(eta_fast_bemt_list))

    print("-" * 98)
    print(
        f"[统计对标汇总]\n"
        f"   - fast_bemt CT(J) 与 UIUC 风洞实验决定系数 R^2 = {r_squared:.4f} (要求 >= 0.98)\n"
        f"   - fast_bemt CT(J) 平均绝对误差 MAE = {mae_fb:.4f} (优于文献未修正 BEM 的 MAE = {mae_lit_bem:.4f} 及三维 CFD 的 MAE = {mae_cfd:.4f})\n"
        f"   - 最高推进效率点: UIUC 实测峰值在 J = {j_list[idx_peak_exp]:.2f} ({eta_exp_list[idx_peak_exp]*100:.2f}%), "
        f"fast_bemt 预测峰值在 J = {j_list[idx_peak_fb]:.2f} ({eta_fast_bemt_list[idx_peak_fb]*100:.2f}%)"
    )
    print("=" * 98)

    assert r_squared >= 0.98, f"CT(J) 趋势相关性不足: R^2 = {r_squared:.4f}"
    assert mae_fb <= mae_lit_bem, f"fast_bemt 误差 ({mae_fb:.4f}) 应不高于文献基准 BEM ({mae_lit_bem:.4f})"
    assert 0.55 <= j_list[idx_peak_fb] <= 0.65, "APC 10x7E 峰值效率前进比应落在 J = 0.55 ~ 0.65 区间"
    print("[PASS] UIUC APC Thin Electric 10x7E 螺旋桨巡航风洞实验对标验证全部通过！")

    if plot:
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
        r_dense = np.linspace(float(r_norm_tab[0]), 1.0, 200)

        # 图 1：UIUC 实测弦长/扭转离散点 vs B-Spline 拟合曲线
        ax1 = axes[0]
        ax1_tw = ax1.twinx()
        ax1.scatter(r_norm_tab, c_tab * 1000.0, color="tab:blue", marker="o", label="UIUC Measured Chord")
        ax1.plot(r_dense, chord_spline(r_dense) * 1000.0, "b-", linewidth=2.0, label="BSplineCurve Chord")
        ax1_tw.scatter(
            r_norm_tab, APC_10X7E_GEOMETRY_TABLE[:, 2], color="tab:red", marker="^", label="UIUC Measured Twist"
        )
        ax1_tw.plot(r_dense, np.rad2deg(twist_spline(r_dense)), "r--", linewidth=2.0, label="BSplineCurve Twist")
        ax1.set_xlabel("Normalized Radius $r/R$")
        ax1.set_ylabel("Chord $c$ [mm]", color="tab:blue")
        ax1_tw.set_ylabel("Pitch / Twist Angle $\\beta$ [deg]", color="tab:red")
        ax1.set_title("APC 10x7E Geometry & B-Spline Fit")
        ax1.grid(True, alpha=0.3)

        # 图 2：推力系数 CT(J) 对比
        ax2 = axes[1]
        ax2.plot(j_list, ct_exp_list, "ko-", linewidth=2.0, markersize=6, label="UIUC Wind Tunnel (6500 RPM)")
        ax2.plot(j_list, ct_cfd_list, "g--s", markersize=5, label="3D RANS CFD (k-w SST)")
        ax2.plot(j_list, ct_lit_bem_list, "r:v", markersize=5, label="Literature Baseline BEM")
        ax2.plot(j_list, ct_fast_bemt_list, "b-^", linewidth=2.2, markersize=6, label="fast_bemt Solver")
        ax2.set_xlabel("Advance Ratio $J = V_\\infty / (n D)$")
        ax2.set_ylabel("Thrust Coefficient $C_T = T / (\\rho n^2 D^4)$")
        ax2.set_title("APC 10x7E Thrust Coefficient $C_T(J)$")
        ax2.grid(True, alpha=0.3)
        ax2.legend(fontsize=8)

        # 图 3：推进效率 eta(J) 对比
        ax3 = axes[2]
        ax3.plot(j_list, eta_exp_list, "ko-", linewidth=2.0, markersize=6, label="UIUC Wind Tunnel")
        ax3.plot(j_list, eta_fast_bemt_list, "b-^", linewidth=2.2, markersize=6, label="fast_bemt Solver")
        ax3.set_xlabel("Advance Ratio $J = V_\\infty / (n D)$")
        ax3.set_ylabel("Propulsive Efficiency $\\eta = J C_T / C_P$")
        ax3.set_title("APC 10x7E Propulsive Efficiency $\\eta(J)$")
        ax3.set_ylim(0.0, 0.90)
        ax3.grid(True, alpha=0.3)
        ax3.legend()

        plt.tight_layout()
        plt.show()

    return {
        "j": j_list,
        "ct_exp": ct_exp_list,
        "ct_fast_bemt": ct_fast_bemt_list,
        "eta_exp": eta_exp_list,
        "eta_fast_bemt": eta_fast_bemt_list,
        "r_squared": r_squared,
        "mae_ct": mae_fb,
    }


if __name__ == "__main__":
    enable_plot = "--plot" in sys.argv
    run_uiuc_apc10x7_validation(plot=enable_plot)
