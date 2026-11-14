"""
==============================================================================
公开风洞实验对标验证脚本 3：APC Thin Electric 10x7E 螺旋桨 0°~90° 斜入流六分量与尾迹验证
==============================================================================
本脚本将 `fast_bemt` 的斜流非均匀入流求解器（方位角离散 + Glauert 斜尾迹动量修正 +
Pitt-Peters 一阶纵向谐波入流模型）与公开斜入流风洞实验数据及经典文献结论进行全方位对标：

1. 实验基准数据源：
   - [Qeios WG08LV.2, 2024] Bilal et al., "Aerodynamic Performance of Small Unmanned Aerial
     Vehicle Propellers in Oblique Flow", Table 1 & Fig. 6–8 (APC Thin Electric 10x7E,
     D = 0.254 m, 0° ~ 90° 倾角扫描, V_inf = 0 ~ 14 m/s, n_max = 98 rev/s).
   - [ETH Zurich / IEEE T-RO, 2019] Gill & D'Andrea, "Computationally Efficient Force and
     Moment Models for Propellers in UAV Forward Flight Applications" (0° ~ 90° 斜入流下
     Glauert + Pitt-Peters 尾迹偏斜与面内 H-force / 俯仰力矩 My / 滚转力矩 Mx 对标).
   - [KU Leuven / Aerospace Sci. Tech., 2017] Theys et al., "Experimental and Numerical Study
     of Micro-Aerial-Vehicle Propeller Performance in Oblique Flow" (0° ~ 90° 全倾角实验).
   - [AIAA Journal, 2021] Serrano et al., "Propeller Aerodynamic Modeling in Oblique Flow"
     (揭示高倾角 60°~90° 下螺旋桨从“轴流减推”向“边条横流类机翼增推”的斜率反转机制).
   - [NASA TM-20210014010 / TM-20220005627] Simmons & Hatke, NASA Langley eVTOL 螺旋桨
     0° ~ 90° 斜入流六分量气动力/力矩与正反桨 (CW / CCW) 对称性准则.

2. 四大核心验证内容：
   - 验证 1 [静态基准与轴流对称性]：对标 n = 98 rev/s (5880 RPM) 实测静态推力 (T_exp = 5.40 N)，
     并验证纯轴流/悬停下六分量中的面内力 (Fx, Fy) 与面外力矩 (Mx, My) 精确为零。
   - 验证 2 [0° ~ 90° 斜入流倾角扫描与 Pitt-Peters 尾迹不动点]：在风洞典型风速 V_inf = 10.5 m/s
     (J = 0.422) 下扫描 0° ~ 90° 入流倾角，对标推力系数 CT 随倾角的变化趋势、面内力 Fx、
     滚转力矩 Mx、俯仰力矩 My，并严格验证尾迹偏斜角 chi 与 Pitt-Peters 纵向入流梯度
     kx = (15*pi/23)*tan(chi/2) 的不动点收敛精度 (< 1e-9)。
   - 验证 3 [高倾角“轴流减推 -> 边条横流增推”反转特性]：验证低倾角 (0°、30°) 下 CT 随 J 增大
     而下降，而在 90° 纯边条横流下 CT 随 J 增大而单调上升（类机翼增升效应）。
   - 验证 4 [尾迹模型对比、CW/CCW 镜像对称性与 45° 斜流全耦合 IFT 解析梯度校验]：
     对比 GlauertPittPeters (My != 0) 与 GlauertAnnulus (My == 0) 的物理差异，检验正反桨
     对称性准则，并通过中心有限差分验证 45° 斜入流下包含 dkx/dphi 秩-1 耦合的六分量解析梯度。

运行方式：
    python tests/test_benchmark_oblique_inflow_apc10x7.py
    python tests/test_benchmark_oblique_inflow_apc10x7.py --plot   # 可选：弹出三联对标可视化图
"""

import sys,os
import numpy as np


sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "python")))
import fast_bemt as fb


# ==============================================================================
# 1. APC Thin Electric 10x7E 官方几何表与展向变厚度极曲线构建 (自包含独立模块)
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


def fit_bspline_from_uiuc_table(
    r_norm_data: np.ndarray,
    values_data: np.ndarray,
    num_ctrl_pts: int = 10,
    degree: int = 3,
) -> fb.BSplineCurve:
    """利用最小二乘法将 UIUC 离散几何表拟合为 Clamped B-Spline 控制点曲线。"""
    ctrl_r_norm = np.linspace(float(r_norm_data[0]), float(r_norm_data[-1]), num_ctrl_pts)
    template_curve = fb.BSplineCurve(ctrl_r_norm, np.zeros(num_ctrl_pts), degree=degree)
    basis_mat = np.asarray(template_curve.basis_matrix(r_norm_data), dtype=float)

    ctrl_vals = np.zeros(num_ctrl_pts, dtype=float)
    ctrl_vals[0] = float(values_data[0])
    ctrl_vals[-1] = float(values_data[-1])

    rhs = values_data - basis_mat[:, 0] * ctrl_vals[0] - basis_mat[:, -1] * ctrl_vals[-1]
    b_inner = basis_mat[:, 1:-1]
    ctrl_vals[1:-1], _, _, _ = np.linalg.lstsq(b_inner, rhs, rcond=None)

    return fb.BSplineCurve(ctrl_r_norm, ctrl_vals, degree=degree)


def build_apc10x7_spanwise_airfoils(reynolds_075: float = 1.15e5) -> tuple[np.ndarray, list[fb.Airfoil3D]]:
    """
    根据 Brandt (2005) Table A.1 及 Takeyeldein (2026) Appendix A，构建 APC Thin Electric 10x7
    桨根 (r/R = 0.15)、中展段 (r/R = 0.55) 与桨尖 (r/R = 1.00) 三个控制截面的 Airfoil3D 极曲线。
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


def build_apc10x7_oblique_geometry(
    num_elements: int = 32,
    spin_dir: int = 1,
) -> tuple[fb.RotorGeometry, fb.BSplineCurve, fb.BSplineCurve]:
    """构建用于斜入流六分量计算的 APC Thin Electric 10x7E 螺旋桨几何对象。"""
    radius = 0.127  # D = 10 inch = 0.254 m -> R = 0.127 m
    r_norm_tab = APC_10X7E_GEOMETRY_TABLE[:, 0]
    c_tab = APC_10X7E_GEOMETRY_TABLE[:, 1] * radius
    twist_rad_tab = np.deg2rad(APC_10X7E_GEOMETRY_TABLE[:, 2])
    hub_radius = float(r_norm_tab[0] * radius)

    chord_spline = fit_bspline_from_uiuc_table(r_norm_tab, c_tab, num_ctrl_pts=10, degree=3)
    twist_spline = fit_bspline_from_uiuc_table(r_norm_tab, twist_rad_tab, num_ctrl_pts=10, degree=3)
    section_r_norm, section_airfoils = build_apc10x7_spanwise_airfoils()

    geom = fb.RotorGeometry(
        num_blades=2,
        radius=radius,
        hub_radius=hub_radius,
        n_radial=num_elements,
        chord_curve=chord_spline,
        twist_curve=twist_spline,
        airfoils=section_airfoils,
        section_r_norm=section_r_norm,
        spin_dir=spin_dir,
    )
    return geom, chord_spline, twist_spline


def eval_qeios_oblique_empirical_ct(j_total: float, incidence_deg: float) -> float:
    """
    计算 Qeios WG08LV.2 (Eq. 4–5 & Fig. 6) 给出的 APC Thin Electric 10x7E 斜入流经验模型基准推力系数：
      CT_ref(J, zeta) = -0.156 * J_par^2 - 0.008 * J_par + 0.109
    其中 J_par = J * cos(zeta) 为沿桨盘法向（推力轴向）的有效前进比分量。
    """
    zeta_rad = np.deg2rad(incidence_deg)
    j_par = j_total * np.cos(zeta_rad)
    return float(-0.156 * (j_par ** 2) - 0.008 * j_par + 0.109)


# ==============================================================================
# 2. 执行 APC 10x7E 斜入流四项核心对标验证
# ==============================================================================

def run_oblique_inflow_apc10x7_validation(plot: bool = False) -> dict:
    print("=" * 106)
    print("  公开风洞实验验证 3：APC Thin Electric 10x7E 螺旋桨 0°~90° 斜入流六分量与 Pitt-Peters 尾迹对标")
    print("  数据源: Qeios WG08LV.2 (2024) | Gill & D'Andrea (ETH, 2019) | Theys et al. (KU Leuven, 2017)")
    print("=" * 106)

    diameter = 0.254
    radius = 0.127
    rho = 1.225

    geom, chord_spline, twist_spline = build_apc10x7_oblique_geometry(num_elements=32, spin_dir=1)

    # 采用 36 个方位角扇区离散 (Delta_psi = 10 deg)，开启 Glauert + Pitt-Peters 斜尾迹谐波入流模型
    cfg_pitt = fb.SolverConfig(
        n_azimuth=36,
        wake_model=fb.WakeModel.GlauertPittPeters,
        enable_tip_loss=True,
        enable_vrs_correction=False,
        enable_stall_delay=True,
        enable_tangential_induction=True,
    )
    solver_pitt = fb.BEMTSolver(geom, cfg_pitt)

    # 同时构建 GlauertAnnulus 求解器（含动量斜流总诱导速度修正与周向推进/后退不对称，不含纵向线性梯度 kx）
    cfg_annulus = fb.SolverConfig(
        n_azimuth=36,
        wake_model=fb.WakeModel.GlauertAnnulus,
        enable_tip_loss=True,
        enable_vrs_correction=False,
        enable_stall_delay=True,
        enable_tangential_induction=True,
    )
    solver_annulus = fb.BEMTSolver(geom, cfg_annulus)

    # --------------------------------------------------------------------------
    # 验证 A：静态最大推力对标 (Qeios WG08LV.2 Table 1: n_max = 98 rev/s = 5880 RPM, T_max = 5.4 N)
    # --------------------------------------------------------------------------
    n_static = 98.0
    omega_static = 2.0 * np.pi * n_static
    cond_static = fb.OperatingCondition(vx=0.0, vy=0.0, vz=0.0, omega=omega_static, rho=rho)
    res_static = solver_pitt.solve(cond_static)
    t_static_exp = 5.40  # Qeios WG08LV.2 Table 1 实测静态推力 [N]
    err_static_pct = (res_static.thrust - t_static_exp) / t_static_exp * 100.0
    print(
        f"[验证 1：静态基准与轴流对称性] n = {n_static:.0f} rev/s (5880 RPM): "
        f"实测静态推力 T_exp = {t_static_exp:.2f} N | fast_bemt 预测 T = {res_static.thrust:.3f} N "
        f"(偏差 {err_static_pct:+.2f}%)"
    )
    assert res_static.converged
    assert abs(err_static_pct) < 10.0, f"静态推力偏差过大: {err_static_pct:.2f}%"
    # 纯轴流/悬停下六分量中的面内力与面外力矩必须精确为零 (< 1e-13)
    assert abs(res_static.fx) < 1e-13 and abs(res_static.fy) < 1e-13
    assert abs(res_static.mx) < 1e-13 and abs(res_static.my) < 1e-13
    assert np.isclose(res_static.wake_skew_chi, 0.0, atol=1e-14)
    assert np.isclose(res_static.pitt_kx, 0.0, atol=1e-14)

    # --------------------------------------------------------------------------
    # 验证 B：斜入流倾角 0° ~ 90° 扫描 (V_inf = 10.5 m/s, n = 98 rev/s, J = 0.4218)
    #         对标风洞斜入流推力变化、Pitt-Peters 尾迹偏斜角 chi、纵向梯度 kx 与六分量载荷
    # --------------------------------------------------------------------------
    v_inf_test = 10.5  # Qeios WG08LV.2 Fig. 8 典型风洞测试风速 [m/s]
    j_test = v_inf_test / (n_static * diameter)
    incidence_angles_deg = [0.0, 15.0, 30.0, 45.0, 60.0, 75.0, 90.0]

    print("-" * 106)
    print(
        f"  [验证 2：斜入流 0°~90° 倾角扫描] V_inf = {v_inf_test:.1f} m/s, n = {n_static:.0f} rps (J = {j_test:.3f}) 六分量与尾迹参数:"
    )
    print("-" * 106)
    print(
        f"{'倾角(deg)':>9} | {'J_par':>6} | {'Qeios基准CT':>11} | {'Pitt-Peters CT':>14} | "
        f"{'推力 T(N)':>9} | {'H力 Fx(N)':>9} | {'滚转 Mx(Nm)':>11} | {'俯仰 My(Nm)':>11} | "
        f"{'偏斜角 chi':>10} | {'Pitt kx':>8}"
    )
    print("-" * 106)

    sweep_records = []
    ct_denom = rho * (n_static ** 2) * (diameter ** 4)

    for zeta_deg in incidence_angles_deg:
        zeta_rad = np.deg2rad(zeta_deg)
        vx = v_inf_test * np.sin(zeta_rad)
        vz = v_inf_test * np.cos(zeta_rad)
        j_par = j_test * np.cos(zeta_rad)

        cond = fb.OperatingCondition(vx=vx, vy=0.0, vz=vz, omega=omega_static, rho=rho)
        res = solver_pitt.solve(cond, compute_gradients=True)
        res_ann = solver_annulus.solve(cond)
        assert res.converged, f"倾角 {zeta_deg} deg 下 BEMT 未收敛"

        ct_fb = float(res.thrust) / ct_denom
        ct_ann = float(res_ann.thrust) / ct_denom
        ct_ref = eval_qeios_oblique_empirical_ct(j_test, zeta_deg)
        chi_deg = float(np.rad2deg(res.wake_skew_chi))

        # 验证 Pitt-Peters 一阶纵向入流梯度公式 kx = (15 * pi / 23) * tan(chi / 2) 收敛精度
        if vx > 1e-5:
            kx_expected = (15.0 * np.pi / 23.0) * np.tan(0.5 * res.wake_skew_chi)
            assert np.isclose(res.pitt_kx, kx_expected, atol=1e-9), (
                f"Pitt-Peters kx 未收敛至理论不动点: {res.pitt_kx:.8f} vs {kx_expected:.8f}"
            )

        print(
            f"{zeta_deg:>9.1f} | {j_par:>6.3f} | {ct_ref:>11.4f} | {ct_fb:>14.4f} | "
            f"{res.thrust:>9.3f} | {res.fx:>9.4f} | {res.mx:>11.5f} | {res.my:>11.5f} | "
            f"{chi_deg:>9.2f}° | {res.pitt_kx:>8.4f}"
        )

        sweep_records.append(
            {
                "incidence_deg": zeta_deg,
                "j_par": j_par,
                "ct_ref": ct_ref,
                "ct_fb": ct_fb,
                "ct_ann": ct_ann,
                "thrust": float(res.thrust),
                "fx": float(res.fx),
                "fy": float(res.fy),
                "mx": float(res.mx),
                "my": float(res.my),
                "mz": float(res.mz),
                "chi_deg": chi_deg,
                "pitt_kx": float(res.pitt_kx),
            }
        )

    # 校验斜入流推力随倾角变化的决定系数 R^2 与平均相对误差
    ct_ref_arr = np.array([r["ct_ref"] for r in sweep_records])
    ct_fb_arr = np.array([r["ct_fb"] for r in sweep_records])
    r2_oblique = float(np.corrcoef(ct_ref_arr, ct_fb_arr)[0, 1] ** 2)
    rel_err_mean = float(np.mean(np.abs(ct_fb_arr - ct_ref_arr) / ct_ref_arr) * 100.0)
    assert r2_oblique >= 0.96, f"斜入流推力随倾角变化趋势相关性不足: R^2 = {r2_oblique:.4f}"
    assert rel_err_mean < 5.0, f"斜入流平均推力偏差超出预期: {rel_err_mean:.2f}%"

    # 校验滚转力矩 Mx 与尾迹偏斜角 chi 随倾角 0°->90° 严格单调递增，且 0°~75° 面内力 Fx 单调递增
    fx_arr = np.array([r["fx"] for r in sweep_records])
    mx_arr = np.array([r["mx"] for r in sweep_records])
    chi_arr = np.array([r["chi_deg"] for r in sweep_records])
    assert np.all(np.diff(fx_arr[:-1]) > 0.0), "0°~75° 倾角区间内面内后向力 Fx (H-force) 应随倾角单调递增"
    assert np.all(np.diff(mx_arr) > 0.0), "前行/后退桨叶动压差引起的滚转力矩 Mx 应随倾角 0°~90° 单调递增"
    assert np.all(np.diff(chi_arr) > 0.0), "尾迹偏斜角 chi 应随入流倾角 0°~90° 单调递增"

    # --------------------------------------------------------------------------
    # 验证 C：Serrano et al. (AIAA J. 2021) 高倾角“轴流减推 -> 类机翼增推”反转特性
    #         对比 0° (轴流)、30°、60°、90° (纯边条横流) 下 CT 随风速 J 的变化斜率
    # --------------------------------------------------------------------------
    j_sweep = [0.15, 0.30, 0.45, 0.60]
    ct_vs_j_by_angle = {}
    for zeta_deg in [0.0, 30.0, 60.0, 90.0]:
        zeta_rad = np.deg2rad(zeta_deg)
        ct_curve = []
        for j_val in j_sweep:
            v_inf = j_val * n_static * diameter
            cond = fb.OperatingCondition(
                vx=v_inf * np.sin(zeta_rad),
                vy=0.0,
                vz=v_inf * np.cos(zeta_rad),
                omega=omega_static,
                rho=rho,
            )
            res = solver_annulus.solve(cond)
            ct_curve.append(float(res.thrust) / ct_denom)
        ct_vs_j_by_angle[zeta_deg] = ct_curve

    # 在 0° 和 30° 下，CT 随 J 增大而显著下降；在 90° 纯横流下，CT 随 J 增大而单调上升（类机翼增升效应）
    assert ct_vs_j_by_angle[0.0][-1] < ct_vs_j_by_angle[0.0][0], "0° 轴流下 CT 应随 J 增大而减小"
    assert ct_vs_j_by_angle[30.0][-1] < ct_vs_j_by_angle[30.0][0], "30° 斜流下 CT 应随 J 增大而减小"
    assert np.all(np.diff(ct_vs_j_by_angle[90.0]) > 0.0), "90° 边条横流下 CT 应随 J 增大而单调递增（机翼增升效应）"

    print("-" * 106)
    print("  [验证 3：高倾角“轴流减推 -> 类机翼增推”反转特性 (Serrano et al., AIAA J. 2021)]:")
    for zeta_deg in [0.0, 30.0, 60.0, 90.0]:
        vals_str = ", ".join(f"J={j:.2f}: CT={c:.4f}" for j, c in zip(j_sweep, ct_vs_j_by_angle[zeta_deg]))
        trend_label = "随风速单调递增 (Wing-like)" if zeta_deg == 90.0 else "随风速递减 (Propeller-like)"
        print(f"   - 倾角 {zeta_deg:>4.0f}° [{trend_label}]: {vals_str}")

    # --------------------------------------------------------------------------
    # 验证 D：斜流尾迹模型对比、正反桨镜像对称性与 IFT 解析梯度有限差分校验
    # --------------------------------------------------------------------------
    cond_45 = fb.OperatingCondition(
        vx=v_inf_test * np.sin(np.deg2rad(45.0)),
        vy=0.0,
        vz=v_inf_test * np.cos(np.deg2rad(45.0)),
        omega=omega_static,
        rho=rho,
    )
    res_pitt_45 = solver_pitt.solve(cond_45, compute_gradients=True)
    res_annulus_45 = solver_annulus.solve(cond_45)

    # 无侧滑 (vy=0) 时，若不计纵向入流梯度 kx (GlauertAnnulus)，前后半盘对称，俯仰力矩 My 必为 0；
    # 而开启 Pitt-Peters 后，前缘上洗/后缘下洗不对称会产生显著非零俯仰力矩 My！
    assert abs(res_annulus_45.my) < 1e-13, "GlauertAnnulus 无纵向梯度时 My 应精确为 0"
    assert abs(res_pitt_45.my) > 0.01, "GlauertPittPeters 纵向入流梯度 kx 应产生显著俯仰力矩 My"

    # 验证正反桨 (spin_dir = +1 vs -1) 对称性准则 (Simmons NASA TM-20220005627 Section III):
    # 纵向量 (T, P, Fx, My) 保持完全不变，横向/偏航量 (Fy, Mx, Mz) 严格反号
    geom_ccw, _, _ = build_apc10x7_oblique_geometry(num_elements=32, spin_dir=-1)
    res_ccw_45 = fb.BEMTSolver(geom_ccw, cfg_pitt).solve(cond_45)
    assert np.isclose(res_pitt_45.thrust, res_ccw_45.thrust, rtol=1e-12)
    assert np.isclose(res_pitt_45.fx, res_ccw_45.fx, rtol=1e-12)
    assert np.isclose(res_pitt_45.my, res_ccw_45.my, rtol=1e-12)
    assert np.isclose(res_pitt_45.fy, -res_ccw_45.fy, atol=1e-12)
    assert np.isclose(res_pitt_45.mx, -res_ccw_45.mx, atol=1e-12)

    # 验证 45° 斜入流下全耦合 IFT 解析梯度与中心有限差分的一致性 (含 Pitt-Peters dkx/dphi 耦合项)
    eps = 1e-6
    ctrl_idx = 4
    c_vals_base = chord_spline.ctrl_vals.copy()
    c_r_base = chord_spline.ctrl_r_norm.copy()

    c_vals_p = c_vals_base.copy()
    c_vals_m = c_vals_base.copy()
    c_vals_p[ctrl_idx] += eps
    c_vals_m[ctrl_idx] -= eps

    geom_p = fb.RotorGeometry(
        2, radius, geom.hub_radius, 32,
        fb.BSplineCurve(c_r_base, c_vals_p, 3), twist_spline,
        geom.section_airfoils, geom.section_r_norm, 1,
    )
    geom_m = fb.RotorGeometry(
        2, radius, geom.hub_radius, 32,
        fb.BSplineCurve(c_r_base, c_vals_m, 3), twist_spline,
        geom.section_airfoils, geom.section_r_norm, 1,
    )
    res_p = fb.BEMTSolver(geom_p, cfg_pitt).solve(cond_45)
    res_m = fb.BEMTSolver(geom_m, cfg_pitt).solve(cond_45)

    dT_dc_fd = (res_p.thrust - res_m.thrust) / (2.0 * eps)
    dFx_dc_fd = (res_p.fx - res_m.fx) / (2.0 * eps)
    dMx_dc_fd = (res_p.mx - res_m.mx) / (2.0 * eps)
    dMy_dc_fd = (res_p.my - res_m.my) / (2.0 * eps)

    dT_dc_ana = float(res_pitt_45.sensitivity.dT_dchord_ctrl[ctrl_idx])
    dFx_dc_ana = float(res_pitt_45.sensitivity.dFx_dchord_ctrl[ctrl_idx])
    dMx_dc_ana = float(res_pitt_45.sensitivity.dMx_dchord_ctrl[ctrl_idx])
    dMy_dc_ana = float(res_pitt_45.sensitivity.dMy_dchord_ctrl[ctrl_idx])

    assert np.isclose(dT_dc_ana, dT_dc_fd, rtol=1e-4, atol=1e-5)
    assert np.isclose(dFx_dc_ana, dFx_dc_fd, rtol=1e-4, atol=1e-5)
    assert np.isclose(dMx_dc_ana, dMx_dc_fd, rtol=1e-4, atol=1e-5)
    assert np.isclose(dMy_dc_ana, dMy_dc_fd, rtol=1e-4, atol=1e-5)

    print("-" * 106)
    print(
        f"  [验证 4：45° 斜入流尾迹模型对比与全耦合 IFT 解析梯度校验]\n"
        f"   - 尾迹模型俯仰力矩对比: GlauertPittPeters My = {res_pitt_45.my:+.5f} N·m vs GlauertAnnulus My = {res_annulus_45.my:+.5f} N·m\n"
        f"   - 正反桨对称性检验 (CW vs CCW): dT = {abs(res_pitt_45.thrust - res_ccw_45.thrust):.2e} N, "
        f"Mx_CW + Mx_CCW = {abs(res_pitt_45.mx + res_ccw_45.mx):.2e} N·m\n"
        f"   - 45° 斜入流弦长控制点 #4 解析梯度 vs 有限差分:\n"
        f"     dT/dc : 解析 = {dT_dc_ana:+.6f}, 差分 = {dT_dc_fd:+.6f} (相对误差 {abs(dT_dc_ana-dT_dc_fd)/abs(dT_dc_fd)*100:.4f}%)\n"
        f"     dFx/dc: 解析 = {dFx_dc_ana:+.6f}, 差分 = {dFx_dc_fd:+.6f} (相对误差 {abs(dFx_dc_ana-dFx_dc_fd)/abs(dFx_dc_fd)*100:.4f}%)\n"
        f"     dMy/dc: 解析 = {dMy_dc_ana:+.6f}, 差分 = {dMy_dc_fd:+.6f} (相对误差 {abs(dMy_dc_ana-dMy_dc_fd)/abs(dMy_dc_fd)*100:.4f}%)"
    )
    print("=" * 106)
    print(
        f"[PASS] APC 10x7E 0°~90° 斜入流风洞对标全部通过 (CT 决定系数 R^2 = {r2_oblique:.4f}, 平均偏差 = {rel_err_mean:.2f}%)！"
    )

    if plot:
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.6))

        # 图 1：0°~90° 斜入流推力系数 CT 对标
        ang_arr = [r["incidence_deg"] for r in sweep_records]
        axes[0].plot(ang_arr, ct_ref_arr, "ko--", linewidth=2.0, markersize=6, label="Qeios Wind Tunnel Fit")
        axes[0].plot(ang_arr, ct_fb_arr, "b-^", linewidth=2.2, markersize=6, label="fast_bemt (Glauert + Pitt-Peters)")
        axes[0].plot(ang_arr, [r["ct_ann"] for r in sweep_records], "g:s", linewidth=1.8, markersize=5, label="fast_bemt (Glauert Annulus)")
        axes[0].set_xlabel("Inflow Incidence Angle $\\varsigma$ [deg]")
        axes[0].set_ylabel("Thrust Coefficient $C_T = T / (\\rho n^2 D^4)$")
        axes[0].set_title(f"APC 10x7E Oblique Thrust ($V_\\infty={v_inf_test}$ m/s)")
        axes[0].grid(True, alpha=0.3)
        axes[0].legend()

        # 图 2：不同倾角下 CT 随前进比 J 的“轴流减推 -> 类机翼增推”反转
        for zeta_deg, style in [(0.0, "k-o"), (30.0, "b-s"), (60.0, "g-^"), (90.0, "r-D")]:
            axes[1].plot(j_sweep, ct_vs_j_by_angle[zeta_deg], style, linewidth=1.8, label=f"$\\varsigma = {zeta_deg:.0f}^\\circ$")
        axes[1].set_xlabel("Total Advance Ratio $J = V_\\infty / (n D)$")
        axes[1].set_ylabel("Thrust Coefficient $C_T$")
        axes[1].set_title("Propeller-to-Wing Transition across Incidence Angles")
        axes[1].grid(True, alpha=0.3)
        axes[1].legend()

        # 图 3：斜入流面内力 Fx 与面外力矩 Mx, My 随倾角变化
        ax3 = axes[2]
        ax3_m = ax3.twinx()
        ax3.plot(ang_arr, [r["fx"] for r in sweep_records], "b-o", linewidth=2.0, label="In-plane H-Force $F_x$ [N]")
        ax3_m.plot(ang_arr, [r["mx"] for r in sweep_records], "r--s", linewidth=1.8, label="Roll Moment $M_x$ [N·m]")
        ax3_m.plot(ang_arr, [r["my"] for r in sweep_records], "g-.^", linewidth=1.8, label="Pitch Moment $M_y$ (Pitt-Peters) [N·m]")
        ax3.set_xlabel("Inflow Incidence Angle $\\varsigma$ [deg]")
        ax3.set_ylabel("In-plane Force $F_x$ [N]", color="b")
        ax3_m.set_ylabel("Hub Moments $M_x, M_y$ [N·m]", color="r")
        ax3.set_title("Oblique Inflow 6-DOF Off-Axis Loads")
        ax3.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

    return {
        "sweep_records": sweep_records,
        "ct_vs_j_by_angle": ct_vs_j_by_angle,
        "r2_oblique": r2_oblique,
        "rel_err_mean": rel_err_mean,
    }


if __name__ == "__main__":
    enable_plot = "--plot" in sys.argv
    run_oblique_inflow_apc10x7_validation(plot=enable_plot)
