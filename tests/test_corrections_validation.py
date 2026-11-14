import os
import sys
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "python")))
import fast_bemt as fb


def create_stalling_airfoil() -> fb.Airfoil3D:
    """构造在 alpha = 13 deg 发生明显失速的二维翼型极曲线表 (用于测试 Du-Selig 3D 失速延迟)"""
    alpha_grid = np.deg2rad(np.linspace(-20.0, 30.0, 101))
    cl_pot = 2.0 * np.pi * alpha_grid
    stall_weight = 1.0 / (1.0 + (np.abs(alpha_grid) / np.deg2rad(13.0)) ** 8)
    cl_1d = cl_pot * stall_weight + 1.05 * np.sin(2.0 * alpha_grid) * (1.0 - stall_weight)
    cd_1d = 0.009 + 0.015 * (alpha_grid / np.deg2rad(10.0)) ** 2 + 0.35 * (1.0 - stall_weight)
    return fb.Airfoil3D.from_1d_polar(alpha_grid, cl_1d, cd_1d, ref_re=1.0e6)


def run_corrections_validation():
    print("=" * 86)
    print("FAST_BEMT 物理正确性验证 (Step 2): 三大物理修正模型孤立对标验证")
    print("=" * 86)

    n_r = 40
    num_blades = 3
    radius = 1.0
    hub_radius = 0.12

    # 1. 手动实例化弦长与扭转分布的 fb.BSplineCurve 曲线对象
    n_ctrl = 6
    x_ctrl = np.linspace(hub_radius / radius, 1.0, n_ctrl)
    chord_curve = fb.BSplineCurve(x_ctrl, np.full(n_ctrl, 0.09), degree=1)
    twist_curve = fb.BSplineCurve(x_ctrl, np.deg2rad(18.0 * (1.0 - x_ctrl)), degree=1)

    # 2. 给定 5 个主截面 (r/R = 0.1, 0.2, 0.5, 0.75, 1.0) 的翼型表，自动插值到 n_radial = 40 个叶素
    section_r_norm = np.array([0.10, 0.20, 0.50, 0.75, 1.00])
    five_airfoils = [create_stalling_airfoil() for _ in range(5)]

    geom = fb.RotorGeometry(
        num_blades=num_blades,
        radius=radius,
        hub_radius=hub_radius,
        n_radial=n_r,
        chord_curve=chord_curve,
        twist_curve=twist_curve,
        airfoils=five_airfoils,
        section_r_norm=section_r_norm,
        spin_dir=1,
    )
    r = geom.r
    x = r / radius
    cond_hover = fb.OperatingCondition(vx=0.0, vy=0.0, vz=0.0, omega=160.0, pitch_0=np.deg2rad(6.0))

    # =========================================================================
    # 验证 2.1：Prandtl 叶尖与叶根损失因子 F(r) 精确公式对标
    # =========================================================================
    print("\n[验证 2.1] Prandtl 叶尖/叶根损失修正 (enable_tip_loss = False vs True):")
    cfg_no_tip = fb.SolverConfig(
        n_azimuth=1, enable_tip_loss=False, enable_vrs_correction=False,
        enable_stall_delay=False, enable_tangential_induction=False
    )
    cfg_with_tip = fb.SolverConfig(
        n_azimuth=1, enable_tip_loss=True, enable_vrs_correction=False,
        enable_stall_delay=False, enable_tangential_induction=False
    )

    res_no_tip = fb.BEMTSolver(geom, cfg_no_tip).solve(cond_hover)
    res_with_tip = fb.BEMTSolver(geom, cfg_with_tip).solve(cond_hover)

    phi_vec = res_with_tip.phi[:, 0]
    abs_sin = np.sqrt(np.sin(phi_vec) ** 2 + (1e-3) ** 2)
    f_tip_arg = 0.5 * num_blades * (radius - r) / (r * abs_sin)
    f_root_arg = 0.5 * num_blades * (r - hub_radius) / (r * abs_sin)
    F_theory = (2.0 / np.pi) * np.arccos(np.exp(-f_tip_arg)) * (2.0 / np.pi) * np.arccos(np.exp(-f_root_arg))
    F_theory = np.maximum(F_theory, 0.05)

    F_cpp = res_with_tip.f_tip[:, 0]
    err_f_prandtl = np.max(np.abs(F_cpp - F_theory))

    b_tip_leishman = 1.0 - np.sqrt(2.0 * res_no_tip.ct) / num_blades
    ct_ratio_actual = res_with_tip.ct / res_no_tip.ct

    print(f"  - Prandtl 因子 F(r) C++内部值 vs 独立理论公式最大绝对偏差 : {err_f_prandtl:.3e}")
    print(f"  - 开启叶尖损失前后拉力系数 C_T : {res_no_tip.ct:.6f} -> {res_with_tip.ct:.6f} (保持率 {ct_ratio_actual*100:.2f}%, 理论 B_tip^2~B_tip^3 约 {(b_tip_leishman**2.5)*100:.2f}%)")
    print(f"  - 开启叶尖损失前后悬停品质因数 : FM = {res_no_tip.fm:.4f} -> {res_with_tip.fm:.4f}")
    print("  - 代表性径向站位 Prandtl 损失因子 F 与局部推力分布 dT/dr 对比:")
    print(f"    r/R = {x[0]:.3f} (近叶根): F = {F_cpp[0]:.4f} | dT/dr (无损失 vs 有损失) = {res_no_tip.dT_dr_mean[0]:6.1f} -> {res_with_tip.dT_dr_mean[0]:6.1f} N/m")
    print(f"    r/R = {x[20]:.3f} (中展段): F = {F_cpp[20]:.4f} | dT/dr (无损失 vs 有损失) = {res_no_tip.dT_dr_mean[20]:6.1f} -> {res_with_tip.dT_dr_mean[20]:6.1f} N/m")
    print(f"    r/R = {x[-1]:.3f} (近叶尖): F = {F_cpp[-1]:.4f} | dT/dr (无损失 vs 有损失) = {res_no_tip.dT_dr_mean[-1]:6.1f} -> {res_with_tip.dT_dr_mean[-1]:6.1f} N/m")

    # =========================================================================
    # 验证 2.2：Buhl (2005) 垂直下降涡环态 (VRS) 诱导速度曲线验证
    # =========================================================================
    print("\n[验证 2.2] 垂直爬升 -> 悬停 -> 深度涡环下降 (VRS) 无量纲诱导速度包线验证:")
    cfg_vrs_on = fb.SolverConfig(n_azimuth=1, enable_tip_loss=True, enable_vrs_correction=True, enable_stall_delay=False)
    cfg_vrs_off = fb.SolverConfig(n_azimuth=1, enable_tip_loss=True, enable_vrs_correction=False, enable_stall_delay=False)
    solver_vrs_on = fb.BEMTSolver(geom, cfg_vrs_on)
    solver_vrs_off = fb.BEMTSolver(geom, cfg_vrs_off)

    vh_disk = solver_vrs_on.solve(cond_hover).vi_disk_mean

    print(f"  - 悬停基准盘平均诱导速度 v_h = {vh_disk:.3f} m/s")
    print("  " + "-" * 78)
    print("   轴向速度比 Vz/vh | 工作状态区域       | vi/vh (开启 VRS 修正) | vi/vh (未修正动量) | 理论预期特征")
    print("  " + "-" * 78)

    for ratio in [0.8, 0.0, -0.5, -1.0, -1.5, -2.2]:
        vz_val = ratio * vh_disk
        c_vz = fb.OperatingCondition(vx=0.0, vy=0.0, vz=vz_val, omega=160.0, pitch_0=np.deg2rad(6.0))
        r_on = solver_vrs_on.solve(c_vz)
        r_off = solver_vrs_off.solve(c_vz)

        vi_ratio_on = r_on.vi_disk_mean / vh_disk
        vi_ratio_off = r_off.vi_disk_mean / vh_disk

        if ratio > 0.1:
            zone = "垂直爬升态 (Climb)"
            th_val = -0.5 * ratio + np.sqrt(0.25 * ratio**2 + 1.0)
            note = f"动量理论解析解 = {th_val:.3f} (< 1.0)"
        elif abs(ratio) <= 0.1:
            zone = "悬停态 (Hover)    "
            note = "基准点 vi/vh = 1.000"
        elif -1.9 <= ratio < -0.1:
            zone = "涡环/湍流态 (VRS) "
            leishman_poly = 1.0 - 1.125 * ratio - 1.372 * (ratio**2) - 1.718 * (ratio**3) - 0.655 * (ratio**4)
            note = f"Leishman 涡环经验曲线 ≈ {leishman_poly:.2f} (先升后降)"
        else:
            zone = "风车制动态 (Windm)"
            wm_val = -0.5 * ratio - np.sqrt(0.25 * ratio**2 - 1.0)
            note = f"风车态动量解析分支 = {wm_val:.3f} (< 1.0 回落)"

        print(f"       {ratio:6.2f}       | {zone} |        {vi_ratio_on:6.3f}         |       {vi_ratio_off:6.3f}       | {note}")

    # =========================================================================
    # 验证 2.3：Du-Selig (1998) 三维旋转失速延迟修正 (叶根增升减阻 vs 叶尖不变)
    # =========================================================================
    print("\n[验证 2.3] Du-Selig 三维旋转失速延迟修正验证 (大总距失速工况 pitch_0 = 18 deg):")
    cond_stall = fb.OperatingCondition(vx=0.0, vy=0.0, vz=2.0, omega=160.0, pitch_0=np.deg2rad(18.0))
    cfg_2d = fb.SolverConfig(n_azimuth=1, enable_tip_loss=True, enable_vrs_correction=True, enable_stall_delay=False)
    cfg_3d = fb.SolverConfig(n_azimuth=1, enable_tip_loss=True, enable_vrs_correction=True, enable_stall_delay=True)

    res_2d = fb.BEMTSolver(geom, cfg_2d).solve(cond_stall)
    res_3d = fb.BEMTSolver(geom, cfg_3d).solve(cond_stall)

    print(f"  - 开启 3D 失速延迟前后总拉力 : {res_2d.thrust:7.1f} N -> {res_3d.thrust:7.1f} N (增量 +{(res_3d.thrust-res_2d.thrust)/res_2d.thrust*100:.2f}%)")
    print("  - 展向不同弦径比 (c/r) 站位的 2D vs 3D 升阻力系数对比:")
    print("  " + "-" * 78)
    print("   站位 r/R | 当地弦径比 c/r | 局部迎角 [deg] |  Cl (2D -> 3D 修正)  |  Cd (2D -> 3D 修正)")
    print("  " + "-" * 78)
    for idx in [1, 6, 15, 25, 38]:
        c_over_r = geom.chord[idx] / r[idx]
        a_deg = np.rad2deg(res_3d.alpha[idx, 0])
        cl2, cl3 = res_2d.cl[idx, 0], res_3d.cl[idx, 0]
        cd2, cd3 = res_2d.cd[idx, 0], res_3d.cd[idx, 0]
        print(
            f"    {x[idx]:5.3f}   |     {c_over_r:5.3f}      |     {a_deg:5.2f}°     | "
            f"{cl2:6.3f} -> {cl3:6.3f} (+{(cl3-cl2)/cl2*100:4.1f}%) | {cd2:6.4f} -> {cd3:6.4f} ({(cd3-cd2)/cd2*100:+5.1f}%)"
        )
    print("=" * 86)


if __name__ == "__main__":
    run_corrections_validation()
