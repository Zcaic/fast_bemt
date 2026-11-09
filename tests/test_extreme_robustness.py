import time
import numpy as np
import os 
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "python")))
import fast_bemt as fb


def build_base_airfoils(n_r: int = 20) -> list[fb.Airfoil3D]:
    """生成 20 个径向站位的三维翼型插值表，网格范围仅为常规的 [-18 deg, +18 deg]"""
    alpha_grid = np.deg2rad(np.linspace(-18.0, 18.0, 73))
    re_grid = np.array([5.0e4, 5.0e5, 2.0e6, 5.0e6])
    ma_grid = np.array([0.0, 0.3, 0.6, 0.85])
    A, RE, MA = np.meshgrid(alpha_grid, re_grid, ma_grid, indexing="ij")

    airfoils = []
    for i in range(n_r):
        span_frac = i / (n_r - 1)
        alpha_0 = np.deg2rad(-2.0 * (1.0 - 0.5 * span_frac))
        pg_factor = 1.0 / np.sqrt(1.0 - (0.75 * MA) ** 2)
        re_scale = (RE / 1.0e6) ** (-0.08)

        a_eff = A - alpha_0
        cl_linear = 2.0 * np.pi * a_eff * pg_factor
        stall_blend = 1.0 / (1.0 + (a_eff / np.deg2rad(14.0)) ** 6)
        cl_3d = cl_linear * stall_blend + 1.05 * np.sin(2.0 * a_eff) * (1.0 - stall_blend)
        cd_min = (0.0085 - 0.002 * span_frac) * re_scale
        cd_3d = cd_min + 0.025 * (a_eff / np.deg2rad(10.0)) ** 2 + 0.05 * (MA**3)

        airfoils.append(fb.create_3d_airfoil(alpha_grid, re_grid, ma_grid, cl_3d, cd_3d))
    return airfoils


def build_standard_rotor(airfoils: list[fb.Airfoil3D]) -> fb.RotorGeometry:
    n_r = len(airfoils)
    radius = 1.25
    hub_radius = 0.15
    edges = np.linspace(hub_radius, radius, n_r + 1)
    r = 0.5 * (edges[:-1] + edges[1:])
    dr = np.diff(edges)
    r_norm = r / radius
    chord = 0.10 * (1.0 - 0.35 * r_norm**2)
    twist = np.deg2rad(28.0 * (1.0 - r_norm) + 4.0)
    return fb.RotorGeometry(3, radius, hub_radius, r, chord, twist, airfoils, 1, dr)


def assert_result_all_finite(res: fb.BEMTResult, case_name: str, check_grad: bool = True):
    """严格断言求解结果中的所有标量、2D 场矩阵和全部解析梯度均无 NaN 或 Inf"""
    scalars = [
        res.thrust, res.torque, res.power,
        res.fx, res.fy, res.fz,
        res.mx, res.my, res.mz,
        res.ct, res.cq, res.cp, res.fm,
        res.vi_disk_mean, res.wake_skew_chi, res.pitt_kx, res.pitt_ky,
    ]
    if not np.all(np.isfinite(scalars)):
        raise AssertionError(f"[{case_name}] 集总气动标量出现 NaN/Inf: {scalars}")

    arrays_2d = [
        res.phi, res.alpha, res.vi, res.vt,
        res.cl, res.cd, res.re, res.ma,
        res.f_tip, res.df_z, res.df_theta,
    ]
    for idx, arr in enumerate(arrays_2d):
        if not np.all(np.isfinite(arr)):
            raise AssertionError(f"[{case_name}] 2D 场矩阵 #{idx} 出现 NaN/Inf!")

    if check_grad:
        s = res.sensitivity
        if not s.valid:
            raise AssertionError(f"[{case_name}] 敏度结构体未标记为 valid!")
        grad_arrays = [
            s.dT_dchord, s.dT_dtwist,
            s.dQ_dchord, s.dQ_dtwist,
            s.dP_dchord, s.dP_dtwist,
            s.dMx_dchord, s.dMx_dtwist,
            s.dMy_dchord, s.dMy_dtwist,
            s.dForces_dCond,
            s.dTr_dchord_diag, s.dTr_dtwist_diag,
        ]
        for idx, garr in enumerate(grad_arrays):
            if not np.all(np.isfinite(garr)):
                raise AssertionError(f"[{case_name}] 解析梯度数组 #{idx} 出现 NaN/Inf!")


def run_extreme_stress_tests():
    print("=" * 82)
    print("FAST_BEMT 极端恶劣工况与病态几何鲁棒性压力测试 (Robustness Stress Suite)")
    print("=" * 82)

    airfoils = build_base_airfoils(20)
    geom = build_standard_rotor(airfoils)
    cfg = fb.SolverConfig(
        n_azimuth=36,
        wake_model=fb.WakeModel.GlauertPittPeters,
        enable_tip_loss=True,
        enable_vrs_correction=True,
        enable_stall_delay=True,
        enable_tangential_induction=True,
    )
    solver = fb.BEMTSolver(geom, cfg)

    # =========================================================================
    # 测试 1：6 类典型极限物理工况测试（验证流场与梯度无 NaN/Inf）
    # =========================================================================
    extreme_cases = [
        (
            "1. 垂直深度涡环状态 (Deep VRS, Vz = -12 m/s)",
            fb.OperatingCondition(vx=0.5, vy=0.0, vz=-12.0, omega=180.0, pitch_0=np.deg2rad(8.0)),
        ),
        (
            "2. 斜下降涡环 + 强侧风 + 大周期变距",
            fb.OperatingCondition(
                vx=8.0, vy=-6.0, vz=-10.5, omega=180.0,
                pitch_0=np.deg2rad(10.0), pitch_1c=np.deg2rad(8.0), pitch_1s=np.deg2rad(-12.0),
            ),
        ),
        (
            "3. 超大前进比边飞强反流区 (Vx = 120 m/s, mu = 0.53)",
            fb.OperatingCondition(vx=120.0, vy=15.0, vz=2.0, omega=180.0, pitch_0=np.deg2rad(4.0)),
        ),
        (
            "4. 大负总距高速风车制动状态 (Pitch = -28 deg, Vz = +35 m/s)",
            fb.OperatingCondition(vx=15.0, vy=0.0, vz=35.0, omega=180.0, pitch_0=np.deg2rad(-28.0)),
        ),
        (
            "5. 全盘深度失速极端大总距 (Pitch = +65 deg, 迎角远超表格边界)",
            fb.OperatingCondition(vx=20.0, vy=5.0, vz=10.0, omega=180.0, pitch_0=np.deg2rad(65.0)),
        ),
        (
            "6. 极低转速怠速停转边缘 (Omega = 0.5 rad/s, 强横风 30 m/s)",
            fb.OperatingCondition(vx=30.0, vy=-10.0, vz=-5.0, omega=0.5, pitch_0=np.deg2rad(15.0)),
        ),
    ]

    print("\n[第一部分] 6 类典型极限恶劣飞行工况单点测试:")
    for name, cond in extreme_cases:
        res = solver.solve(cond, compute_gradients=True)
        assert_result_all_finite(res, name, check_grad=True)
        alpha_min_deg = np.rad2deg(np.min(res.alpha))
        alpha_max_deg = np.rad2deg(np.max(res.alpha))
        print(
            f"  [PASS] {name}\n"
            f"         -> Conv={res.converged}, Fallback={res.total_fallback_cells:3d}/720 | "
            f"T={res.thrust:8.1f} N, Q={res.torque:7.1f} N·m | "
            f"迎角范围=[{alpha_min_deg:6.1f}°, {alpha_max_deg:6.1f}°] | "
            f"||dT/dtwist||={np.linalg.norm(res.sensitivity.dT_dtwist):.2e}"
        )

    # =========================================================================
    # 测试 2：模拟智能优化算法（GA/PSO/DE）随机生成的 50 组畸形/病态几何测试
    # =========================================================================
    print("\n[第二部分] 模拟进化算法初期随机撒点：50 组极端畸形桨叶几何 + 解析梯度压测:")
    rng = np.random.default_rng(42)
    n_random_geom = 50
    cond_stress = fb.OperatingCondition(
        vx=28.0, vy=-8.0, vz=-7.0, omega=190.0,
        pitch_0=np.deg2rad(12.0), pitch_1c=np.deg2rad(4.0), pitch_1s=np.deg2rad(-6.0),
    )

    t0 = time.perf_counter()
    fallback_total = 0
    for k in range(n_random_geom):
        # 随机生成极不合理的弦长: 包含 0.2mm 极细弦长到 0.55m 超宽弦长，甚至夹杂锯齿跳变
        rand_chord = rng.uniform(0.0002, 0.55, size=20)
        # 随机生成剧烈震荡的扭转角: -60 deg 到 +85 deg
        rand_twist = np.deg2rad(rng.uniform(-60.0, 85.0, size=20))
        spin = 1 if (k % 2 == 0) else -1

        bad_geom = fb.RotorGeometry(
            num_blades=int(rng.integers(2, 7)),
            radius=geom.radius,
            hub_radius=geom.hub_radius,
            r=geom.r,
            chord=rand_chord,
            twist=rand_twist,
            airfoils=airfoils,
            spin_dir=spin,
            dr=geom.dr,
        )
        bad_solver = fb.BEMTSolver(bad_geom, cfg)
        res_bad = bad_solver.solve(cond_stress, compute_gradients=True)
        assert_result_all_finite(res_bad, f"RandomGeom #{k}", check_grad=True)
        fallback_total += res_bad.total_fallback_cells

    elapsed_geom_ms = (time.perf_counter() - t0) * 1e3
    print(
        f"  [PASS] 50 组极端随机几何 (弦长 0.2mm~0.55m, 扭转 -60°~+85°) 100% 通过！\n"
        f"         -> 无任何 NaN/Inf 崩溃，总耗时: {elapsed_geom_ms:.1f} ms, 累计触发 Brent 保护微元: {fallback_total}"
    )

    # =========================================================================
    # 测试 3：300 组全域随机极端工况 solve_batch 多线程并发压测
    # =========================================================================
    print("\n[第三部分] 300 组全域随机极限工况 solve_batch() 多线程并发压测:")
    random_conds = []
    for _ in range(300):
        c = fb.OperatingCondition(
            vx=float(rng.uniform(-80.0, 120.0)),
            vy=float(rng.uniform(-60.0, 60.0)),
            vz=float(rng.uniform(-25.0, 45.0)),
            omega=float(rng.uniform(5.0, 300.0)),
            pitch_0=float(np.deg2rad(rng.uniform(-35.0, 55.0))),
            pitch_1c=float(np.deg2rad(rng.uniform(-15.0, 15.0))),
            pitch_1s=float(np.deg2rad(rng.uniform(-15.0, 15.0))),
        )
        random_conds.append(c)

    t0 = time.perf_counter()
    batch_results = solver.solve_batch(random_conds, compute_gradients=True)
    t_batch_ms = (time.perf_counter() - t0) * 1e3

    for idx, r in enumerate(batch_results):
        assert_result_all_finite(r, f"BatchExtreme #{idx}", check_grad=True)

    conv_count = sum(1 for r in batch_results if r.converged)
    print(
        f"  [PASS] 300 组全域随机极端工况并行求解 + 求导 100% 有限无崩溃！\n"
        f"         -> 严格收敛率: {conv_count}/300 ({100.0*conv_count/300:.1f}%), 总耗时: {t_batch_ms:.1f} ms"
    )

    # =========================================================================
    # 测试 4：在“深度失速 + 反流区 + 下降涡环”复合极端工况下校验解析梯度与有限差分一致性
    # =========================================================================
    print("\n[第四部分] 复合极端工况下 (含反流与失速外推) 解析梯度 vs 中心有限差分检验:")
    cond_hard = fb.OperatingCondition(
        vx=65.0, vy=-15.0, vz=-6.0, omega=160.0,
        pitch_0=np.deg2rad(22.0), pitch_1c=np.deg2rad(3.0), pitch_1s=np.deg2rad(-5.0),
    )
    res_hard = solver.solve(cond_hard, compute_gradients=True)

    eps = 1e-6
    fd_dT_dtwist = np.zeros(20)
    fd_dQ_dchord = np.zeros(20)
    for i in range(20):
        tw_p, tw_m = geom.twist.copy(), geom.twist.copy()
        tw_p[i] += eps
        tw_m[i] -= eps
        rp = fb.BEMTSolver(fb.RotorGeometry(3, geom.radius, geom.hub_radius, geom.r, geom.chord, tw_p, airfoils, 1, geom.dr), cfg).solve(cond_hard, False)
        rm = fb.BEMTSolver(fb.RotorGeometry(3, geom.radius, geom.hub_radius, geom.r, geom.chord, tw_m, airfoils, 1, geom.dr), cfg).solve(cond_hard, False)
        fd_dT_dtwist[i] = (rp.thrust - rm.thrust) / (2.0 * eps)

        ch_p, ch_m = geom.chord.copy(), geom.chord.copy()
        ch_p[i] += eps
        ch_m[i] -= eps
        rp_c = fb.BEMTSolver(fb.RotorGeometry(3, geom.radius, geom.hub_radius, geom.r, ch_p, geom.twist, airfoils, 1, geom.dr), cfg).solve(cond_hard, False)
        rm_c = fb.BEMTSolver(fb.RotorGeometry(3, geom.radius, geom.hub_radius, geom.r, ch_m, geom.twist, airfoils, 1, geom.dr), cfg).solve(cond_hard, False)
        fd_dQ_dchord[i] = (rp_c.torque - rm_c.torque) / (2.0 * eps)

    err_twist = np.linalg.norm(res_hard.sensitivity.dT_dtwist - fd_dT_dtwist) / np.linalg.norm(fd_dT_dtwist)
    err_chord = np.linalg.norm(res_hard.sensitivity.dQ_dchord - fd_dQ_dchord) / np.linalg.norm(fd_dQ_dchord)
    print(f"  - 复合极端工况下 dT/d(twist) 相对 L2 误差 : {err_twist:.3e}")
    print(f"  - 复合极端工况下 dQ/d(chord) 相对 L2 误差 : {err_chord:.3e}")
    print("=" * 82)


if __name__ == "__main__":
    run_extreme_stress_tests()