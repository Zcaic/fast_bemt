import os
import sys

import time
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "python")))
import fast_bemt as fb


def create_test_rotor(n_r: int = 20) -> fb.RotorGeometry:
    """构建 20 径向站位 + 20 独立三维翼型表的测试旋翼"""
    radius = 1.25
    hub_radius = 0.15
    edges = np.linspace(hub_radius, radius, n_r + 1)
    r = 0.5 * (edges[:-1] + edges[1:])
    dr = np.diff(edges)

    r_norm = r / radius
    chord = 0.10 * (1.0 - 0.35 * r_norm**2)
    twist = np.deg2rad(28.0 * (1.0 - r_norm) + 4.0)

    alpha_grid = np.deg2rad(np.linspace(-18.0, 18.0, 73))
    re_grid = np.array([1.0e5, 5.0e5, 1.5e6, 5.0e6])
    ma_grid = np.array([0.0, 0.3, 0.5, 0.7])
    A, RE, MA = np.meshgrid(alpha_grid, re_grid, ma_grid, indexing="ij")

    airfoils = []
    for i in range(n_r):
        span_frac = i / (n_r - 1)
        alpha_0 = np.deg2rad(-2.0 * (1.0 - 0.5 * span_frac))
        pg_factor = 1.0 / np.sqrt(1.0 - (0.8 * MA) ** 2)
        re_scale = (RE / 1.0e6) ** (-0.08)

        a_eff = A - alpha_0
        cl_linear = 2.0 * np.pi * a_eff * pg_factor
        stall_blend = 1.0 / (1.0 + (a_eff / np.deg2rad(14.0)) ** 6)
        cl_3d = cl_linear * stall_blend + 1.05 * np.sin(2.0 * a_eff) * (1.0 - stall_blend)
        cd_min = (0.0085 - 0.002 * span_frac) * re_scale
        cd_3d = cd_min + 0.025 * (a_eff / np.deg2rad(10.0)) ** 2 + 0.05 * (MA**3)

        airfoils.append(fb.create_3d_airfoil(alpha_grid, re_grid, ma_grid, cl_3d, cd_3d))

    return fb.RotorGeometry(
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


def generate_transition_envelope_conditions() -> list[fb.OperatingCondition]:
    """
    生成倾转过渡走廊多维工况包线 (共 10 x 10 x 5 = 500 个斜入流工况)
    - 来流速度 V_inf: 0 ~ 45 m/s (10 个点)
    - 桨盘倾角 alpha_disk: 0 deg (轴流) ~ 90 deg (边飞) (10 个点)
    - 总距角 pitch_0: 2 deg ~ 14 deg (5 个点)
    """
    v_list = np.linspace(0.0, 45.0, 10)
    alpha_list = np.deg2rad(np.linspace(0.0, 90.0, 10))
    pitch_list = np.deg2rad(np.linspace(2.0, 14.0, 5))
    omega = 180.0

    conds: list[fb.OperatingCondition] = []
    for v_inf in v_list:
        for alpha_d in alpha_list:
            for p0 in pitch_list:
                conds.append(
                    fb.OperatingCondition.from_freestream(
                        v_inf=float(v_inf),
                        alpha_disk_rad=float(alpha_d),
                        omega_rad_s=omega,
                        pitch_0_rad=float(p0),
                    )
                )
    return conds


def run_benchmark():
    geom = create_test_rotor(n_r=20)
    cfg = fb.SolverConfig(
        n_azimuth=36,
        wake_model=fb.WakeModel.GlauertPittPeters,
        enable_tip_loss=True,
        enable_vrs_correction=True,
        enable_stall_delay=True,
        enable_tangential_induction=True,
    )
    solver = fb.BEMTSolver(geom, cfg)

    conds = generate_transition_envelope_conditions()
    n_cases = len(conds)
    total_cells = n_cases * geom.r.size * cfg.n_azimuth

    print("=" * 78)
    print(f"多工况性能基准测试: 共 {n_cases} 个斜入流工况 (总计 {total_cells:,} 个非线性叶素微元)")
    print("=" * 78)

    # 预热一次
    _ = solver.solve_batch(conds[:10], compute_gradients=True)

    # ------------------------------------------------------------------------
    # Part 1: 不含求导的纯气动求解对比 (compute_gradients = False)
    # ------------------------------------------------------------------------
    print("\n[测试 1] 仅求解流场与六分量力/力矩 (compute_gradients = False):")

    # 1.1 Python 列表循环逐个串行调用 solver.solve
    t0 = time.perf_counter()
    res_serial = [solver.solve(c, compute_gradients=False) for c in conds]
    t_serial_ms = (time.perf_counter() - t0) * 1e3

    # 1.2 C++ OpenMP 批量多线程并行调用 solver.solve_batch
    t0 = time.perf_counter()
    res_parallel = solver.solve_batch(conds, compute_gradients=False)
    t_parallel_ms = (time.perf_counter() - t0) * 1e3

    speedup_solve = t_serial_ms / t_parallel_ms
    print(f"  - Python 循环逐个调用 solve() 耗时 : {t_serial_ms:8.2f} ms (平均 {t_serial_ms*1e3/n_cases:6.1f} us/工况)")
    print(f"  - C++ solve_batch() 多线程并行耗时 : {t_parallel_ms:8.2f} ms (平均 {t_parallel_ms*1e3/n_cases:6.1f} us/工况)")
    print(f"  -> 批量多线程加速比                : {speedup_solve:8.2f} x")

    # ------------------------------------------------------------------------
    # Part 2: 含全解析梯度雅可比矩阵计算对比 (compute_gradients = True)
    # ------------------------------------------------------------------------
    print("\n[测试 2] 求解流场 + 隐函数定理 (IFT) 全解析雅可比梯度 (compute_gradients = True):")

    # 2.1 Python 列表循环逐个串行调用 solver.solve(..., True)
    t0 = time.perf_counter()
    res_grad_serial = [solver.solve(c, compute_gradients=True) for c in conds]
    t_grad_serial_ms = (time.perf_counter() - t0) * 1e3

    # 2.2 C++ OpenMP 批量多线程并行调用 solver.solve_batch(..., True)
    t0 = time.perf_counter()
    res_grad_parallel = solver.solve_batch(conds, compute_gradients=True)
    t_grad_parallel_ms = (time.perf_counter() - t0) * 1e3

    speedup_grad = t_grad_serial_ms / t_grad_parallel_ms
    print(f"  - Python 循环逐个调用 solve() 耗时 : {t_grad_serial_ms:8.2f} ms (平均 {t_grad_serial_ms*1e3/n_cases:6.1f} us/工况)")
    print(f"  - C++ solve_batch() 多线程并行耗时 : {t_grad_parallel_ms:8.2f} ms (平均 {t_grad_parallel_ms*1e3/n_cases:6.1f} us/工况)")
    print(f"  -> 批量多线程加速比                : {speedup_grad:8.2f} x")

    # ------------------------------------------------------------------------
    # Part 3: 严格校验串行与多线程并行结果的一致性
    # ------------------------------------------------------------------------
    thrust_serial = np.array([r.thrust for r in res_grad_serial])
    thrust_parallel = np.array([r.thrust for r in res_grad_parallel])
    grad_serial = np.vstack([r.sensitivity.dT_dtwist for r in res_grad_serial])
    grad_parallel = np.vstack([r.sensitivity.dT_dtwist for r in res_grad_parallel])

    max_diff_t = np.max(np.abs(thrust_serial - thrust_parallel))
    max_diff_g = np.max(np.abs(grad_serial - grad_parallel))
    n_conv = sum(1 for r in res_grad_parallel if r.converged)

    print("\n[测试 3] 数值一致性与收敛性检查:")
    print(f"  - 500 工况全部收敛率               : {n_conv}/{n_cases} ({100.0*n_conv/n_cases:.1f}%)")
    print(f"  - 串行 vs 并行 拉力最大绝对偏差     : {max_diff_t:.3e} N")
    print(f"  - 串行 vs 并行 解析梯度最大绝对偏差 : {max_diff_g:.3e} N/rad")
    print("=" * 78)


if __name__ == "__main__":
    run_benchmark()