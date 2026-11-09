from __future__ import annotations
import os
import sys
import numpy as np

# 若使用 cmake --build build 直接在本地编译，自动将 build 目录加入搜索路径以便直接导入调试
# _pkg_dir = os.path.dirname(os.path.abspath(__file__))
# _root_dir = os.path.abspath(os.path.join(_pkg_dir, "..", ".."))
# _build_dir = os.path.join(_root_dir, "build")
# if os.path.isdir(_build_dir) and _build_dir not in sys.path:
#     sys.path.insert(0, _build_dir)

from ._fast_bemt_impl import (
    WakeModel,
    Airfoil3D,
    RotorGeometry,
    OperatingCondition,
    SolverConfig,
    BEMTSensitivity,
    BEMTResult,
    BEMTSolver,
    set_num_threads,
    get_max_threads
)


def create_3d_airfoil(
    alpha_deg_or_rad: np.ndarray,
    re_grid: np.ndarray,
    ma_grid: np.ndarray,
    cl_3d: np.ndarray,
    cd_3d: np.ndarray,
    degrees: bool = False,
    aspect_ratio: float = 10.0,
    enable_extrapolation: bool = True,
) -> Airfoil3D:
    """
    便捷创建三维翼型表 Airfoil3D 的辅助函数。
    支持直接传入形状为 (N_alpha, N_Re, N_Ma) 的三维 NumPy 数组。
    """
    alpha = np.asarray(alpha_deg_or_rad, dtype=np.float64)
    if degrees:
        alpha = np.deg2rad(alpha)
    re = np.asarray(re_grid, dtype=np.float64)
    ma = np.asarray(ma_grid, dtype=np.float64)

    expected_shape = (alpha.size, re.size, ma.size)
    cl_arr = np.asarray(cl_3d, dtype=np.float64)
    cd_arr = np.asarray(cd_3d, dtype=np.float64)

    if cl_arr.ndim == 3 and cl_arr.shape != expected_shape:
        raise ValueError(f"cl_3d shape {cl_arr.shape} does not match {expected_shape}")

    return Airfoil3D(
        alpha_grid=alpha,
        re_grid=re,
        ma_grid=ma,
        cl_table=np.ascontiguousarray(cl_arr).ravel(),
        cd_table=np.ascontiguousarray(cd_arr).ravel(),
        aspect_ratio=aspect_ratio,
        enable_extrapolation=enable_extrapolation,
    )


__all__ = [
    "WakeModel",
    "Airfoil3D",
    "RotorGeometry",
    "OperatingCondition",
    "SolverConfig",
    "BEMTSensitivity",
    "BEMTResult",
    "BEMTSolver",
    "create_3d_airfoil",
    "set_num_threads",
    "get_max_threads"
]