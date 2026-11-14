from __future__ import annotations
import os
import sys
import numpy as np

from ._fast_bemt_impl import (
    WakeModel,
    BSplineCurve,
    Airfoil3D,
    RotorGeometry,
    OperatingCondition,
    SolverConfig,
    BEMTSensitivity,
    BEMTResult,
    BEMTSolver,
    build_clamped_uniform_knots,
    build_greville_abscissae,
    build_bspline_basis_matrix,
    build_parametric_bspline_basis_matrix,
    set_num_threads,
    get_max_threads,
)

from .optimization import (
    KulfanCSTAirfoil,
    create_airfoil_from_neuralfoil,
    create_section_airfoils_from_neuralfoil,
    build_spanwise_section_weight_matrix,
    compute_element_polar_sensitivities,
    compute_cst_aerodynamic_gradients,
    build_planform_and_twist_distribution_constraints,
    BladeOptimizationResult,
    RotorBladeOptimizer,
    HoverCruiseOptimizationResult,
    HoverCruiseBladeOptimizer,
    OperatingPointSpec,
    ObjectiveTermSpec,
    MetricConstraintSpec,
    MultiPointOptimizationResult,
    MultiPointBladeOptimizer,
    BEMTRotorComp,
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
    "BSplineCurve",
    "Airfoil3D",
    "RotorGeometry",
    "OperatingCondition",
    "SolverConfig",
    "BEMTSensitivity",
    "BEMTResult",
    "BEMTSolver",
    "KulfanCSTAirfoil",
    "create_3d_airfoil",
    "create_airfoil_from_neuralfoil",
    "create_section_airfoils_from_neuralfoil",
    "build_spanwise_section_weight_matrix",
    "compute_element_polar_sensitivities",
    "compute_cst_aerodynamic_gradients",
    "build_planform_and_twist_distribution_constraints",
    "BladeOptimizationResult",
    "RotorBladeOptimizer",
    "HoverCruiseOptimizationResult",
    "HoverCruiseBladeOptimizer",
    "OperatingPointSpec",
    "ObjectiveTermSpec",
    "MetricConstraintSpec",
    "MultiPointOptimizationResult",
    "MultiPointBladeOptimizer",
    "BEMTRotorComp",
    "build_clamped_uniform_knots",
    "build_greville_abscissae",
    "build_bspline_basis_matrix",
    "build_parametric_bspline_basis_matrix",
    "set_num_threads",
    "get_max_threads",
]
