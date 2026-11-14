#pragma once

#include "bemt/config.hpp"
#include "bemt/solver.hpp"

namespace bemt {

/**
 * @brief 基于隐函数定理 (Implicit Function Theorem, IFT) 与前向自动微分 Dual<9>
 *        计算全桨盘气动力/力矩/功率对展向几何分布 (chord, twist) 及全局工况参数的解析梯度。
 *
 * @param solver 已配置好几何与修正开关的 BEMT 求解器引用
 * @param cond   当前运行工况
 * @param res    已完成前向收敛求解的 BEMTResult（函数内部将计算并填充 res.sensitivity）
 */
void compute_ift_sensitivities(const BEMTSolver& solver,
                               const OperatingCondition& cond,
                               BEMTResult& res);

} // namespace bemt