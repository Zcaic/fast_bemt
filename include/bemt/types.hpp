#pragma once

#include <Eigen/Dense>
#include <cmath>
#include <cstdint>
#include <memory>
#include <vector>

namespace bemt {

// ============================================================================
// 1. 数学与物理常数
// ============================================================================
constexpr double kPi      = 3.14159265358979323846;
constexpr double kTwoPi   = 6.28318530717958647692;
constexpr double kHalfPi  = 1.57079632679489661923;
constexpr double kDeg2Rad = kPi / 180.0;
constexpr double kRad2Deg = 180.0 / kPi;
constexpr double kEps     = 1e-12; // 通用防除零极小量

// ============================================================================
// 2. Eigen 向量与矩阵类型别名
// 关键设计：二维数组/矩阵显式指定 Eigen::RowMajor（行优先）
// 原因：1. 与 Python NumPy 默认 C-contiguous 内存布局 100% 匹配，支持 nanobind 零拷贝；
//      2. 第 i 行代表第 i 个径向站位 (r_i) 在所有方位角 (psi_j) 上的连续内存，
//         调用 airfoils[i] 向量化查表时具备完美的 L1 Cache 空间局部性！
// ============================================================================

// 一维列向量/数组（用于径向分布 Nr x 1 或周向分布 Npsi x 1）
using ArrayXd  = Eigen::ArrayXd;
using VectorXd = Eigen::VectorXd;
using ArrayXi  = Eigen::ArrayXi;
using ArrayXb  = Eigen::Array<bool, Eigen::Dynamic, 1>;

// 二维行优先数组/矩阵（维度通常为 Nr x Npsi）
using ArrayXXd  = Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
using MatrixXd  = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

// 零拷贝只读/读写视图引用（用于函数传参及 nanobind 接口映射）
using RefArrayXd       = Eigen::Ref<ArrayXd>;
using ConstRefArrayXd  = Eigen::Ref<const ArrayXd>;
using RefArrayXXd      = Eigen::Ref<ArrayXXd>;
using ConstRefArrayXXd = Eigen::Ref<const ArrayXXd>;
using RefMatrixXd      = Eigen::Ref<MatrixXd>;
using ConstRefMatrixXd = Eigen::Ref<const MatrixXd>;

// ============================================================================
// 3. 诱导速度与尾迹模型枚举
// ============================================================================
enum class WakeModel : int {
    kGlauertPittPeters = 0, // 方案 A（主力推荐）：圆环 Glauert 动量平衡 + Pitt-Peters 斜尾迹一阶谐波修正
    kGlauertAnnulus    = 1, // 方案 A 退化特例：纯圆环 Glauert 斜入流动量平衡（无周向谐波梯度 kx=ky=0）
    kLocalElement      = 2  // 方案 B：局部微元 (r_i, psi_j) 独立动量闭环求解
};

// ============================================================================
// 4. 可微平滑正则化辅助函数 (Smooth Regularization Functions)
// 说明：同时支持 double 和 Dual<N> 标量类型，消除绝对值、平方根、反余弦在边界处的导数奇异性
// ============================================================================
namespace math {

// 平滑绝对值: sqrt(x^2 + eps^2)，在 x = 0 处一阶、二阶导数光滑连续
template <typename Scalar>
inline Scalar smooth_abs(const Scalar& x, double eps = 1e-7) {
    using std::sqrt;
    return sqrt(x * x + eps * eps);
}

// 平滑最大值: max(a, b) 的 C^infty 光滑近似
template <typename Scalar>
inline Scalar smooth_max(const Scalar& a, const Scalar& b, double eps = 1e-7) {
    return 0.5 * (a + b + smooth_abs(a - b, eps));
}

// 平滑最小值: min(a, b) 的 C^infty 光滑近似
template <typename Scalar>
inline Scalar smooth_min(const Scalar& a, const Scalar& b, double eps = 1e-7) {
    return 0.5 * (a + b - smooth_abs(a - b, eps));
}

// 平滑裁剪函数: 将 x 平滑限制在 [lo, hi] 区间内
template <typename Scalar>
inline Scalar smooth_clamp(const Scalar& x, double lo, double hi, double eps = 1e-7) {
    return smooth_min(smooth_max(x, Scalar(lo), eps), Scalar(hi), eps);
}

// Sigmoid 平滑过渡权重函数: 在 x = x0 附近以宽度 width 从 0 平滑过渡到 1
template <typename Scalar>
inline Scalar sigmoid_weight(const Scalar& x, double x0, double width) {
    using std::exp;
    Scalar z = (x - x0) / width;
    // 防止极端大值导致 exp 溢出
    if (z > 40.0) return Scalar(1.0);
    if (z < -40.0) return Scalar(0.0);
    return Scalar(1.0) / (Scalar(1.0) + exp(-z));
}

} // namespace math

} // namespace bemt