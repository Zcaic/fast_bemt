#pragma once

#include "bemt/types.hpp"
#include <algorithm>
#include <cmath>
#include <iostream>

namespace bemt {

// ============================================================================
// 前向模式自动微分双数类 Dual<N>
// val : 函数值 f(x)
// d   : 梯度向量 [df/dx_0, df/dx_1, ..., df/dx_{N-1}]
// ============================================================================
template <int N = 1>
struct Dual {
    using DerivArray = Eigen::Array<double, N, 1>;

    double val;
    DerivArray d;

    // 默认构造：值和导数均为 0
    inline Dual() : val(0.0), d(DerivArray::Zero()) {}

    // 常数构造：值为 v，导数为 0
    inline Dual(double v) : val(v), d(DerivArray::Zero()) {}

    // 单一自变量种子构造：值为 v，第 idx 个导数分量为 1.0（或指定 seed）
    inline Dual(double v, int idx, double seed = 1.0) : val(v), d(DerivArray::Zero()) {
        d[idx] = seed;
    }

    // 完整构造：直接给定值和导数数组
    template <typename Derived>
    inline Dual(double v, const Eigen::ArrayBase<Derived>& deriv) : val(v), d(deriv) {}

    // 针对 N=1 的便捷构造：Dual<1>(val, deriv_scalar)
    template <int M = N, typename = std::enable_if_t<M == 1>>
    inline Dual(double v, double d0) : val(v) {
        d[0] = d0;
    }

    // 便捷读取第 i 个偏导数
    inline double grad(int i = 0) const { return d[i]; }

    // ------------------------------------------------------------------------
    // 一元运算符与复合赋值运算符
    // ------------------------------------------------------------------------
    inline Dual operator+() const { return *this; }
    inline Dual operator-() const { return Dual(-val, -d); }

    inline Dual& operator+=(const Dual& rhs) {
        val += rhs.val;
        d += rhs.d;
        return *this;
    }
    inline Dual& operator+=(double rhs) {
        val += rhs;
        return *this;
    }

    inline Dual& operator-=(const Dual& rhs) {
        val -= rhs.val;
        d -= rhs.d;
        return *this;
    }
    inline Dual& operator-=(double rhs) {
        val -= rhs;
        return *this;
    }

    inline Dual& operator*=(const Dual& rhs) {
        d = d * rhs.val + val * rhs.d;
        val *= rhs.val;
        return *this;
    }
    inline Dual& operator*=(double rhs) {
        val *= rhs;
        d *= rhs;
        return *this;
    }

    inline Dual& operator/=(const Dual& rhs) {
        const double inv = 1.0 / rhs.val;
        const double inv2 = inv * inv;
        d = (d * rhs.val - val * rhs.d) * inv2;
        val *= inv;
        return *this;
    }
    inline Dual& operator/=(double rhs) {
        const double inv = 1.0 / rhs;
        val *= inv;
        d *= inv;
        return *this;
    }
};

// 常用维度别名
using Dual1 = Dual<1>; // 单变量求导（用于牛顿迭代求根 R(phi) = 0）
using Dual6 = Dual<6>; // 局部叶素对 (phi, chord, twist, Vx, Vy, Vz/Omega) 同時求偏导

// ============================================================================
// 二元算术运算符 (Dual op Dual, Dual op double, double op Dual)
// ============================================================================
template <int N>
inline Dual<N> operator+(const Dual<N>& a, const Dual<N>& b) {
    return Dual<N>(a.val + b.val, a.d + b.d);
}
template <int N>
inline Dual<N> operator+(const Dual<N>& a, double b) {
    return Dual<N>(a.val + b, a.d);
}
template <int N>
inline Dual<N> operator+(double a, const Dual<N>& b) {
    return Dual<N>(a + b.val, b.d);
}

template <int N>
inline Dual<N> operator-(const Dual<N>& a, const Dual<N>& b) {
    return Dual<N>(a.val - b.val, a.d - b.d);
}
template <int N>
inline Dual<N> operator-(const Dual<N>& a, double b) {
    return Dual<N>(a.val - b, a.d);
}
template <int N>
inline Dual<N> operator-(double a, const Dual<N>& b) {
    return Dual<N>(a - b.val, -b.d);
}

template <int N>
inline Dual<N> operator*(const Dual<N>& a, const Dual<N>& b) {
    return Dual<N>(a.val * b.val, a.d * b.val + a.val * b.d);
}
template <int N>
inline Dual<N> operator*(const Dual<N>& a, double b) {
    return Dual<N>(a.val * b, a.d * b);
}
template <int N>
inline Dual<N> operator*(double a, const Dual<N>& b) {
    return Dual<N>(a * b.val, a * b.d);
}

template <int N>
inline Dual<N> operator/(const Dual<N>& a, const Dual<N>& b) {
    const double inv = 1.0 / b.val;
    return Dual<N>(a.val * inv, (a.d * b.val - a.val * b.d) * (inv * inv));
}
template <int N>
inline Dual<N> operator/(const Dual<N>& a, double b) {
    const double inv = 1.0 / b;
    return Dual<N>(a.val * inv, a.d * inv);
}
template <int N>
inline Dual<N> operator/(double a, const Dual<N>& b) {
    const double inv = 1.0 / b.val;
    return Dual<N>(a * inv, (-a * inv * inv) * b.d);
}

// ============================================================================
// 比较运算符（仅比较实部 val，支持分支判断与区间裁剪）
// ============================================================================
template <int N> inline bool operator<(const Dual<N>& a, const Dual<N>& b)  { return a.val < b.val; }
template <int N> inline bool operator<(const Dual<N>& a, double b)          { return a.val < b; }
template <int N> inline bool operator<(double a, const Dual<N>& b)          { return a < b.val; }

template <int N> inline bool operator<=(const Dual<N>& a, const Dual<N>& b) { return a.val <= b.val; }
template <int N> inline bool operator<=(const Dual<N>& a, double b)         { return a.val <= b; }
template <int N> inline bool operator<=(double a, const Dual<N>& b)         { return a <= b.val; }

template <int N> inline bool operator>(const Dual<N>& a, const Dual<N>& b)  { return a.val > b.val; }
template <int N> inline bool operator>(const Dual<N>& a, double b)          { return a.val > b; }
template <int N> inline bool operator>(double a, const Dual<N>& b)          { return a > b.val; }

template <int N> inline bool operator>=(const Dual<N>& a, const Dual<N>& b) { return a.val >= b.val; }
template <int N> inline bool operator>=(const Dual<N>& a, double b)         { return a.val >= b; }
template <int N> inline bool operator>=(double a, const Dual<N>& b)         { return a >= b.val; }

template <int N> inline bool operator==(const Dual<N>& a, const Dual<N>& b) { return a.val == b.val; }
template <int N> inline bool operator!=(const Dual<N>& a, const Dual<N>& b) { return a.val != b.val; }

// ============================================================================
// 初等数学函数重载（支持 ADL 实参依赖查找，可直接 using std::sin; sin(x) 调用）
// ============================================================================
template <int N>
inline Dual<N> sin(const Dual<N>& x) {
    return Dual<N>(std::sin(x.val), std::cos(x.val) * x.d);
}

template <int N>
inline Dual<N> cos(const Dual<N>& x) {
    return Dual<N>(std::cos(x.val), -std::sin(x.val) * x.d);
}

template <int N>
inline Dual<N> tan(const Dual<N>& x) {
    const double c = std::cos(x.val);
    return Dual<N>(std::tan(x.val), x.d / (c * c));
}

template <int N>
inline Dual<N> asin(const Dual<N>& x) {
    // 边界正则化保护，防止 |x| -> 1 时导数变为 NaN
    const double arg = std::clamp(x.val, -1.0 + 1e-12, 1.0 - 1e-12);
    return Dual<N>(std::asin(arg), x.d / std::sqrt(1.0 - arg * arg));
}

template <int N>
inline Dual<N> acos(const Dual<N>& x) {
    // Prandtl 叶尖损失因子 F = (2/pi)*acos(exp(-f)) 核心调用，必须带边界防除零保护
    const double arg = std::clamp(x.val, -1.0 + 1e-12, 1.0 - 1e-12);
    return Dual<N>(std::acos(arg), -x.d / std::sqrt(1.0 - arg * arg));
}

template <int N>
inline Dual<N> atan(const Dual<N>& x) {
    return Dual<N>(std::atan(x.val), x.d / (1.0 + x.val * x.val));
}

template <int N>
inline Dual<N> atan2(const Dual<N>& y, const Dual<N>& x) {
    const double denom = std::max(x.val * x.val + y.val * y.val, 1e-24);
    return Dual<N>(std::atan2(y.val, x.val), (x.val * y.d - y.val * x.d) / denom);
}
template <int N>
inline Dual<N> atan2(const Dual<N>& y, double x) {
    const double denom = std::max(x * x + y.val * y.val, 1e-24);
    return Dual<N>(std::atan2(y.val, x), (x * y.d) / denom);
}
template <int N>
inline Dual<N> atan2(double y, const Dual<N>& x) {
    const double denom = std::max(x.val * x.val + y * y, 1e-24);
    return Dual<N>(std::atan2(y, x.val), (-y * x.d) / denom);
}

template <int N>
inline Dual<N> sqrt(const Dual<N>& x) {
    const double s = std::sqrt(std::max(x.val, 0.0));
    const double inv_2s = (s > 1e-15) ? (0.5 / s) : 0.0;
    return Dual<N>(s, inv_2s * x.d);
}

template <int N>
inline Dual<N> exp(const Dual<N>& x) {
    const double e = std::exp(x.val);
    return Dual<N>(e, e * x.d);
}

template <int N>
inline Dual<N> log(const Dual<N>& x) {
    const double safe_x = std::max(x.val, 1e-30);
    return Dual<N>(std::log(safe_x), x.d / safe_x);
}

template <int N>
inline Dual<N> log10(const Dual<N>& x) {
    constexpr double kInvLn10 = 0.43429448190325182765;
    const double safe_x = std::max(x.val, 1e-30);
    return Dual<N>(std::log10(safe_x), (kInvLn10 / safe_x) * x.d);
}

template <int N>
inline Dual<N> pow(const Dual<N>& x, double p) {
    const double v = std::pow(x.val, p);
    const double dv = (std::abs(x.val) > 1e-15 || p >= 1.0) ? (p * std::pow(x.val, p - 1.0)) : 0.0;
    return Dual<N>(v, dv * x.d);
}

template <int N>
inline Dual<N> abs(const Dual<N>& x) {
    return (x.val >= 0.0) ? x : -x;
}

template <int N>
inline Dual<N> fabs(const Dual<N>& x) {
    return abs(x);
}

template <int N>
inline bool isfinite(const Dual<N>& x) {
    return std::isfinite(x.val) && x.d.allFinite();
}

template <int N>
inline std::ostream& operator<<(std::ostream& os, const Dual<N>& x) {
    os << "Dual(" << x.val << ", d=[" << x.d.transpose() << "])";
    return os;
}

// 提取实部辅助函数（兼容 double 和 Dual<N> 泛型模板编程）
inline double val_of(double x) { return x; }
template <int N>
inline double val_of(const Dual<N>& x) { return x.val; }

} // namespace bemt

// ============================================================================
// Eigen::NumTraits 特化，允许直接构建 Eigen::Array<Dual<N>, ...>
// ============================================================================
namespace Eigen {
template <int N>
struct NumTraits<bemt::Dual<N>> : NumTraits<double> {
    using Real = bemt::Dual<N>;
    using NonInteger = bemt::Dual<N>;
    using Nested = bemt::Dual<N>;
    enum {
        IsComplex = 0,
        IsInteger = 0,
        IsSigned = 1,
        RequireInitialization = 1,
        ReadCost = 1 + N,
        AddCost = 1 + N,
        MulCost = 1 + 2 * N
    };
};
} // namespace Eigen