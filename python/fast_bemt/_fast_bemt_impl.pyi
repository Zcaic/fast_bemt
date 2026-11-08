from __future__ import annotations
import enum
from typing import Sequence, Tuple
import numpy as np
from numpy.typing import NDArray


class WakeModel(enum.Enum):
    """
    斜入流诱导速度与尾迹模型枚举

    Attributes
    ----------
    GlauertPittPeters : int (0)
        方案 A（主力推荐）：圆环 Glauert 动量平衡 + Pitt-Peters 斜尾迹一阶谐波修正
    GlauertAnnulus : int (1)
        方案 A 退化特例：纯圆环 Glauert 斜入流动量平衡（无周向谐波梯度 kx=ky=0）
    LocalElement : int (2)
        方案 B：局部微元 (r_i, psi_j) 独立动量闭环求解
    """

    GlauertPittPeters = 0
    GlauertAnnulus = 1
    LocalElement = 2


GlauertPittPeters: WakeModel
GlauertAnnulus: WakeModel
LocalElement: WakeModel


class Airfoil3D:
    """
    三维翼型气动极曲线保单调三次 Hermite 插值器: (alpha, Re, Ma) -> (Cl, Cd)
    内置 Viterna-Corrigan 全迎角 [-pi, pi] 平滑外推与自动微分支持。
    """

    def __init__(
        self,
        alpha_grid: NDArray[np.float64],
        re_grid: NDArray[np.float64],
        ma_grid: NDArray[np.float64],
        cl_table: NDArray[np.float64],
        cd_table: NDArray[np.float64],
        aspect_ratio: float = 10.0,
        enable_extrapolation: bool = True,
    ) -> None:
        """
        构造三维翼型极曲线插值表。

        Parameters
        ----------
        alpha_grid : NDArray[np.float64], shape (N_a,)
            迎角网格一维向量 [rad]，必须严格递增，长度 N_a >= 2
        re_grid : NDArray[np.float64], shape (N_re,)
            雷诺数网格一维向量 [-]，必须严格递增 (>0)，长度 N_re >= 1
        ma_grid : NDArray[np.float64], shape (N_ma,)
            马赫数网格一维向量 [-]，必须严格递增 (>=0)，长度 N_ma >= 1
        cl_table : NDArray[np.float64], shape (N_a * N_re * N_ma,)
            按行优先 C-order (N_a, N_re, N_ma) 展平的一维升力系数数组
        cd_table : NDArray[np.float64], shape (N_a * N_re * N_ma,)
            按行优先 C-order (N_a, N_re, N_ma) 展平的一维阻力系数数组
        aspect_ratio : float, default=10.0
            用于 Viterna-Corrigan 深失速外推的参考展弦比
        enable_extrapolation : bool, default=True
            是否开启超出网格范围时的 [-pi, pi] 全迎角平滑外推
        """
        ...

    @staticmethod
    def from_1d_polar(
        alpha_grid: NDArray[np.float64],
        cl_1d: NDArray[np.float64],
        cd_1d: NDArray[np.float64],
        ref_re: float = 1e6,
        aspect_ratio: float = 10.0,
        enable_extrapolation: bool = True,
    ) -> Airfoil3D:
        """
        由一维极曲线 (alpha [rad], Cl, Cd) 快速构建 Airfoil3D 对象。

        Parameters
        ----------
        alpha_grid : NDArray[np.float64], shape (N_a,)
            迎角网格向量 [rad]
        cl_1d : NDArray[np.float64], shape (N_a,)
            一维升力系数数组 [-]
        cd_1d : NDArray[np.float64], shape (N_a,)
            一维阻力系数数组 [-]
        ref_re : float, default=1e6
            参考雷诺数 [-]
        aspect_ratio : float, default=10.0
            参考展弦比（用于 Viterna-Corrigan 大迎角外推）
        enable_extrapolation : bool, default=True
            是否启用全迎角 [-pi, pi] 平滑外推
        """
        ...

    def eval(self, alpha: float, re: float = 1e6, ma: float = 0.0) -> Tuple[float, float]:
        """
        计算单点 (alpha, re, ma) 处的气动系数。

        Parameters
        ----------
        alpha : float
            局部有效迎角 [rad]
        re : float, default=1e6
            局部雷诺数 [-]
        ma : float, default=0.0
            局部马赫数 [-]

        Returns
        -------
        Tuple[float, float]
            (Cl, Cd) 升力系数与阻力系数
        """
        ...

    def eval_batch(
        self,
        alpha_vec: NDArray[np.float64],
        re_vec: NDArray[np.float64],
        ma_vec: NDArray[np.float64],
    ) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
        """
        批量向量化计算一组 (alpha_vec, re_vec, ma_vec) 对应的气动系数。

        Parameters
        ----------
        alpha_vec : NDArray[np.float64], shape (N,)
            迎角向量 [rad]
        re_vec : NDArray[np.float64], shape (N,)
            雷诺数向量 [-]
        ma_vec : NDArray[np.float64], shape (N,)
            马赫数向量 [-]

        Returns
        -------
        Tuple[NDArray[np.float64], NDArray[np.float64]]
            (cl_array, cd_array)，形状均为 (N,)
        """
        ...

    @property
    def zero_lift_alpha(self) -> float:
        """零升迎角 alpha_0 [rad]"""
        ...

    @property
    def linear_lift_slope(self) -> float:
        """小迎角线性段升力线斜率 dCl/dalpha [1/rad]"""
        ...

    @property
    def alpha_min(self) -> float:
        """输入表格最小迎角边界 [rad]"""
        ...

    @property
    def alpha_max(self) -> float:
        """输入表格最大迎角边界 [rad]"""
        ...


class RotorGeometry:
    """
    旋翼/螺旋桨几何定义类（支持每个径向站位 1对1 绑定独立的三维翼型表）
    """

    num_blades: int
    """桨叶数 B [-]"""
    radius: float
    """旋翼叶尖半径 R [m]"""
    hub_radius: float
    """桨毂半径 R_hub [m]"""
    spin_dir: int
    """旋向：+1 为逆时针 (CCW，从上往下看)，-1 为顺时针 (CW)"""
    r: NDArray[np.float64]
    """径向叶素站位坐标向量 [m]，形状 (N_r,)"""
    dr: NDArray[np.float64]
    """径向叶素微元积分宽度向量 [m]，形状 (N_r,)"""
    chord: NDArray[np.float64]
    """展向弦长分布向量 [m]，形状 (N_r,)"""
    twist: NDArray[np.float64]
    """展向几何扭转角分布向量 [rad]，形状 (N_r,)"""
    airfoils: list[Airfoil3D]
    """长度为 N_r 的 Airfoil3D 翼型表列表（与径向站位一一对应）"""

    def __init__(
        self,
        num_blades: int = 2,
        radius: float = 1.0,
        hub_radius: float = 0.1,
        r: NDArray[np.float64] = ...,
        chord: NDArray[np.float64] = ...,
        twist: NDArray[np.float64] = ...,
        airfoils: Sequence[Airfoil3D] = ...,
        spin_dir: int = 1,
        dr: NDArray[np.float64] = ...,
    ) -> None:
        """
        构造旋翼/螺旋桨几何对象。

        Parameters
        ----------
        num_blades : int
            桨叶数量 B (例如 2, 3, 4)
        radius : float
            旋翼叶尖半径 R [m]
        hub_radius : float
            旋翼桨毂根部半径 R_hub [m] (需满足 0 <= hub_radius < radius)
        r : NDArray[np.float64], shape (N_r,)
            各径向叶素站位的径向坐标数组 [m]，长度 N_r >= 2
        chord : NDArray[np.float64], shape (N_r,)
            各径向叶素站位对应的弦长数组 [m]，长度 N_r
        twist : NDArray[np.float64], shape (N_r,)
            各径向叶素站位对应的几何扭转角数组 [rad]，长度 N_r
        airfoils : Sequence[Airfoil3D], length N_r
            与 N_r 个径向站位一一对应的 Airfoil3D 翼型插值对象列表（共 N_r 个表）
        spin_dir : int, default=1
            旋转方向：+1 代表逆时针旋转 (CCW)，-1 代表顺时针旋转 (CW)
        dr : NDArray[np.float64], shape (N_r,), optional
            各径向叶素的微元积分宽度 [m]。若不传入或传入空数组，将自动按控制体边界生成保守恒积分宽度
        """
        ...

    def validate_and_finalize(self) -> None:
        """校验几何数组长度一致性并自动补全径向微元宽度 dr"""
        ...


class OperatingCondition:
    """
    旋翼运行工况与大气环境参数（定义在桨盘坐标系 Rotor Disk Frame 下）
    """

    vx: float
    """桨盘面内 X 向来流速度（指向下游后行边 psi=0 方向为正）[m/s]"""
    vy: float
    """桨盘面内 Y 向来流速度（指向前行边 psi=90 deg 方向为正）[m/s]"""
    vz: float
    """垂直桨盘轴向来流速度（逆推力方向/垂直爬升来流为正，下降流为负）[m/s]"""
    omega: float
    """旋翼旋转角速度 [rad/s] (> 0)"""
    pitch_0: float
    """总距操纵角 theta_0 [rad]"""
    pitch_1c: float
    """横向周期变距角 theta_1c (cos psi 项) [rad]"""
    pitch_1s: float
    """纵向周期变距角 theta_1s (sin psi 项) [rad]"""
    rho: float
    """空气密度 [kg/m^3]，默认 1.225"""
    mu: float
    """空气动力粘度 [Pa·s]，默认 1.7894e-5"""
    a_sound: float
    """当地声速 [m/s]，默认 340.29"""

    def __init__(
        self,
        vx: float = 0.0,
        vy: float = 0.0,
        vz: float = 0.0,
        omega: float = 100.0,
        pitch_0: float = 0.0,
        pitch_1c: float = 0.0,
        pitch_1s: float = 0.0,
        rho: float = 1.225,
        mu: float = 1.7894e-5,
        a_sound: float = 340.29,
    ) -> None:
        """
        直接由桨盘坐标系三轴速度分量 (vx, vy, vz) 与操纵参数构造运行工况。

        Parameters
        ----------
        vx : float, default=0.0
            桨盘平面内 X 向来流速度（顺风向/指向 psi=0 为正）[m/s]
        vy : float, default=0.0
            桨盘平面内 Y 向来流速度（侧滑向/指向 psi=90 deg 为正）[m/s]
        vz : float, default=0.0
            垂直桨盘轴向来流速度（轴向爬升/正迎面来流为正，垂直下降为负）[m/s]
        omega : float, default=100.0
            旋转角速度 [rad/s]
        pitch_0 : float, default=0.0
            桨叶总距角 theta_0 [rad]
        pitch_1c : float, default=0.0
            横向周期变距角 theta_1c (cos psi 项) [rad]
        pitch_1s : float, default=0.0
            纵向周期变距角 theta_1s (sin psi 项) [rad]
        rho : float, default=1.225
            空气密度 [kg/m^3]
        mu : float, default=1.7894e-5
            空气动力粘度 [Pa·s]
        a_sound : float, default=340.29
            声速 [m/s]
        """
        ...

    @staticmethod
    def from_freestream(
        v_inf: float,
        alpha_disk_rad: float,
        omega_rad_s: float,
        pitch_0_rad: float = 0.0,
        pitch_1c_rad: float = 0.0,
        pitch_1s_rad: float = 0.0,
        beta_rad: float = 0.0,
        rho: float = 1.225,
        mu: float = 1.7894e-5,
        a_sound: float = 340.29,
    ) -> OperatingCondition:
        """
        由远前方来流合速度、桨盘倾角及侧滑角构造运行工况。

        Parameters
        ----------
        v_inf : float
            远前方来流合速度大小 [m/s]
        alpha_disk_rad : float
            来流与旋翼旋转轴的夹角 [rad]（0 代表纯轴流垂直上升/平飞螺旋桨，pi/2 代表纯边飞）
        omega_rad_s : float
            旋转角速度 [rad/s]
        pitch_0_rad : float, default=0.0
            总距角 theta_0 [rad]
        beta_rad : float, default=0.0
            面内来流侧滑偏角 [rad]
        rho : float, default=1.225
            空气密度 [kg/m^3]
        mu : float, default=1.7894e-5
            空气动力粘度 [Pa·s]
        a_sound : float, default=340.29
            声速 [m/s]

        Returns
        -------
        OperatingCondition
            转换后的桨盘系运行工况对象
        """
        ...


class SolverConfig:
    """
    BEMT 求解器网格、尾迹模型与三大物理修正开关配置
    """

    n_azimuth: int
    """周向方位角离散网格数 N_psi（纯轴流可设为 1，斜入流推荐 36 ~ 72）"""
    wake_model: WakeModel
    """尾迹诱导速度模型（默认 WakeModel.GlauertPittPeters 方案 A）"""
    enable_tip_loss: bool
    """开关 1：是否启用 Prandtl 叶尖与叶根损失修正"""
    enable_vrs_correction: bool
    """开关 2：是否启用 Buhl / Glauert 高诱导速度区（涡环/风车态）平滑修正"""
    enable_stall_delay: bool
    """开关 3：是否启用 Du-Selig (1998) 三维旋转失速延迟修正"""
    enable_tangential_induction: bool
    """是否包含切向诱导速度 vt（尾流旋转效应）"""
    max_newton_iters: int
    """内层阻尼牛顿法最大迭代步数，默认 15"""
    max_brent_iters: int
    """标量 Brent 兜底求根最大迭代步数，默认 60"""
    max_wake_iters: int
    """外层 Pitt-Peters 尾迹偏斜角自洽迭代步数，默认 4"""
    tol: float
    """残差收敛容差，默认 1e-10"""
    newton_max_step: float
    """牛顿法单步最大角度更新限幅 [rad]，默认 0.25"""

    def __init__(
        self,
        n_azimuth: int = 36,
        wake_model: WakeModel = WakeModel.GlauertPittPeters,
        enable_tip_loss: bool = True,
        enable_vrs_correction: bool = True,
        enable_stall_delay: bool = True,
        enable_tangential_induction: bool = True,
        max_newton_iters: int = 15,
        max_brent_iters: int = 60,
        max_wake_iters: int = 4,
        tol: float = 1e-10,
    ) -> None:
        """
        配置 BEMT 求解器参数与物理修正开关。

        Parameters
        ----------
        n_azimuth : int, default=36
            周向方位角网格数 N_psi（纯轴流工况可设为 1，斜入流建议 36 或 72）
        wake_model : WakeModel, default=WakeModel.GlauertPittPeters
            诱导速度模型选择（GlauertPittPeters / GlauertAnnulus / LocalElement）
        enable_tip_loss : bool, default=True
            是否开启 Prandtl 叶尖与叶根损失修正
        enable_vrs_correction : bool, default=True
            是否开启 Buhl 高诱导速度区（涡环/风车状态）经验修正
        enable_stall_delay : bool, default=True
            是否开启 Du-Selig 三维旋转失速延迟修正
        enable_tangential_induction : bool, default=True
            是否求解切向诱导速度 vt
        max_newton_iters : int, default=15
            内层向量化牛顿法最大迭代次数
        max_brent_iters : int, default=60
            未收敛微元的 Brent 算法兜底最大迭代次数
        max_wake_iters : int, default=4
            外层 Pitt-Peters 尾迹偏斜角 chi 自洽迭代次数
        tol : float, default=1e-10
            非线性残差收敛阈值
        """
        ...


class BEMTSensitivity:
    """基于隐函数定理 (IFT) 与前向自动微分计算的精确解析敏度/梯度"""

    @property
    def valid(self) -> bool:
        """当前敏度数据是否已计算并有效"""
        ...

    @property
    def dT_dchord(self) -> NDArray[np.float64]:
        """总拉力 T 对各径向站位弦长 chord 的梯度向量 [N/m]，形状 (N_r,)"""
        ...

    @property
    def dT_dtwist(self) -> NDArray[np.float64]:
        """总拉力 T 对各径向站位扭转角 twist 的梯度向量 [N/rad]，形状 (N_r,)"""
        ...

    @property
    def dQ_dchord(self) -> NDArray[np.float64]:
        """总扭矩 Q 对各径向站位弦长 chord 的梯度向量 [N·m/m]，形状 (N_r,)"""
        ...

    @property
    def dQ_dtwist(self) -> NDArray[np.float64]:
        """总扭矩 Q 对各径向站位扭转角 twist 的梯度向量 [N·m/rad]，形状 (N_r,)"""
        ...

    @property
    def dP_dchord(self) -> NDArray[np.float64]:
        """总功率 P 对各径向站位弦长 chord 的梯度向量 [W/m]，形状 (N_r,)"""
        ...

    @property
    def dP_dtwist(self) -> NDArray[np.float64]:
        """总功率 P 对各径向站位扭转角 twist 的梯度向量 [W/rad]，形状 (N_r,)"""
        ...

    @property
    def dMx_dchord(self) -> NDArray[np.float64]:
        """滚转力矩 Mx 对弦长 chord 的梯度向量 [N·m/m]，形状 (N_r,)"""
        ...

    @property
    def dMx_dtwist(self) -> NDArray[np.float64]:
        """滚转力矩 Mx 对扭转角 twist 的梯度向量 [N·m/rad]，形状 (N_r,)"""
        ...

    @property
    def dMy_dchord(self) -> NDArray[np.float64]:
        """俯仰力矩 My 对弦长 chord 的梯度向量 [N·m/m]，形状 (N_r,)"""
        ...

    @property
    def dMy_dtwist(self) -> NDArray[np.float64]:
        """俯仰力矩 My 对扭转角 twist 的梯度向量 [N·m/rad]，形状 (N_r,)"""
        ...

    @property
    def dForces_dCond(self) -> NDArray[np.float64]:
        """
        7 个全局输出量 [0:T, 1:Q, 2:P, 3:Fx, 4:Fy, 5:Mx, 6:My] 对
        5 个工况参数 [0:omega, 1:pitch_0, 2:vx, 3:vy, 4:vz] 的雅可比矩阵，形状 (7, 5)
        """
        ...

    @property
    def dTr_dchord_diag(self) -> NDArray[np.float64]:
        """径向平均拉力分布 dT/dr 对当地弦长 chord 的对角导数向量 [N/m^2]，形状 (N_r,)"""
        ...

    @property
    def dTr_dtwist_diag(self) -> NDArray[np.float64]:
        """径向平均拉力分布 dT/dr 对当地扭转角 twist 的对角导数向量 [N/(m·rad)]，形状 (N_r,)"""
        ...


class BEMTResult:
    """BEMT 完整求解输出数据结构（包含集总气动力、2D 桨盘流场矩阵及解析敏度）"""

    @property
    def converged(self) -> bool:
        """所有微元是否均收敛"""
        ...

    @property
    def total_fallback_cells(self) -> int:
        """触发 Brent 兜底求根的微元个数统计"""
        ...

    @property
    def thrust(self) -> float:
        """轴向总拉力 T (= Fz) [N]"""
        ...

    @property
    def torque(self) -> float:
        """旋转轴反扭矩 Q (= Mz) [N·m]"""
        ...

    @property
    def power(self) -> float:
        """消耗轴功率 P = Q * omega [W]"""
        ...

    @property
    def fx(self) -> float:
        """桨盘面内后向力 Fx (H-force) [N]"""
        ...

    @property
    def fy(self) -> float:
        """桨盘面内侧向力 Fy (Y-force) [N]"""
        ...

    @property
    def fz(self) -> float:
        """桨盘轴向推力 Fz (= T) [N]"""
        ...

    @property
    def mx(self) -> float:
        """桨盘滚转力矩 Mx (Roll moment) [N·m]"""
        ...

    @property
    def my(self) -> float:
        """桨盘俯仰力矩 My (Pitch moment) [N·m]"""
        ...

    @property
    def mz(self) -> float:
        """桨盘偏航/扭矩力矩 Mz (= Q) [N·m]"""
        ...

    @property
    def ct(self) -> float:
        """拉力系数 C_T = T / (rho * pi * R^2 * V_tip^2)"""
        ...

    @property
    def cq(self) -> float:
        """扭矩系数 C_Q = Q / (rho * pi * R^2 * V_tip^2 * R)"""
        ...

    @property
    def cp(self) -> float:
        """功率系数 C_P = P / (rho * pi * R^2 * V_tip^3)"""
        ...

    @property
    def fm(self) -> float:
        """悬停品质因数 FM = C_T^(1.5) / (sqrt(2) * C_P)"""
        ...

    @property
    def vi_disk_mean(self) -> float:
        """桨盘面积加权平均轴向诱导速度 [m/s]"""
        ...

    @property
    def wake_skew_chi(self) -> float:
        """尾迹偏斜角 chi [rad]"""
        ...

    @property
    def pitt_kx(self) -> float:
        """Pitt-Peters 纵向一阶谐波入流梯度系数 kx"""
        ...

    @property
    def pitt_ky(self) -> float:
        """Pitt-Peters 横向一阶谐波入流梯度系数 ky"""
        ...

    @property
    def psi_grid(self) -> NDArray[np.float64]:
        """方位角离散网格向量 [rad]，形状 (N_psi,)"""
        ...

    @property
    def vi_annulus(self) -> NDArray[np.float64]:
        """各径向圆环周向平均诱导速度 v_i0(r) [m/s]，形状 (N_r,)"""
        ...

    @property
    def dT_dr_mean(self) -> NDArray[np.float64]:
        """周向平均径向拉力分布 dT/dr [N/m]，形状 (N_r,)"""
        ...

    @property
    def dQ_dr_mean(self) -> NDArray[np.float64]:
        """周向平均径向扭矩分布 dQ/dr [N·m/m]，形状 (N_r,)"""
        ...

    @property
    def phi(self) -> NDArray[np.float64]:
        """二维局部入流角分布矩阵 [rad]，形状 (N_r, N_psi)"""
        ...

    @property
    def alpha(self) -> NDArray[np.float64]:
        """二维局部有效迎角分布矩阵 [rad]，形状 (N_r, N_psi)"""
        ...

    @property
    def vi(self) -> NDArray[np.float64]:
        """二维局部轴向诱导速度矩阵 [m/s]，形状 (N_r, N_psi)"""
        ...

    @property
    def vt(self) -> NDArray[np.float64]:
        """二维局部切向诱导速度矩阵 [m/s]，形状 (N_r, N_psi)"""
        ...

    @property
    def cl(self) -> NDArray[np.float64]:
        """二维局部升力系数矩阵（含三维失速延迟修正）[-]，形状 (N_r, N_psi)"""
        ...

    @property
    def cd(self) -> NDArray[np.float64]:
        """二维局部阻力系数矩阵（含三维失速延迟修正）[-]，形状 (N_r, N_psi)"""
        ...

    @property
    def re(self) -> NDArray[np.float64]:
        """二维局部雷诺数矩阵 [-]，形状 (N_r, N_psi)"""
        ...

    @property
    def ma(self) -> NDArray[np.float64]:
        """二维局部马赫数矩阵 [-]，形状 (N_r, N_psi)"""
        ...

    @property
    def f_tip(self) -> NDArray[np.float64]:
        """二维局部 Prandtl 叶尖/叶根损失系数矩阵 F [-]，形状 (N_r, N_psi)"""
        ...

    @property
    def df_z(self) -> NDArray[np.float64]:
        """单桨叶在 (r_i, psi_j) 处的单位展长轴向推力 dFz/dr [N/m]，形状 (N_r, N_psi)"""
        ...

    @property
    def df_theta(self) -> NDArray[np.float64]:
        """单桨叶在 (r_i, psi_j) 处的单位展长周向阻力 dF_theta/dr [N/m]，形状 (N_r, N_psi)"""
        ...

    @property
    def sensitivity(self) -> BEMTSensitivity:
        """解析敏度/雅可比矩阵结果（当 solve(..., compute_gradients=True) 时填充）"""
        ...


class BEMTSolver:
    """高效矩阵向量化 + OpenMP 多核并行斜入流 BEMT 核心求解器"""

    def __init__(
        self, geom: RotorGeometry, cfg: SolverConfig = ...
    ) -> None:
        """
        初始化 BEMT 求解器。

        Parameters
        ----------
        geom : RotorGeometry
            旋翼/螺旋桨几何对象（含展向站位与独立三维翼型表）
        cfg : SolverConfig, optional
            求解器网格、尾迹模型与三大修正开关配置（若不传则使用默认配置）
        """
        ...

    def solve(
        self, cond: OperatingCondition, compute_gradients: bool = False
    ) -> BEMTResult:
        """
        求解单个运行工况。

        Parameters
        ----------
        cond : OperatingCondition
            当前飞行/悬停运行工况
        compute_gradients : bool, default=False
            若设为 True，将在收敛后利用隐函数定理 (IFT) + Dual<8> 一次性算出所有气动力对弦长、扭转角和工况参数的解析导数

        Returns
        -------
        BEMTResult
            包含全盘气动力、力矩、二维流场分布及解析梯度的结果对象
        """
        ...

    def solve_batch(
        self, conds: Sequence[OperatingCondition], compute_gradients: bool = False
    ) -> list[BEMTResult]:
        """
        利用 OpenMP 多线程并行求解一组运行工况（用于包线扫描或多点气动优化）。

        Parameters
        ----------
        conds : Sequence[OperatingCondition]
            运行工况列表
        compute_gradients : bool, default=False
            是否为每个工况同时计算解析梯度

        Returns
        -------
        list[BEMTResult]
            与输入工况列表一一对应的求解结果列表
        """
        ...

    @property
    def geometry(self) -> RotorGeometry:
        """绑定的旋翼几何对象"""
        ...

    @property
    def config(self) -> SolverConfig:
        """绑定的求解器配置对象"""
        ...