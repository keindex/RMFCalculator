r"""
crust_EOS.py - 中子星壳层 EOS 与核区 EOS 的拼接

中子星由外到内分为外壳 (outer crust)、内壳 (inner crust) 和核区 (core)。
壳层 EOS 由原子核物理 (如 Tolos 等人计算) 给出, 核区 EOS 由 RMF 模型给出。
两者之间通过一段多方态方程 (polytrope) 平滑过渡, 避免数值不连续。
"""

import numpy as np
from RMFCalculator.tov.unit import g_cm_3, dyn_cm_2


def PolyInterpolate(eps_crust, pressure_crust):
    r"""
    用一段多方态方程 (polytrope) 连接壳层 EOS 与核区 EOS。

    在壳层高密度端之上, 插入经验多方关系:

    $$P(\varepsilon) = a + b\,\varepsilon^{4/3}$$

    其中 $\varepsilon$ 为能量密度, $P$ 为压强。指数 $4/3$ 对应
    超相对论费米气体的多方指数 $\Gamma=4/3$ ($P\propto\rho^\Gamma$
    在该极限下与能量密度的关系)。系数 $a=4.75764\times10^{29}$,
    $b=9.04238\times10^{13}$ 由壳层高密度端与核区低密度端拟合得到。

    过渡密度范围: $\varepsilon \in [10^{11.7}, 10^{14.28}]$ g/cm$^3$,
    覆盖从内壳到核区过渡的典型密度区间。

    Args:
        eps_crust (array): 壳层能量密度 $\varepsilon_\text{crust}$, 单位: g·cm$^{-3}$
        pressure_crust (array): 壳层压强 $P_\text{crust}$, 单位: dyn·cm$^{-2}$

    Returns:
        tuple: $(\varepsilon_\text{combine},\, P_\text{combine})$
            壳层数据与多方过渡段拼接后的能量密度与压强
    """
    # 在对数空间生成过渡段能量密度网格: $10^{11.7}$ 到 $10^{14.28}$ g/cm³
    xs_polytro = np.logspace(11.7, 14.28, num=1000, base=10)
    # 多方关系: $P = a + b \varepsilon^{4/3}$, 单位 dyn/cm²
    ys_polytro = 4.75764e29 + 9.04238e13 * xs_polytro ** (4 / 3)

    # 单位转换: g/cm³ → MeV/fm³, dyn/cm² → MeV/fm³
    xs_polytropic = xs_polytro * g_cm_3
    ys_polytropic = ys_polytro * dyn_cm_2

    # 拼接壳层数据与多方过渡段
    eps_combine = np.append(eps_crust, xs_polytropic)
    pres_combine = np.append(pressure_crust, ys_polytropic)
    return eps_combine, pres_combine
