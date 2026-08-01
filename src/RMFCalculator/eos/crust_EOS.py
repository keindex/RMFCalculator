import numpy as np
from RMFCalculator.tov.unit import g_cm_3, dyn_cm_2


def PolyInterpolate(eps_crust, pressure_crust):
    r"""
    用一段多方态方程 (polytrope) 连接壳层 EOS 与核区 EOS。

    在壳层高密度端之上, 插入经验多方关系:

    $P(\varepsilon) = a + b\,\varepsilon^{4/3}$

    其中 $\varepsilon$ 为能量密度, $P$ 为压强。指数 $4/3$ 对应
    超相对论费米气体的多方指数 $\Gamma=4/3$ ($P\propto\rho^\Gamma$
    在该极限下与能量密度的关系)。

    Args:
        eps_crust (array): 壳层能量密度 $\varepsilon_\text{crust}$, 单位: g·cm$^{-3}$
        pressure_crust (array): 壳层压强 $P_\text{crust}$, 单位: dyn·cm$^{-2}$

    Returns:
        tuple: $(\varepsilon_\text{combine},\, P_\text{combine})$
            壳层数据与多方过渡段拼接后的能量密度与压强
    """
    xs_polytro = np.logspace(11.7, 14.28, num=1000, base=10)
    ys_polytro = 4.75764e29 + 9.04238e13 * xs_polytro ** (4 / 3)

    xs_polytropic = xs_polytro * g_cm_3
    ys_polytropic = ys_polytro * dyn_cm_2

    eps_combine = np.append(eps_crust, xs_polytropic)
    pres_combine = np.append(pressure_crust, ys_polytropic)
    return eps_combine, pres_combine
