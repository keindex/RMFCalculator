# ============================================================================
# 声速计算模块
# ============================================================================
#
# 声速是中子星 EOS 的重要性质, 由热力学关系定义:
# $$C_s^2 = \left(\frac{\partial P}{\partial \varepsilon}\right)_S$$
#
# 其中:
# - $P$: 压力
# - $\varepsilon$: 能量密度
# - $S$: 熵 (对于理想流体, 在声波传播过程中保持恒定)
#
# 在相对论框架下, 声速满足因果极限:
# $$0 \leq C_s^2 \leq 1$$
# (即 $c = 1$ 单位下, 声速不能超过光速)
#
# ============================================================================
from RMFCalculator.tov.constant import c, G
from RMFCalculator.tov.unit import g_cm_3, dyn_cm_2


def speed_of_sound_calc(density, pressure):
    """Function that calculates the speed of sound by taking the gradient of the equation of state.

    声速通过 EOS 的数值梯度计算:
    $$C_s^2 = \frac{dP}{d\varepsilon} \approx \frac{\Delta P}{\Delta \varepsilon} = \frac{P_{i+1} - P_i}{\varepsilon_{i+1} - \varepsilon_i}$$

    Args:
        density (array): numpy 1Darray. 能量密度 (cgs 单位)
        pressure (array): numpy 1Darray. 压力 (cgs 单位)

    Returns:
        speed_of_sound (array): numpy 1Darray. 声速平方 $C_s^2$
        d2 (array): 对应的能量密度
    """
    density = density / c**2 * G / g_cm_3
    pressure = pressure / c**4 * G / dyn_cm_2

    drho = density[1:] - density[:-1]
    dp = pressure[1:] - pressure[:-1]
    speed_of_sound = dp / drho

    mask = density[:-1] > 1.5e-14
    d2 = density[:-1][mask]
    C_s = speed_of_sound[mask]

    return C_s, d2
