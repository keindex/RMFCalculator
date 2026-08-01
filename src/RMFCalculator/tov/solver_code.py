# ============================================================================
# TOV (Tolman-Oppenheimer-Volkoff) 方程求解器核心
# ============================================================================
#
# 本模块实现了中子星结构的 TOV 方程数值积分
#
# TOV 方程组 (在几何单位 G=c=1 下):
#
#   (1) 压力梯度方程:
#       $$\frac{dP}{dr} = -\frac{(\varepsilon + P)(m + 4\pi r^3 P)}{r(r - 2m)}$$
#
#   (2) 质量累积方程:
#       $$\frac{dm}{dr} = 4\pi r^2 \varepsilon$$
#
#   (3) 表面边界条件:
#       当 $P(R) = P_{\text{min}}$ 时, $R$ 为半径, $m(R) = M$ 为总质量
#
# 潮汐形变扩展方程:
#   (4)
#       $$\frac{dh}{dr} = b$$
#   (5)
#       $$\frac{df}{dr} = \cdots$$
#       其中 $h$, $b$ 是用于计算潮汐 Love 数的辅助变量
#
# ============================================================================

# Python packages
import numpy as np
from scipy.constants import pi
from scipy.integrate import ode
from scipy.interpolate import interp1d
from RMFCalculator.tov import unit


def m1_from_mc_m2(mc, m2):
    """a function that feed back the companion star mass from GW event measurement.

    Args:
        mc (float): chrip mass of a GW event, unit in solar mass.
        m2 (float or numpy array): the determined mass for one of the star, this is computed from sampling of EoS.

    Returns:
        m1 (array): the companion star mass in solar mass.
    """

    m2 = np.array(m2)
    num1 = (2.0 / 3.0) ** (1.0 / 3.0) * mc**5.0
    denom1 = (
        9 * m2**7.0 * mc**5.0
        + np.sqrt(3.0)
        * np.sqrt(abs(27 * m2**14.0 * mc**10.0 - 4.0 * m2**9.0 * mc**15.0))
    ) ** (1.0 / 3.0)
    denom2 = 2.0 ** (1.0 / 3.0) * 3.0 ** (2.0 / 3.0) * m2**3.0
    return num1 / denom1 + denom1 / denom2


def pressure_adind(P, epsgrid, presgrid):
    """计算绝热指数 (Adiabatic Index)

    绝热指数定义为:
    $$\Gamma = \left(\frac{\partial \ln P}{\partial \ln \varepsilon}\right)_S
             = \frac{\varepsilon + P}{P} \frac{dP}{d\varepsilon}$$

    在多方 EOS 情况下 ($P = K\varepsilon^{\Gamma}$), $\Gamma$ 为常数。
    对于真实 EOS, 通过数值插值计算。

    Args:
        P (float): 当地压力
        epsgrid (array): 能量密度网格
        presgrid (array): 压力网格

    Returns:
        adind (float): 绝热指数
    """
    idx = np.searchsorted(presgrid, P)
    if idx == 0:
        eds = epsgrid[0] * np.power(P / presgrid[0], 3.0 / 5.0)
        adind = (
            5.0
            / 3.0
            * presgrid[0]
            * np.power(eds / epsgrid[0], 5.0 / 3.0)
            * 1.0
            / eds
            * (eds + P)
            / P
        )
    if idx == len(presgrid):
        eds = epsgrid[-1] * np.power(P / presgrid[-1], 3.0 / 5.0)
        adind = (
            5.0
            / 3.0
            * presgrid[-1]
            * np.power(eds / epsgrid[-1], 5.0 / 3.0)
            * 1.0
            / eds
            * (eds + P)
            / P
        )
    else:
        ci = np.log(presgrid[idx] / presgrid[idx - 1]) / np.log(
            epsgrid[idx] / epsgrid[idx - 1]
        )
        eds = epsgrid[idx - 1] * np.power(P / presgrid[idx - 1], 1.0 / ci)
        adind = (
            ci
            * presgrid[idx - 1]
            * np.power(eds / epsgrid[idx - 1], ci)
            * 1.0
            / eds
            * (eds + P)
            / P
        )
    return adind


def TOV(r, y, inveos):
    """TOV 方程组 (简化版, 无潮汐项)

    用于纯质量-半径计算:


    $$\frac{dP}{dr} = -\frac{(\varepsilon + P)(m + 4\pi r^3 P)}{r(r - 2m)}$$


    $$\frac{dm}{dr} = 4\pi r^2 \varepsilon$$

    Args:
        r (float): 径向坐标
        y (array): [pressure, mass] 状态向量
        inveos (function): 逆 EOS 函数 $P \rightarrow \varepsilon$

    Returns:
        array: [dpdr, dmdr] 导数向量
    """
    pres, m = y

    eps = inveos(pres)

    dpdr = -(eps + pres) * (m + 4.0 * pi * r**3.0 * pres)
    dpdr = dpdr / (r * (r - 2.0 * m))

    dmdr = 4.0 * pi * r**2.0 * eps

    return np.array([dpdr, dmdr])


def TOV_def(r, y, inveos, ad_index):
    """Packs the complete set of TOV equations for neutron stars with tidal deformability

    完整的 TOV + 潮汐形变方程组:

    (1) 压力梯度:
    $$\frac{dP}{dr} = -\frac{(\varepsilon + P)(m + 4\pi r^3 P)}{r(r - 2m)}$$

    (2) 质量梯度:
    $$\frac{dm}{dr} = 4\pi r^2 \varepsilon$$

    (3) 潮汐变量 h 梯度:
    $$\frac{dh}{dr} = b$$

    (4) 潮汐变量 b 梯度:
    $$\frac{df}{dr} = 2(1-2m/r)^{-1} h \left[ -2\pi(5\varepsilon + 9P + \frac{(\varepsilon+P)^2}{P\Gamma}) + \frac{3}{r^2} + 2(1-2m/r)^{-1}(\frac{m}{r^2} + 4\pi r P)^2 \right] + \frac{2b}{r}(1-2m/r)^{-1}[-1 + \frac{m}{r} + 2\pi r^2(\varepsilon - P)]$$

    Args:
        r (float): Radius as integration variable
        y (array): Array of integration variables containing [pressure, mass, h, b]
        inveos (function): Inverse equation of state function
        ad_index (float): Adiabatic index

    Returns:
        array: Array containing derivatives [dpdr, dmdr, dhdr, dfdr]
    """

    pres, m, h, b = y

    eps = inveos(pres)

    dpdr = -(eps + pres) * (m + 4.0 * pi * r**3.0 * pres)
    dpdr = dpdr / (r * (r - 2.0 * m))

    dmdr = 4.0 * pi * r**2.0 * eps

    dhdr = b

    dfdr = 2.0 * np.power(1.0 - 2.0 * m / r, -1) * h * (
        -2.0
        * np.pi
        * (5.0 * eps + 9.0 * pres + (eps + pres) ** 2.0 / (pres * ad_index))
        + 3.0 / np.power(r, 2)
        + 2.0
        * np.power(1.0 - 2.0 * m / r, -1)
        * np.power(m / np.power(r, 2) + 4.0 * np.pi * r * pres, 2)
    ) + 2.0 * b / r * np.power(1.0 - 2.0 * y[1] / r, -1) * (
        -1.0 + m / r + 2.0 * np.pi * np.power(r, 2) * (eps - pres)
    )

    return np.array([dpdr, dmdr, dhdr, dfdr])


def tidal_deformability(y2, Mns, Rns):
    """Compute Tidal deformability from y2, neutron star mass and radius

    潮汐形变参数 (Tidal Deformability) $\Lambda$ 的计算:

    首先定义致密度:
    $$C = \frac{M}{R}$$

    然后计算辅助量 $E$:
    $$E = 4C^3(13 - 11y + C(3y - 2) + 2C^2(1 + y)) + 3(1-2C)^2(2-y+2C(y-1))\ln(1-2C) + 2C(6-3y+3C(5y-8))$$

    最终得到潮汐形变参数:
    $$\Lambda = \frac{16}{15E}(1-2C)^2(2+2C(y-1)-y)$$

    Args:
        y2 (array): midiate varrible that computing tidal
        Mns (array): neutron star mass in g/cm3
        Rns (array): neutron star radius in cm.

    Returns:
        tidal_def (array): neutron star tidal deformability with unit-less.
    """
    C = Mns / Rns

    Eps = (
        4.0
        * C**3.0
        * (13.0 - 11.0 * y2 + C * (3.0 * y2 - 2.0) + 2.0 * C**2.0 * (1.0 + y2))
        + 3.0
        * (1.0 - 2.0 * C) ** 2.0
        * (2.0 - y2 + 2.0 * C * (y2 - 1.0))
        * np.log(1.0 - 2.0 * C)
        + 2.0 * C * (6.0 - 3.0 * y2 + 3.0 * C * (5.0 * y2 - 8.0))
    )

    tidal_def = (
        16.0 / (15.0 * Eps) * (1.0 - 2.0 * C) ** 2.0 * (2.0 + 2.0 * C * (y2 - 1.0) - y2)
    )

    return tidal_def


def solveTOV_tidal(center_rho, energy_density, pressure, max_radius=30e5 * unit.cm):
    """Solve TOV equation from given Equation of state in the neutron star core density range

    Args:
        center_rho(array): This is the energy density here is fixed in main that is np.logspace(14.3, 15.6, 50)
        energy_density (array): Desity array of the neutron star EoS, in MeV/fm^{-3}.
        pressure (array): Pressure array of neutron star EoS, also in nautral unit with MeV/fm^{-3}.

    Returns:
        tuple[float, float, float]: (mass_Msun, radius_km, tidal_deformability)
            Tidal deformability is dimensionless.
    """
    c = 3e10
    G = 6.67428e-8

    center_rho = center_rho / unit.g_cm_3
    energy_density = energy_density * G / c**2 / unit.g_cm_3
    pressure = pressure * G / c**4 / unit.dyn_cm_2

    unique_pressure_indices = np.unique(pressure, return_index=True)[1]
    unique_pressure = pressure[np.sort(unique_pressure_indices)]

    if np.any(np.diff(unique_pressure) == 0):
        raise ValueError("Pressure values are not unique")

    if np.any(np.diff(energy_density) == 0):
        raise ValueError("energy_density values are not unique")

    eos = interp1d(energy_density, pressure, kind="cubic", fill_value="extrapolate")

    inveos = interp1d(
        unique_pressure,
        energy_density[unique_pressure_indices],
        kind="cubic",
        fill_value="extrapolate",
    )

    Pmin = pressure[20]

    r = 1e-18
    dr = 10.0
    rhocent = center_rho * G / c**2.0
    pcent = eos(rhocent)
    P0 = pcent - (2.0 * pi / 3.0) * (pcent + rhocent) * (3.0 * pcent + rhocent) * r**2.0
    m0 = 4.0 / 3.0 * pi * rhocent * r**3.0
    h0 = r**2.0
    b0 = 2.0 * r
    stateTOV = np.array([P0, m0, h0, b0])
    ad_index = pressure_adind(P0, energy_density, pressure)

    sy = ode(TOV_def, None).set_integrator("dopri5", atol=1e-8, rtol=1e-8)

    sy.set_initial_value(stateTOV, r).set_f_params(inveos, ad_index)

    prev_mass = m0
    rho_current = rhocent

    while (sy.successful() and
           stateTOV[0] > Pmin and
           sy.t < max_radius and
           rho_current > 1e-13 * rhocent):

        stateTOV = sy.integrate(sy.t + dr)

        if stateTOV[1] < prev_mass:
            break
        prev_mass = stateTOV[1]

        dpdr, dmdr, dhdr, dfdr = TOV_def(sy.t + dr, stateTOV, inveos, ad_index)

        delta = 0.23
        dr = delta / ((1.0 / stateTOV[1]) * dmdr - (1.0 / stateTOV[0]) * dpdr)

        if stateTOV[0] < Pmin * 10:
            dr = min(dr, 50.0)

    if not sy.successful() or stateTOV[0] <= 0:
        Mb = prev_mass
        Rns = sy.t
        tidal = 0
    else:
        Mb = stateTOV[1]
        Rns = sy.t
        y = Rns * stateTOV[3] / stateTOV[2]
        tidal = tidal_deformability(y, Mb, Rns)

    return Mb * c**2.0 / G * unit.g, Rns * unit.cm, tidal


def solveTOV(center_rho, Pmin, eos, inveos, max_radius=30e5 * unit.cm):
    """Solve the Tolman-Oppenheimer-Volkoff (TOV) equation to determine the structure of a neutron star

    Args:
        center_rho (float): The central energy density of the neutron star in unit.g_cm_3
        Pmin (float): Minimum pressure threshold that defines the star's surface
        eos (function): Function mapping energy density to pressure
        inveos (function): Inverse equation of state function
        max_radius (float, optional): Safety parameter to prevent runaway integration

    Returns:
        tuple[float, float]: (mass_Msun, radius_km)
    """

    r = 1e-18 * unit.cm
    dr = 10.0 * unit.cm

    pcent = eos(center_rho)

    P0 = (
        pcent
        - (4.0 * pi / 3.0) * (pcent + center_rho) * (3.0 * pcent + center_rho) * r**2.0
    )

    m0 = 4.0 / 3.0 * pi * center_rho * r**3.0

    stateTOV = np.array([P0, m0])

    sy = ode(TOV, None).set_integrator("dopri5", atol=1e-8, rtol=1e-8)

    sy.set_initial_value(stateTOV, r).set_f_params(inveos)

    prev_mass = m0
    rho_current = center_rho

    while (sy.successful() and
           stateTOV[0] > Pmin and
           sy.t < max_radius and
           rho_current > 1e-13 * center_rho):

        stateTOV = sy.integrate(sy.t + dr)

        if stateTOV[1] < prev_mass:
            break
        prev_mass = stateTOV[1]

        dpdr, dmdr = TOV(sy.t + dr, stateTOV, inveos)
        delta = 0.23
        dr = delta / ((1.0 / stateTOV[1]) * dmdr - (1.0 / stateTOV[0]) * dpdr)

        if stateTOV[0] < Pmin * 10:
            dr = min(dr, 50.0 * unit.cm)

    if not sy.successful() or stateTOV[0] <= 0:
        return prev_mass * unit.c**2 / unit.G, sy.t

    return stateTOV[1] * unit.c**2 / unit.G, sy.t
