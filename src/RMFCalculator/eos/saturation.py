r"""
saturation.py - 核物质饱和性质计算 (K0, J0, L 等)

基于论文第一章的公式。

对称核物质的每核子结合能 $E_0(n)$ 在饱和密度 $n_0$ 附近展开:

$$E_0(n) = E_0(n_0) + \frac{K_0}{2!}\,\chi^2 + \frac{J_0}{3!}\,\chi^3 + O(\chi^4), \qquad \chi \equiv \frac{n - n_0}{3 n_0}$$

其中饱和性质定义为:

$$K_0 = 9\, n^2 \left.\frac{\partial^2 E_0}{\partial n^2}\right|_{n=n_0} \quad (\text{不可压缩系数})$$

$$J_0 = 27\, n^3 \left.\frac{\partial^3 E_0}{\partial n^3}\right|_{n=n_0} \quad (\text{偏斜系数})$$

对称能在密度 $n_r$ 处展开:

$$E_{\rm sym}(n) = E_{\rm sym}(n_r) + L\,\chi_r + \frac{K_{\rm sym}}{2!}\,\chi_r^2 + O(\chi_r^3), \qquad \chi_r \equiv \frac{n - n_r}{3 n_r}$$

$$L = 3\, n_r \left.\frac{\partial E_{\rm sym}}{\partial n}\right|_{n=n_r} \quad (\text{对称能斜率})$$

对称能通过 $E_{\rm sym}(n) = E(n, \alpha=1) - E(n, \alpha=0)$ 计算,
其中 $\alpha=1$ 为纯中子物质 (PNM, $x_p=0$), $\alpha=0$ 为对称核物质 (SNM, $x_p=0.5$)。
"""

import numpy as np
from scipy.signal import savgol_filter

from RMFCalculator.eos.RMF_INEOS import compute_INEOS
from RMFCalculator.eos.unit import m_n, fm_MeV


# ==================== 状态方程辅助 ====================

def fluid_energy(theta, x_p, n_points=200, dt=0.005):
    r"""
    计算给定质子分数 $x_p$ 的每核子结合能 $E/A(\rho)$ 与压强 $P(\rho)$, 过滤非物理解。

    Args:
        theta (array): RMF 参数向量
        x_p (float): 质子分数
        n_points (int): 密度点数
        dt (float): 密度步长因子

    Returns:
        tuple: $(\rho,\, E/A,\, P)$ — 过滤后的密度 (fm$^{-3}$)、
        每核子结合能 (MeV)、压强 (MeV/fm$^3$)
    """
    EoS = compute_INEOS(theta, x_p=x_p, n_points=n_points, dt=dt)
    rho = EoS[:, 0]
    eps = EoS[:, 1] * fm_MeV
    P = EoS[:, 2] * fm_MeV
    binding = eps / rho - m_n * fm_MeV

    # 过滤低密度求解器失效的非物理解。
    # 下限: 排除求解器崩溃产生的极端负结合能 (E/A < -25 MeV 为病态)。
    # 上限: 纯中子物质 (PNM, x_p=0) 是不束缚的, 其 E/A 随密度增大而增大,
    #       可能超过 5 MeV, 因此对 x_p=0 不设上限。
    mask = binding > -25.0
    if x_p > 0.0:
        mask &= binding < 5.0
    return rho[mask], binding[mask], P[mask]


# ==================== 饱和密度 ====================

def find_saturation_density(theta, n_points=124, dt=0.05, rho_0=0.1505):
    r"""
    计算对称核物质的饱和密度及其饱和性质。

    原理: 对称核物质 ($x_p = 0.5$) 的压强 $P(\rho)$ 在 $\rho = \rho_0$ 处过零点,
    通过线性插值 $P(\rho)$ 的过零点精确定位。

    Args:
        theta (array): RMF 参数向量
        n_points (int): 密度点数
        dt (float): 密度步长因子
        rho_0 (float): 参考饱和密度 (仅用于网格构建)

    Returns:
        dict: {
            'rho_0'      : 饱和密度 (fm$^{-3}$),
            'E_A'        : 每核子结合能 $E/A = \varepsilon/\rho - m_n$ (MeV),
            'energy_dens': 饱和处能量密度 (MeV/fm$^3$),
        }
    """
    # 计算对称核物质 EOS: 返回 [ρ, ε, P, μ_n, μ_p, x_p]
    # 注意: compute_INEOS 返回的能量密度 ε 与压强 P 均为自然单位 (fm^-4),
    # 需乘以 ħc = fm_MeV 转换为 MeV/fm^3。
    EoS = compute_INEOS(theta, x_p=0.5, n_points=n_points, dt=dt, rho_0=rho_0)

    rho = EoS[:, 0]
    energy_density = EoS[:, 1] * fm_MeV   # → MeV/fm^3
    pressure = EoS[:, 2] * fm_MeV         # → MeV/fm^3

    # 每核子结合能: E/A = ε/ρ - m_n (单位 MeV)
    binding = energy_density / rho - m_n * fm_MeV

    # 过滤低密度处求解器失效产生的非物理解 (E/A 异常, 如 -460 MeV)
    physical = (binding > -25.0) & (binding < 5.0)
    rho = rho[physical]
    pressure = pressure[physical]
    energy_density = energy_density[physical]

    # 寻找压强过零点 (饱和密度附近 P 由负转正)
    # P(ρ) 在饱和密度处过零: ρ < ρ_0 时 P < 0, ρ > ρ_0 时 P > 0
    sign_change = np.where(np.sign(pressure[:-1]) != np.sign(pressure[1:]))[0]

    if len(sign_change) == 0:
        raise RuntimeError(
            "未找到压强过零点, 请检查 RMF 参数或增大密度网格范围。"
        )

    # 取第一个过零点 (最接近饱和密度)
    i = sign_change[0]

    # 在 [ρ_i, ρ_{i+1}] 区间线性插值求出 P = 0 的密度
    rho_grid = np.array([rho[i], rho[i + 1]])
    p_grid = np.array([pressure[i], pressure[i + 1]])
    rho_sat = np.interp(0.0, p_grid, rho_grid)

    # 饱和处能量密度与每核子结合能
    energy_density_sat = np.interp(rho_sat, rho, energy_density)
    binding_energy = energy_density_sat / rho_sat - m_n * fm_MeV

    return {
        'rho_0': float(rho_sat),
        'E_A': float(binding_energy),
        'energy_dens': float(energy_density_sat),
    }


# ==================== 数值导数 ====================

def _deriv_interp(y, x, x0, order=1, deg=5, window=0.05):
    r"""
    用局部多项式拟合 $y(x)$ 并求 $x_0$ 处的导数, 用于光滑曲线。

    在 $x_0$ 附近取宽度 `window` 的邻域, 用 `deg` 阶多项式拟合后求导,
    可有效抑制数值噪声。

    Args:
        y (array): 因变量
        x (array): 自变量
        x0 (float): 求导位置
        order (int): 求导阶数
        deg (int): 多项式阶数
        window (float): 拟合邻域宽度

    Returns:
        float: $d^{\text{order}}y/dx^{\text{order}}|_{x=x_0}$
    """
    mask = np.abs(x - x0) <= window
    if mask.sum() < deg + 1:
        mask = np.ones_like(x, dtype=bool)
    xl, yl = x[mask] - x0, y[mask]
    coeff = np.polyfit(xl, yl, deg)   # 高次到低次
    d = np.polyder(coeff, order)
    return np.polyval(d, 0.0)


# ==================== 饱和性质 ====================

def compute_saturation_properties(theta, n_r=0.1, rho_0_override=None):
    r"""
    计算饱和性质 $K_0, J_0, L$ 及其它饱和系数。

    Args:
        theta (array): RMF 参数向量
        n_r (float): 计算对称能斜率 $L$ 的参考密度 (fm$^{-3}$), 默认 0.1
        rho_0_override (float): 若提供, 覆盖通过数值求根得到的饱和密度

    Returns:
        dict: {
            'rho_0': 饱和密度 (fm$^{-3}$),
            'E0'   : 饱和处每核子结合能 (MeV),
            'K0'   : 不可压缩系数 (MeV),
            'J0'   : 偏斜系数 (MeV),
            'Esym' : 对称能 $E_{\rm sym}(n_r)$ (MeV),
            'Esym0': 对称能 $E_{\rm sym}(n_0)$ (MeV),
            'L'    : 对称能斜率 $L$ 在 $n_r$ 处 (MeV),
            'L0'   : 对称能斜率 $L(n_0)$ (MeV),
            'Ksym' : 对称能曲率 $K_{\rm sym}$ 在 $n_r$ 处 (MeV),
        }
    """
    # ---- 1. 对称核物质 (SNM, x_p=0.5) 与纯中子物质 (PNM, x_p=0) ----
    rho_s, e_s, P_s = fluid_energy(theta, 0.5)

    # 求饱和密度: P = n² dE/dn = 0 即 dE/dn = 0
    de_dn = np.gradient(e_s, rho_s)
    # 找 de/dn 过零点 (由负转正)
    crossings = np.where(np.diff(np.sign(de_dn)))[0]
    if len(crossings) == 0:
        raise RuntimeError("未找到饱和密度 (dE/dn 过零点)。")

    if rho_0_override is not None:
        n0 = rho_0_override
    else:
        # 取第一个过零点, 线性插值精确定位
        idx = crossings[0]
        n0 = np.interp(0.0, de_dn[idx:idx + 2], rho_s[idx:idx + 2])

    # 饱和处性质
    E0 = np.interp(n0, rho_s, e_s)

    # ---- 2. 计算 K0 和 J0 (基于压强 P, 数值更稳健) ----
    # 饱和密度处 P = n² dE/dn = 0, 且 dP/dn = n² d²E/dn², 于是
    #   K0 = 9 n² d²E/dn² = 9 dP/dn
    #   J0 = 27 n³ d³E/dn³ = 27 n₀ d²P/dn² − 108 dP/dn
    # 压强 P 比 E/A 更光滑, 用 Savitzky-Golay 求导可避免三阶差分的数值噪声。
    P = P_s
    h = rho_s[1] - rho_s[0]

    # 取约 0.02 fm⁻³ 的窗口 (奇数长度), 保证落在 n0 附近
    wl = max(21, int(0.02 / h) | 1)
    dP = savgol_filter(P, wl, 5, deriv=1, delta=h)
    d2P = savgol_filter(P, wl, 5, deriv=2, delta=h)
    dP0 = np.interp(n0, rho_s, dP)
    d2P0 = np.interp(n0, rho_s, d2P)

    K0 = 9.0 * dP0
    J0 = 27.0 * n0 * d2P0 - 108.0 * dP0

    # ---- 3. 对称能 (PNM - SNM) ----
    rho_pnm, e_pnm, _ = fluid_energy(theta, 0.0)

    # 对齐密度网格: 只保留两种物质都有效的密度范围
    lo = max(rho_s.min(), rho_pnm.min())
    hi = min(rho_s.max(), rho_pnm.max())
    rho_common = rho_s[(rho_s >= lo) & (rho_s <= hi)]
    e_snm_c = np.interp(rho_common, rho_s, e_s)
    e_pnm_c = np.interp(rho_common, rho_pnm, e_pnm)

    # 对称能: E_sym(n) = E_PNM(n) - E_SNM(n)
    e_sym = e_pnm_c - e_snm_c

    # 对称能在饱和密度 n0 处
    Esym0 = np.interp(n0, rho_common, e_sym)

    # 在 n_r 处求 L 和 K_sym
    Esym = np.interp(n_r, rho_common, e_sym)
    L = 3.0 * n_r * _deriv_interp(e_sym, rho_common, n_r, 1)
    Ksym = 9.0 * n_r**2 * _deriv_interp(e_sym, rho_common, n_r, 2)

    # 对称能斜率在饱和密度 n0 处: L(n0) = 3 n0 dE_sym/dn|_{n0}
    L0 = 3.0 * n0 * _deriv_interp(e_sym, rho_common, n0, 1)

    return {
        'rho_0': float(n0),
        'E0': float(E0),
        'K0': float(K0),
        'J0': float(J0),
        'Esym': float(Esym),
        'Esym0': float(Esym0),
        'L': float(L),
        'L0': float(L0),
        'Ksym': float(Ksym),
    }