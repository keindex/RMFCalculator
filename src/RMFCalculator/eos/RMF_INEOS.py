"""
RMF_INEOS.py - 无限核物质状态方程计算 (不含β平衡)

基于相对论平均场 (RMF) 理论计算无限核物质的能量密度和压强。
不考虑β平衡条件，仅计算核物质部分。
"""

from RMFCalculator.eos.unit import *

from scipy import optimize

import numpy as np

import math


def _validate_rmf_theta(theta):
    r"""
    校验并返回长度为 10 的数值型 RMF 参数向量 $\theta$。

    $$\theta = [m_\sigma,\, m_\omega,\, m_\rho,\, g_\sigma,\, g_\omega,\, g_\rho,\, \kappa,\, \lambda_0,\, \zeta,\, \Lambda_\omega]$$
    """
    try:
        theta_array = np.asarray(theta, dtype=np.float64)

    except (TypeError, ValueError) as exc:
        raise TypeError(
            "RMF_INEOS expects a length-10 numeric RMF theta."
        ) from exc

    if theta_array.shape != (10,):
        raise TypeError(
            "RMF_INEOS expects a length-10 numeric RMF theta."
        )

    return theta_array


def initial_values(rho, theta, proton_fraction=0.5):
    r"""
    在给定重子数密度 $\rho$ 下, 估算 RMF 场与化学势的初值。

    基于线性化/平均场近似, 对对称核物质 ($x_p = 0.5$) 近似处理。

    Args:
        rho (float): 重子数密度 $\rho$, 单位: fm$^{-3}$
        theta (array): RMF 模型 10 个参数
        proton_fraction (float): 质子分数, 默认 0.5 (对称核物质)

    Returns:
        tuple: $(\sigma,\, \omega,\, \rho_{03},\, \mu_n,\, \mu_p)$
    """
    # 解包 RMF 参数向量 $\theta$
    # $$\theta = [m_\sigma,\, m_\omega,\, m_\rho,\, g_\sigma,\, g_\omega,\, g_\rho,\, \kappa,\, \lambda_0,\, \zeta,\, \Lambda_\omega]$$
    m_sig, m_w, m_rho, g_sigma, g_omega, g_rho, kappa, lambda_0, zeta, Lambda_w = theta

    # $\sigma$ 场线性近似: $\sigma \approx \dfrac{g_\sigma \,\rho}{m_\sigma^2}$
    sigma = g_sigma * rho / (m_sig**2)

    # 同位旋矢量 $\rho_{03}$ 场线性近似:
    # $$\rho_{03} \approx -\frac{g_\rho \,\rho\, (0.5 - x_p)}{m_\rho^2}$$
    # 其中 $x_p$ 为质子分数, 对称核物质时 $x_p = 0.5$, $\rho_{03} = 0$
    rho_03 = -g_rho * rho * (0.5 - proton_fraction) / (m_rho**2)

    # $\omega$ 场近似 (含 $\omega$-$\rho$ 交叉耦合):
    # $$\omega \approx \frac{\rho}{\dfrac{m_\omega^2}{g_\omega} + 2\,\Lambda_\omega\, g_\omega\, (g_\rho\, \rho_{03})^2}$$
    omega = rho / (m_w**2 / g_omega + 2.0 * Lambda_w * (g_rho * rho_03)**2 * g_omega)

    # 核子有效质量: $m^* = m_n - g_\sigma\, \sigma$
    m_eff = m_n - g_sigma * sigma

    # 费米动量: $k_F = (3\pi^2 \rho)^{1/3}$, 单位: fm$^{-1}$
    k_f = (3 * math.pi**2 * rho)**(1/3)

    # 费米能量: $E_F = \sqrt{k_F^2 + m^{*2}}$
    E_F = math.sqrt(k_f**2 + m_eff**2)

    # 核子化学势近似 = 费米能量 + 矢量势:
    # $$\mu_b = E_F + g_\omega\, \omega + g_\rho\, \rho_{03}\, I_3^b$$
    # 其中 $I_3$ 为同位旋第三分量 (质子 $+1/2$, 中子 $-1/2$)。
    # 必须包含矢量势, 否则 $k_F^2 = E_F^2 - m^{*2} < 0$, 求解器会收敛到平凡解。
    mu_n = E_F + g_omega * omega + g_rho * rho_03 * Matrix_b[1, 2]

    mu_p = E_F + g_omega * omega + g_rho * rho_03 * Matrix_b[0, 2]

    return sigma, omega, rho_03, mu_n, mu_p


def functie(x, args):
    r"""
    RMF 平均场自洽方程残差向量 $f(\sigma,\, \omega,\, \rho_{03},\, \mu_n,\, \mu_p) = 0$。

    对于给定质子分数的核物质, 求解介子场与化学势。

    Args:
        x (array): $[\sigma,\, \omega,\, \rho_{03},\, \mu_n,\, \mu_p]$ 当前试探解
        args (array): $\theta$ 的 10 个参数 + 密度 $\rho$ + 质子分数 $x_p$ (共 12 元)

    Returns:
        list: 五个残差 $f_i^2$
    """
    # ==================== 解包参数 ====================

    # 解包 RMF 参数向量 $\theta$:
    # $$\theta = [m_\sigma,\, m_\omega,\, m_\rho,\, g_\sigma,\, g_\omega,\, g_\rho,\, \kappa,\, \lambda_0,\, \zeta,\, \Lambda_\omega]$$
    m_sig = args[0]

    m_w = args[1]

    m_rho = args[2]

    g_sigma = args[3]

    g_omega = args[4]

    g_rho = args[5]

    kappa = args[6]

    lambda_0 = args[7]

    zeta = args[8]

    Lambda_w = args[9]

    # 重子数密度 $\rho$, 单位: fm$^{-3}$
    rho = args[10]

    # 质子分数 $x_p$
    x_p = args[11]

    # ==================== 解包试探解 ====================

    # 当前试探解: $x = [\sigma,\, \omega,\, \rho_{03},\, \mu_n,\, \mu_p]$
    sigma = x[0]

    omega = x[1]

    rho_03 = x[2]

    mu_n = x[3]

    mu_p = x[4]

    # 化学势列表: $[\mu_p,\, \mu_n]$, 对应 $i=0$ (质子), $i=1$ (中子)
    mu_B_list = [mu_p, mu_n]

    # 初始化重子数密度列表 $\rho_B$ 和标量密度列表 $\rho_{SB}$
    rho_B_list = []

    rho_SB_list = []

    # ==================== 核子有效质量 ====================

    # 核子有效质量: $m^* = m_n - g_\sigma\, \sigma$
    # 其中 $m_n$ 为核子质量, $g_\sigma$ 为 $\sigma$ 介子耦合常数, $\sigma$ 为标量场期望值
    m_eff = m_n - g_sigma * sigma

    # ==================== 遍历两种核子 (质子 i=0, 中子 i=1) ====================

    # 对每种核子计算费米能量、费米动量、重子数密度和标量密度
    # 质子: $I_3 = +1/2$; 中子: $I_3 = -1/2$ (由 Matrix_b[i, 2] 给出)
    for i in range(2):

        # 费米能量 (单粒子能量):
        # $$E_F^b = \mu_b - g_\omega\, \omega - g_\rho\, \rho_{03}\, I_3^b$$
        # 其中 $\mu_b$ 为核子化学势, $g_\omega \omega$ 为矢量排斥势, $g_\rho \rho_{03} I_3$ 为同位旋矢量势
        E_fb = mu_B_list[i] - g_omega * omega - g_rho * rho_03 * Matrix_b[i, 2]

        # 费米动量平方: $(k_F^b)^2 = (E_F^b)^2 - m^{*2}$
        # 若 $(k_F^b)^2 < 0$, 说明费米能量低于有效质量, 取 $k_F^b = 0$, $E_F^b = m^*$
        k_fb_sq = E_fb**2 - m_eff**2

        if k_fb_sq < 0:
            k_fb_sq = np.clip(k_fb_sq, a_min=0.0, a_max=None)

            E_fb = m_eff

        # 费米动量: $k_F^b = \sqrt{(E_F^b)^2 - m^{*2}}$, 单位: fm$^{-1}$
        k_fb = math.sqrt(k_fb_sq)

        # 重子数密度:
        # $$\rho_b = \frac{(k_F^b)^3}{3\pi^2}$$
        # 单位: fm$^{-3}$, 由费米动量在动量空间积分得到
        rho_B = k_fb**3 / (3 * math.pi**2)

        rho_B_list.append(rho_B)

        # 标量密度:
        # $$\rho_{SB}^b = \frac{m^*}{2\pi^2}\left[E_F^b\, k_F^b - m^{*2}\, \ln\frac{E_F^b + k_F^b}{m^*}\right]$$
        # 标量密度出现在 $\sigma$ 场方程中, 描述核子的标量源
        rho_SB = (m_eff / (2 * math.pi**2)) * (
            E_fb * k_fb - m_eff**2 * np.log((E_fb + k_fb) / m_eff)
        )

        rho_SB_list.append(rho_SB)

    # ==================== 提取质子与中子数密度 ====================

    # 质子数密度: $\rho_p = \rho_B^{(0)}$
    rho_p = rho_B_list[0]

    # 中子数密度: $\rho_n = \rho_B^{(1)}$
    rho_n = rho_B_list[1]

    # 总重子数密度: $\rho_B = \rho_p + \rho_n$
    rho_total = rho_p + rho_n

    # ==================== RMF 自洽方程残差 (5个方程) ====================

    f = [
        # (1) $\sigma$ 场方程残差:
        # $$\frac{m_\sigma^2}{g_\sigma}\,\sigma = \sum_b \rho_{SB}^b - \frac{\kappa}{2}(g_\sigma\,\sigma)^2 - \frac{\lambda_0}{6}(g_\sigma\,\sigma)^3$$
        # 移项后取平方作为残差
        (
            sigma * m_sig**2 / g_sigma
            - sum(rho_SB_list)
            + kappa * (g_sigma * sigma)**2 / 2
            + lambda_0 * (g_sigma * sigma)**3 / 6
        )**2,

        # (2) $\omega$ 场方程残差:
        # $$\frac{m_\omega^2}{g_\omega}\,\omega = \rho_B - \frac{\zeta}{6}(g_\omega\,\omega)^3 - 2\,\Lambda_\omega\, g_\omega\, \omega\, (g_\rho\, \rho_{03})^2$$
        # 移项后取平方作为残差
        (
            omega * m_w**2 / g_omega
            - rho_total
            + zeta * (g_omega * omega)**3 / 6
            + 2 * Lambda_w * g_omega * omega * (rho_03 * g_rho)**2
        )**2,

        # (3) $\rho$ 场方程残差:
        # $$\frac{m_\rho^2}{g_\rho}\,\rho_{03} = \frac{1}{2}(\rho_p - \rho_n) - 2\,\Lambda_\omega\, g_\rho\, \rho_{03}\, (g_\omega\, \omega)^2$$
        # 移项后取平方作为残差
        (
            rho_03 * m_rho**2 / g_rho
            - (rho_p * 0.5 - rho_n * 0.5)
            + 2 * Lambda_w * g_rho * rho_03 * (omega * g_omega)**2
        )**2,

        # (4) 重子数守恒约束: $\rho = \rho_p + \rho_n$
        (rho - rho_total)**2,

        # (5) 质子分数约束: $x_p = \dfrac{\rho_p}{\rho}$
        (x_p - rho_p / rho)**2,
    ]

    return f


def Energy_density_Pressure(x, rho, theta, x_p=0.5):
    r"""
    计算无限核物质的能量密度和压强。

    Args:
        x (array): $[\sigma,\, \omega,\, \rho_{03},\, \mu_n,\, \mu_p]$
        rho (float): 重子数密度, 单位: fm$^{-3}$
        theta (array): RMF 参数向量
        x_p (float): 质子分数, 默认 0.5 (对称核物质)

    Returns:
        list: $[\rho,\, \varepsilon,\, P,\, \mu_n,\, \mu_p,\, x_p]$
    """
    # ==================== 解包试探解 ====================

    # 介子场与化学势: $x = [\sigma,\, \omega,\, \rho_{03},\, \mu_n,\, \mu_p]$
    sigma, omega, rho_03, mu_n, mu_p = x

    # 解包 RMF 参数向量 $\theta$:
    # $$\theta = [m_\sigma,\, m_\omega,\, m_\rho,\, g_\sigma,\, g_\omega,\, g_\rho,\, \kappa,\, \lambda_0,\, \zeta,\, \Lambda_\omega]$$
    m_sig, m_w, m_rho, g_sigma, g_omega, g_rho, kappa, lambda_0, zeta, Lambda_w = theta

    # 初始化能量密度列表
    energy_b_list = []

    # 化学势列表: $[\mu_p,\, \mu_n]$, 对应 $i=0$ (质子), $i=1$ (中子)
    mu_B_list = [mu_p, mu_n]

    # 初始化重子数密度列表
    rho_B_list = []

    # ==================== 核子有效质量 ====================

    # 核子有效质量: $m^* = m_n - g_\sigma\, \sigma$
    m_eff = m_n - g_sigma * sigma

    # ==================== 遍历两种核子 (质子 i=0, 中子 i=1) ====================

    # 对每种核子计算费米能量、费米动量、重子数密度和能量密度
    for i in range(2):

        # 费米能量 (单粒子能量):
        # $$E_F^b = \mu_b - g_\omega\, \omega - g_\rho\, \rho_{03}\, I_3^b$$
        E_fb = mu_B_list[i] - g_omega * omega - g_rho * rho_03 * Matrix_b[i, 2]

        # 费米动量平方: $(k_F^b)^2 = (E_F^b)^2 - m^{*2}$
        k_fb_sq = E_fb**2 - m_eff**2

        if k_fb_sq < 0:
            k_fb_sq = 0.0

            E_fb = m_eff

        # 费米动量: $k_F^b = \sqrt{(E_F^b)^2 - m^{*2}}$
        k_fb = math.sqrt(k_fb_sq)

        # 重子数密度:
        # $$\rho_b = \frac{(k_F^b)^3}{3\pi^2}$$
        rho_B = k_fb**3 / (3 * math.pi**2)

        rho_B_list.append(rho_B)

        # 核子动能密度 (相对论费米气体积分):
        # $$\varepsilon_b = \frac{1}{8\pi^2}\left[k_F^b\, (E_F^b)^3 + (k_F^b)^3\, E_F^b - m^{*4}\, \ln\frac{k_F^b + E_F^b}{m^*}\right]$$
        # 单位: MeV/fm$^3$
        energy_baryon = (1 / (8 * math.pi**2)) * (
            k_fb * E_fb**3 + k_fb**3 * E_fb - m_eff**4 * np.log((k_fb + E_fb) / m_eff)
        )

        energy_b_list.append(energy_baryon)

    # ==================== 介子场能量项 ====================

    # $\sigma$ 介子场能量:
    # $$\varepsilon_\sigma = \frac{1}{2}(m_\sigma\, \sigma)^2 + \frac{\kappa}{6}(g_\sigma\, \sigma)^3 + \frac{\lambda_0}{24}(g_\sigma\, \sigma)^4$$
    sigma_terms = (
        0.5 * (sigma * m_sig)**2
        + kappa * (g_sigma * sigma)**3 / 6
        + lambda_0 * (g_sigma * sigma)**4 / 24
    )

    # $\omega$ 介子场能量:
    # $$\varepsilon_\omega = \frac{1}{2}(m_\omega\, \omega)^2 + \frac{\zeta}{8}(g_\omega\, \omega)^4$$
    omega_terms = 0.5 * (omega * m_w)**2 + zeta * (g_omega * omega)**4 / 8

    # $\rho$ 介子场能量 (含 $\omega$-$\rho$ 交叉耦合):
    # $$\varepsilon_\rho = \frac{1}{2}(m_\rho\, \rho_{03})^2 + 3\,\Lambda_\omega\, (g_\rho\, \rho_{03}\, g_\omega\, \omega)^2$$
    rho_terms = 0.5 * (rho_03 * m_rho)**2 + 3 * Lambda_w * (g_rho * rho_03 * g_omega * omega)**2

    # ==================== 总能量密度 ====================

    # 总能量密度:
    # $$\varepsilon = \sum_b \varepsilon_b + \varepsilon_\sigma + \varepsilon_\omega + \varepsilon_\rho$$
    energy_density = sum(energy_b_list) + sigma_terms + omega_terms + rho_terms

    # ==================== 压强 ====================

    # 压强 (热力学关系):
    # $$P = \sum_b \mu_b\, \rho_b - \varepsilon$$
    Pressure = sum(mu_B_list[i] * rho_B_list[i] for i in range(2)) - energy_density

    return [rho, energy_density, Pressure, mu_n, mu_p, x_p]


def compute_INEOS(theta, x_p=0.5, n_points=124, dt=0.05, rho_0=0.1505):
    r"""
    计算无限核物质状态方程。

    Args:
        theta (array): RMF 参数向量 $[m_\sigma,\, m_\omega,\, m_\rho,\, g_\sigma,\, g_\omega,\, g_\rho,\, \kappa,\, \lambda_0,\, \zeta,\, \Lambda_\omega]$
        x_p (float): 质子分数, 默认 0.5 (对称核物质)
        n_points (int): 密度点数量, 默认 124
        dt (float): 密度步长因子, 默认 0.05
        rho_0 (float): 饱和密度, 默认 0.1505 fm$^{-3}$

    Returns:
        ndarray: $(6,\, n_\text{points})$ 数组, 包含:
            [0] 重子数密度 $\rho$ (fm$^{-3}$)
            [1] 能量密度 $\varepsilon$ (MeV/fm$^3$)
            [2] 压强 $P$ (MeV/fm$^3$)
            [3] 中子化学势 $\mu_n$ (MeV)
            [4] 质子化学势 $\mu_p$ (MeV)
            [5] 质子分数 $x_p$
    """
    # 校验参数向量 $\theta$
    theta = _validate_rmf_theta(theta)

    # 在最低密度点估算初值: $\rho_\text{init} = 0.1\, \rho_0$
    x_init = np.array(initial_values(0.1 * rho_0, theta, x_p))

    # 初始化 EOS 数组
    EoS = [[] for _ in range(n_points)]

    # ==================== 密度循环 ====================

    # 从 $\rho = 0.1\, \rho_0$ 开始, 以步长 $\Delta\rho = dt \cdot \rho_0$ 递增
    # $$\rho_i = 0.1\, \rho_0 + dt \cdot i \cdot \rho_0, \quad i = 0, 1, \ldots, n_\text{points}-1$$
    for i in range(n_points):

        # 当前密度: $\rho_i = 0.1\, \rho_0 + dt \cdot i \cdot \rho_0$
        rho = 0.1 * rho_0 + dt * i * rho_0

        # 构造求解器参数: $[\theta,\, \rho,\, x_p]$
        arg = np.array([*theta, rho, x_p])

        # 求解 RMF 自洽方程: $f(\sigma,\, \omega,\, \rho_{03},\, \mu_n,\, \mu_p) = 0$
        # 使用 Levenberg-Marquardt 方法
        sol = optimize.root(functie, x_init, method="lm", args=arg)

        if not sol.success:
            print(f"Warning: solver failed at i={i}, rho={rho:.4f}")

        # 计算能量密度和压强: $\varepsilon,\, P$
        Re = Energy_density_Pressure(sol.x, rho, theta, x_p)

        EoS[i] = Re

        # 以上一密度点的解作为下一密度点的初值 (热力学连续性)
        x_init = sol.x

    EoS = np.array(EoS)

    return EoS


def compute_symmetric_INEOS(theta, n_points=124, dt=0.05, rho_0=0.1505):
    r"""
    计算对称核物质 ($x_p = 0.5$) 状态方程。

    Args:
        theta (array): RMF 参数向量
        n_points (int): 密度点数量
        dt (float): 密度步长因子
        rho_0 (float): 饱和密度

    Returns:
        ndarray: 对称核物质 EOS
    """
    return compute_INEOS(theta, x_p=0.5, n_points=n_points, dt=dt, rho_0=rho_0)


def convert_to_cgs(EoS):
    r"""
    将 EOS 单位从 MeV/fm$^3$ 转换为 cgs 单位。

    Args:
        EoS (ndarray): $(6,\, n)$ 数组

    Returns:
        ndarray: $(6,\, n)$ 数组, 单位:
            $\rho$: g/cm$^3$
            $\varepsilon$: g/cm$^3$
            $P$: dyn/cm$^2$
    """
    result = EoS.copy()

    # 能量密度单位换算: MeV/fm$^3$ → g/cm$^3$
    # $$\varepsilon\,[\mathrm{g/cm^3}] = \varepsilon\,[\mathrm{MeV/fm^3}] \times \frac{\hbar c}{\mathrm{g/cm^3 \to MeV/fm^3}}$$
    result[1] = EoS[1] * fm_MeV / gcm3_to_MeVfm3

    # 压强单位换算: MeV/fm$^3$ → dyn/cm$^2$
    # $$P\,[\mathrm{dyn/cm^2}] = P\,[\mathrm{MeV/fm^3}] \times \frac{\hbar c}{\mathrm{dyn/cm^2 \to MeV/fm^3}}$$
    result[2] = EoS[2] * fm_MeV / dyncm2_to_MeVfm3

    # 重子数密度单位换算: fm$^{-3}$ → g/cm$^3$
    # $$\rho\,[\mathrm{g/cm^3}] = \rho\,[\mathrm{fm^{-3}}] \times 1.66054 \times 10^{14}$$
    result[0] = EoS[0] * 1.66054e14

    return result


if __name__ == "__main__":
    # 测试: 使用典型 RMF 参数 (NL3)
    # $$\theta_\text{NL3} = [m_\sigma,\, m_\omega,\, m_\rho,\, g_\sigma,\, g_\omega,\, g_\rho,\, \kappa,\, \lambda_0,\, \zeta,\, \Lambda_\omega]$$
    theta_nl3 = [
        510.0,   # $m_\sigma$ (MeV)
        783.0,   # $m_\omega$ (MeV)
        770.0,   # $m_\rho$ (MeV)
        8.91,    # $g_\sigma$
        11.47,   # $g_\omega$
        4.70,    # $g_\rho$
        1.58,    # $\kappa$
        -0.04,   # $\lambda_0$
        0.00,    # $\zeta$
        0.00,    # $\Lambda_\omega$
    ]

    print("Computing symmetric nuclear matter EOS with NL3 parameters...")

    EoS = compute_symmetric_INEOS(theta_nl3)

    print("\nResults (first 5 points):")

    print(f"{'ρ(fm⁻³)':<12} {'ε(MeV/fm³)':<14} {'P(MeV/fm³)':<14} {'μ_n(MeV)':<12} {'μ_p(MeV)':<12}")

    print("-" * 64)

    print(len(EoS))

    for i in range(len(EoS)):
        print(f"{EoS[i,0]:<12.4f} {EoS[i,1]:<14.4f} {EoS[i,2]:<14.4f} {EoS[i,3]:<12.4f} {EoS[i,4]:<12.4f}")


