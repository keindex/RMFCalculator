from RMFCalculator.tov.unit import g_cm_3, dyn_cm_2
from scipy import optimize
import numpy as np
import math

# ==================== 物理常数与单位换算 ====================
# 光速 $c$, 单位: cm/s
c = 2.99792458e10
# 引力常数 $G$, 单位: cm³/(g·s²)
G = 6.67428e-8
# 太阳质量 $M_\odot$, 单位: g
Msun = 1.989e33

# 压强单位换算: dyn/cm² → MeV/fm³
dyncm2_to_MeVfm3 = 1.0 / (1.6022e33)
# 能量密度单位换算: g/cm³ → MeV/fm³
gcm3_to_MeVfm3 = 1.0 / (1.7827e12)
# ħc 换算因子: $ħc \approx 197.327$ MeV·fm, 用于 fm⁻¹ ↔ MeV
oneoverfm_MeV = 197.327053

# ==================== 粒子质量 (自然单位: fm⁻¹) ====================
# 电子质量 $m_e \approx 0.511$ MeV → fm⁻¹
m_e = 2.5896041 * 10**-3
# μ 子质量 $m_\mu \approx 105.66$ MeV → fm⁻¹
m_mu = 0.5354479981
# 中子质量 $m_n \approx 938.92$ MeV → fm⁻¹
m_n = 4.7583690772
# 质子质量 $m_p$ (RMF 中通常取 $m_p = m_n$)
m_p = 4.7583690772

# 核子自旋 $J_B = 1/2$, 简并度因子 $(2J_B+1) = 2$
J_B = 1 / 2.0
# 重子数 $b_B = 1$ (核子)
b_B = 1

# 轻子质量列表: $[m_e, m_\mu]$
m_l = [m_e, m_mu]
# 核子质量列表: $[m_p, m_n]$
m_b = [m_p, m_n]

# 重子量子数矩阵 Matrix_b[i]: 行 i=0 为质子, i=1 为中子
# 列含义: [重子数 $b$, 电荷 $q$, 同位旋第三分量 $I_3$, σ耦合, ω耦合, ρ耦合]
# 质子: $b=1, q=+1, I_3=+1/2$; 中子: $b=1, q=0, I_3=-1/2$
Matrix_b = np.array(
    [[1.0, 1.0, 1 / 2.0, 1.0, 1.0, 1.0], [1.0, 0.0, -1 / 2.0, 1.0, 1.0, 1.0]]
)

# 轻子量子数矩阵 Matrix_l[j]: 行 j=0 为电子, j=1 为 μ 子
# 列含义: [重子数 $b=0$, 电荷 $q=-1$, 自旋 $J=1/2$]
Matrix_l = np.array([[0.0, -1.0, 1 / 2.0], [0.0, -1.0, 1 / 2.0]])


def _validate_rmf_theta(theta):
    """
    校验并返回长度为 10 的数值型 RMF 参数向量 $\theta$。

    $\theta = [m_\sigma, m_\omega, m_\rho, g_\sigma, g_\omega, g_\rho,\kappa, \lambda_0, \zeta, \Lambda_\omega]$

    若传入含可调用密度依赖耦合的 DDH 参数, 则抛出 TypeError,
    提示改用 EOSgenerators.RMF_DDH.compute_eos。
    """
    try:
        theta_array = np.asarray(theta, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise TypeError(
            "RMF.compute_EOS expects a length-10 numeric RMF theta. "
            "DDH theta values contain callable density-dependent couplings and "
            "must be passed to EOSgenerators.RMF_DDH.compute_eos instead."
        ) from exc

    if theta_array.shape != (10,):
        raise TypeError(
            "RMF.compute_EOS expects a length-10 numeric RMF theta. "
            "DDH theta values contain callable density-dependent couplings and "
            "must be passed to EOSgenerators.RMF_DDH.compute_eos instead."
        )

    return theta_array


def initial_values(rho, theta):
    r"""
    在给定重子数密度 $\rho$ 下, 估算 RMF 场与化学势的初值, 供非线性求解器起步。

    基于线性化/平均场近似的场方程:

    $\sigma \approx \frac{g_\sigma \rho}{m_\sigma^2},\quad
        \rho_{03} \approx -\frac{g_\rho \rho}{2 m_\rho^2}$

    Args:
        rho (float): 重子数密度 $\rho$, 单位: fm$^{-3}$
        theta (array): RMF 模型 10 个参数
            $[m_\sigma, m_\omega, m_\rho, g_\sigma, g_\omega, g_\rho,
            \kappa, \lambda_0, \zeta, \Lambda_\omega]$

    Returns:
        tuple: $(\sigma, \omega, \rho_{03}, \mu_n, \mu_e)$
            - $\sigma$: 标量介子场
            - $\omega$: 矢量介子场
            - $\rho_{03}$: 同位旋矢量介子场第三分量
            - $\mu_n$: 中子化学势, 单位: fm$^{-1}$
            - $\mu_e$: 电子化学势, 单位: fm$^{-1}$
    """
    m_sig, m_w, m_rho, g_sigma, g_omega, g_rho, kappa, lambda_0, zeta, Lambda_w = theta

    m_e = 2.5896041 * 10**-3
    m_mu = 0.5354479981
    m_n = 4.7583690772
    m_p = 4.7583690772

    rho_0 = 0.1505

    sigma = g_sigma * rho / (m_sig**2)
    rho_03 = -g_rho * rho / (2.0 * (m_rho**2))
    omega = rho * (
        (((m_w**2) / g_omega) + (2.0 * Lambda_w * ((g_rho * rho_03) ** 2) * g_omega))
        ** (-1.0)
    )
    m_eff = m_n - (g_sigma * sigma)
    mu_n = m_eff + g_omega * omega + g_rho * rho_03 * Matrix_b[1, 2]
    mu_e = 0.12 * m_e * (rho / rho_0) ** (2 / 3.0)

    return sigma, omega, rho_03, mu_n, mu_e


def functie(x, args):
    r"""
    RMF 平均场自洽方程残差向量 $f(\sigma,\omega,\rho_{03},\mu_n,\mu_e)=0$。

    在 $\beta$ 平衡与电荷中性条件下, 求解介子场与化学势。

    Args:
        x (array): $[\sigma, \omega, \rho_{03}, \mu_n, \mu_e]$ 当前试探解
        args (array): $\theta$ 的 10 个参数再附加当前密度 $\rho$ (共 11 元)

    Returns:
        list: 五个残差的平方 $f_i^2$, 供 ``scipy.optimize.root`` (LM) 最小化
    """
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
    rho = args[10]

    m_e = 2.5896041 * 10**-3
    m_mu = 0.5354479981
    m_n = 4.7583690772
    m_p = 4.7583690772

    J_B = 1 / 2.0
    b_B = 1

    m_l = np.array([m_e, m_mu])
    m_b = np.array([m_p, m_n])

    Matrix_b = np.array(
        [[1.0, 1.0, 1 / 2.0, 1.0, 1.0, 1.0], [1.0, 0.0, -1 / 2.0, 1.0, 1.0, 1.0]]
    )

    Matrix_l = np.array([[0.0, -1.0, 1 / 2.0], [0.0, -1.0, 1 / 2.0]])

    sigma = x[0]
    omega = x[1]
    rho_03 = x[2]
    mu_n = x[3]
    mu_e = x[4]

    rho_B_list = []
    rho_SB_list = []
    q_list = []

    m_eff = m_n - (g_sigma * sigma)

    for i in range(len(Matrix_b)):
        mu_b = Matrix_b[i, 0] * mu_n - Matrix_b[i, 1] * mu_e

        E_fb = mu_b - g_omega * omega - g_rho * rho_03 * Matrix_b[i, 2]

        k_fb_sq = E_fb**2 - m_eff**2
        if k_fb_sq < 0:
            k_fb_sq = np.clip(k_fb_sq, a_min=0.0, a_max=None)
            E_fb = m_eff

        k_fb = math.sqrt(k_fb_sq)

        rho_B = ((2 * J_B) + 1) * b_B * k_fb**3 / (6.0 * math.pi**2)
        rho_SB = (m_eff / (2.0 * math.pi**2)) * (
            E_fb * k_fb - (m_eff ** (2)) * np.log((E_fb + k_fb) / m_eff)
        )

        rho_B_list.append(rho_B)
        rho_SB_list.append(rho_SB)

        Q_B = ((2.0 * J_B) + 1.0) * Matrix_b[i, 1] * k_fb**3 / (6.0 * math.pi**2)
        q_list.append(Q_B)

    for j in range(len(Matrix_l)):
        mu_l = Matrix_l[j, 0] * mu_n - Matrix_l[i, 1] * mu_e
        E_fl = mu_l

        k_fl_sq = E_fl**2 - m_l[j] ** 2
        k_fl_sq = np.clip(k_fl_sq, a_min=0.0, a_max=None)
        k_fl = math.sqrt(k_fl_sq)

        Q_L = ((2.0 * J_B) + 1.0) * Matrix_l[i, 1] * (k_fl**3) / (6.0 * (math.pi**2))
        q_list.append(Q_L)

    f = [
        (
            sigma * (m_sig**2) / g_sigma
            - sum(np.array(rho_SB_list) * Matrix_b[:, 3])
            + (kappa * (g_sigma * sigma) ** 2) / 2.0
            + (lambda_0 * (g_sigma * sigma) ** 3) / 6.0
        )
        ** 2,
        (
            omega * (m_w**2) / g_omega
            - sum(np.array(rho_B_list) * Matrix_b[:, 4])
            + (zeta * (g_omega * omega) ** 3) / 6.0
            + 2.0 * Lambda_w * g_omega * omega * (rho_03 * g_rho) ** 2
        )
        ** 2,
        (
            rho_03 * (m_rho**2) / g_rho
            - sum(np.array(rho_B_list) * Matrix_b[:, 5] * Matrix_b[:, 2])
            + 2.0 * Lambda_w * g_rho * rho_03 * (omega * g_omega) ** 2
        )
        ** 2,
        (rho - sum(rho_B_list)) ** 2,
        (sum(q_list)) ** 2,
    ]

    return f


def Energy_density_Pressure(x, rho, theta, return_tag=False):
    r"""
    计算 RMF 模型的状态方程 (能量密度和压强)。

    Args:
        x (array): $[\sigma, \omega, \rho_{03}, \mu_n, \mu_e]$
        rho (float): 重子数密度 $\rho$, 单位: fm$^{-3}$
        theta (array): 长度 10 的数值型 RMF 参数向量
        return_tag (bool, optional): 是否返回完整组成信息

    Returns:
        tuple 或 list: (能量密度, 压强) 或完整 EOS 组成
    """
    sigma, omega, rho_03, mu_n, mu_e = x

    m_sig, m_w, m_rho, g_sigma, g_omega, g_rho, kappa, lambda_0, zeta, Lambda_w = theta

    m_e = 2.5896041 * 10**-3
    m_mu = 0.5354479981
    m_n = 4.7583690772
    m_p = 4.7583690772

    J_B = 1 / 2.0
    b_B = 1

    m_l = np.array([m_e, m_mu])
    m_b = np.array([m_p, m_n])

    energy_b = 0
    energy_l = 0
    multi = 0

    Composition = ["mu_p", "mu_n", "mu_e", "mu_mu", "proton_fraction"]

    m_eff = m_n - (g_sigma * sigma)

    for i in range(len(Matrix_b)):
        mu_b = Matrix_b[i, 0] * mu_n - Matrix_b[i, 1] * mu_e

        Composition[i] = mu_b

        E_fb = mu_b - g_omega * omega - g_rho * rho_03 * Matrix_b[i, 2]

        k_fb_sq = E_fb**2 - m_eff**2
        if k_fb_sq < 0:
            k_fb_sq = 0.0
            E_fb = m_eff

        k_fb = math.sqrt(k_fb_sq)

        rho_B = ((2.0 * J_B) + 1.0) * b_B * (k_fb**3) / (6.0 * math.pi**2)

        if i == 0:
            Composition[4] = rho_B / rho

        multi = multi + mu_b * rho_B
        energy_baryon = (1 / (8.0 * (math.pi**2))) * (
            k_fb * (E_fb**3)
            + (k_fb**3) * E_fb
            - np.log((k_fb + E_fb) / m_eff) * m_eff**4
        )

        energy_b = energy_b + energy_baryon

    for j in range(len(Matrix_l)):
        mu_l = Matrix_l[i, 0] * mu_n - Matrix_l[j, 1] * mu_e

        Composition[2 + j] = mu_l

        k_fl_sq = mu_l**2 - m_l[j] ** 2
        if k_fl_sq < 0.0:
            k_fl_sq = 0.0
        k_fl = math.sqrt(k_fl_sq)

        rho_l = k_fl**3 / (3.0 * math.pi**2)

        multi = multi + mu_l * rho_l
        energy_lepton = (1 / (8.0 * (math.pi**2))) * (
            k_fl * (mu_l**3)
            + mu_l * (k_fl**3)
            - (m_l[j] ** 4) * np.log((k_fl + mu_l) / m_l[j])
        )

        energy_l = energy_l + energy_lepton

    sigma_terms = (
        0.5 * ((sigma * m_sig) ** 2)
        + (kappa * ((g_sigma * sigma) ** 3)) / 6.0
        + (lambda_0 * ((g_sigma * sigma) ** 4)) / 24.0
    )

    omega_terms = 0.5 * ((omega * m_w) ** 2) + (zeta * ((g_omega * omega) ** 4)) / 8.0

    rho_terms = 0.5 * ((rho_03 * m_rho) ** 2) + +3.0 * Lambda_w * (
        (g_rho * rho_03 * g_omega * omega) ** 2
    )

    energy_density = energy_b + energy_l + sigma_terms + omega_terms + rho_terms

    Pressure = multi - energy_density

    if return_tag:
        EoS = [rho, energy_density, Pressure] + Composition
        return EoS
    else:
        return energy_density, Pressure


def compute_EOS(eps_crust, pres_crust, theta, return_tag=False):
    r"""
    由 RMF 参数向量生成中子星内核状态方程, 并与壳层 EOS 拼接。

    Args:
        eps_crust (array): 壳层能量密度, 单位: g·cm$^{-3}$
        pres_crust (array): 壳层压强, 单位: dyn·cm$^{-2}$
        theta (array): 长度 10 的数值型 RMF 参数向量
        return_tag (bool, optional): 是否同时返回化学势与质子分数

    Returns:
        tuple 或 ndarray: (能量密度, 压强) 数组, 单位 g·cm$^{-3}$ / dyn·cm$^{-2}$
    """
    theta = _validate_rmf_theta(theta)
    dt = 0.05
    rho_0 = 0.1505

    x_init = np.array(initial_values(0.1 * rho_0, theta))

    if return_tag:
        EoS = [[] for i in range(124)]
    else:
        Energy = []
        Pressure = []

    for i in range(1, 125):
        rho = i * dt * rho_0

        arg = np.append(theta, rho)
        sol = optimize.root(functie, x_init, method="lm", args=arg)

        Re = Energy_density_Pressure(x_init, rho, theta, return_tag)

        if return_tag:
            Re[1] = Re[1] * oneoverfm_MeV / gcm3_to_MeVfm3
            Re[2] = Re[2] * oneoverfm_MeV / dyncm2_to_MeVfm3

            EoS[i - 1] = Re
        else:
            Energy.append(Re[0] * oneoverfm_MeV / gcm3_to_MeVfm3)
            Pressure.append(Re[1] * oneoverfm_MeV / dyncm2_to_MeVfm3)

        x_init = sol.x

    if return_tag:
        EoS = np.array(EoS)

        end = 0
        for i in range(0, len(EoS) - 1):
            if EoS[i][1] > max(eps_crust / g_cm_3) and i > 18:
                end = i + 2
                break
            end += 1
        EoS = EoS[end::].T
        EoS[1] = EoS[1] * g_cm_3
        EoS[2] = EoS[2] * dyn_cm_2

        return EoS
    else:
        Energy = np.array(Energy)
        Pressure = np.array(Pressure)

        end = 0
        for i in range(0, len(Energy) - 1):
            if Energy[i] > max(eps_crust / g_cm_3) and i > 18:
                end = i + 2
                break
            end += 1
        ep = Energy[end::]
        pr = Pressure[end::]

        ep = ep * g_cm_3
        pr = pr * dyn_cm_2

        return ep, pr
