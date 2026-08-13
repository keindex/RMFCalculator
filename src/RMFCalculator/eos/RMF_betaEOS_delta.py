r"""
RMF_betaEOS_delta.py - 含同位旋矢量-标量介子 $\delta$ 的电中性 $\beta$ 平衡中子星物质状态方程

基于 FSUGold-$\delta$ 模型 (学位论文第四章: FSU-δ6.7, FSU-δ6.2),
在 β-平衡 (npeμ) 与电荷中性条件下求解:

$$\mu_n - \mu_p = \mu_e = \mu_\mu$$

场方程为静态均匀平均场近似 (与 ``RMF_INEOS_delta`` 相同), 但质子分数 $x_p$
由化学势关系确定, 轻子 ($e^-, \mu^-$) 作为自由费米气体计入能量密度与压强。

本模块求解变量 $x = [\sigma, \omega, \rho_{03}, \delta, \mu_n, \mu_e]$ (6 个),
$\mu_p = \mu_n - \mu_e$, $\mu_\mu = \mu_e$。

约定:
- "大 $g_\rho$" 约定 (与 ``RMF_betaEOS`` 一致);
- 参数向量为 13 元 $\theta_{13}$ (见 ``RMF_INEOS_delta``);
- 能量密度与压强输出为自然单位 (fm$^{-4}$), 通常需要乘以 $\hbar c$ 转换为 MeV/fm$^3$。
"""

import math

import numpy as np
from scipy import optimize

from RMFCalculator.eos.unit import (
    m_n, m_e, m_l, Matrix_b, Matrix_l, J_B, b_B, fm_MeV,
)
from RMFCalculator.eos.RMF_INEOS_delta import _validate_theta13, _unpack_theta13


def initial_values(rho, theta13):
    r"""
    在给定重子数密度 $\rho$ 下估算 6 个变量初值 (供求解器起步)。

    Args:
        rho (float): 重子数密度 (fm$^{-3}$)
        theta13 (array): 13 元参数向量

    Returns:
        tuple: $(\sigma, \omega, \rho_{03}, \delta, \mu_n, \mu_e)$
    """
    (m_sig, m_w, m_rho, m_delta, g_sigma, g_omega, g_rho,
     g_delta, _kappa, _lambda_0, _zeta, Lambda_w, _Lambda_sd) = _unpack_theta13(theta13)

    rho_0 = 0.1505

    sigma = g_sigma * rho / (m_sig**2)
    rho_03 = -g_rho * rho / (2.0 * m_rho**2)
    omega = rho * (
        (((m_w**2) / g_omega) + (2.0 * Lambda_w * ((g_rho * rho_03)**2) * g_omega)) ** (-1.0)
    )
    # δ 场线性近似 (丰中子物质 $n_p^s - n_n^s < 0$)
    delta = -g_delta * rho * 0.5 / (m_delta**2)

    m_eff = m_n - g_sigma * sigma
    mu_n = m_eff + g_omega * omega + g_rho * rho_03 * Matrix_b[1, 2]
    mu_e = 0.12 * m_e * (rho / rho_0) ** (2 / 3.0)

    return sigma, omega, rho_03, delta, mu_n, mu_e


def functie(x, args):
    r"""
    RMF 自洽方程残差向量 $f(\sigma, \omega, \rho_{03}, \delta, \mu_n, \mu_e) = 0$。

    Args:
        x (array): $[\sigma, \omega, \rho_{03}, \delta, \mu_n, \mu_e]$
        args (array): theta13 13 元 + 密度 $\rho$ (共 14 元)

    Returns:
        list: 6 个残差的平方 (供 LM 方法使用)
    """
    (m_sig, m_w, m_rho, m_delta, g_sigma, g_omega, g_rho,
     g_delta, kappa, lambda_0, zeta, Lambda_w, Lambda_sigma_delta) = _unpack_theta13(args[:13])
    rho = args[13]

    sigma, omega, rho_03, delta, mu_n, mu_e = x

    tau3 = np.array([1.0, -1.0])   # 论文式 (4.3a): 质子 +1, 中子 -1
    I3 = Matrix_b[:, 2]            # ρ 介子耦合: I₃ = ±1/2

    m_eff_p = m_n - g_sigma * sigma - g_delta * delta * tau3[0]
    m_eff_n = m_n - g_sigma * sigma - g_delta * delta * tau3[1]
    m_eff_list = [m_eff_p, m_eff_n]

    rho_B_list = []
    rho_SB_list = []
    q_list = []

    # ---------- 重子 (质子 i=0, 中子 i=1) ----------
    for i in range(2):
        mf = m_eff_list[i]
        if mf <= 0.0:   # 负有效质量: 空费米海
            rho_B_list.append(0.0)
            rho_SB_list.append(0.0)
            q_list.append(0.0)
            continue
        mu_b = Matrix_b[i, 0] * mu_n - Matrix_b[i, 1] * mu_e
        E_fb = mu_b - g_omega * omega - g_rho * rho_03 * I3[i]
        k_fb_sq = E_fb**2 - mf**2
        if k_fb_sq < 0:
            k_fb_sq = np.clip(k_fb_sq, a_min=0.0, a_max=None)
            E_fb = mf
        k_fb = math.sqrt(k_fb_sq)

        rho_B = ((2 * J_B) + 1) * b_B * k_fb**3 / (6.0 * math.pi**2)
        rho_B_list.append(rho_B)

        rho_SB = (mf / (2.0 * math.pi**2)) * (
            E_fb * k_fb - mf**2 * np.log((E_fb + k_fb) / mf)
        )
        rho_SB_list.append(rho_SB)

        Q_B = ((2.0 * J_B) + 1.0) * Matrix_b[i, 1] * k_fb**3 / (6.0 * math.pi**2)
        q_list.append(Q_B)

    rho_p = rho_B_list[0]
    rho_n = rho_B_list[1]
    rho_s_p = rho_SB_list[0]
    rho_s_n = rho_SB_list[1]
    rho_total = rho_p + rho_n

    # ---------- 轻子 (电子 j=0, μ子 j=1) ----------
    for j in range(2):
        mu_l = Matrix_l[j, 0] * mu_n - Matrix_l[j, 1] * mu_e
        E_fl = mu_l
        k_fl_sq = E_fl**2 - m_l[j]**2
        k_fl_sq = np.clip(k_fl_sq, a_min=0.0, a_max=None)
        k_fl = math.sqrt(k_fl_sq)

        Q_L = ((2.0 * J_B) + 1.0) * Matrix_l[j, 1] * (k_fl**3) / (6.0 * math.pi**2)
        q_list.append(Q_L)

    f = [
        # (1) σ 场方程
        (
            sigma * m_sig**2 / g_sigma
            - (rho_s_p + rho_s_n)
            + kappa * (g_sigma * sigma)**2 / 2
            + lambda_0 * (g_sigma * sigma)**3 / 6
            - Lambda_sigma_delta * (g_sigma * sigma) * (g_delta * delta)**2
        )**2,
        # (2) ω 场方程
        (
            omega * m_w**2 / g_omega
            - rho_total
            + zeta * (g_omega * omega)**3 / 6
            + 2 * Lambda_w * g_omega * omega * (rho_03 * g_rho)**2
        )**2,
        # (3) ρ 场方程
        (
            rho_03 * m_rho**2 / g_rho
            - 0.5 * (rho_p - rho_n)
            + 2 * Lambda_w * g_rho * rho_03 * (omega * g_omega)**2
        )**2,
        # (4) δ 场方程
        (
            delta * m_delta**2 / g_delta
            - (rho_s_p - rho_s_n)
            - Lambda_sigma_delta * (g_delta * delta) * (g_sigma * sigma)**2
        )**2,
        # (5) 重子数守恒: $\rho = \rho_p + \rho_n$
        (rho - rho_total)**2,
        # (6) 电荷中性: $\sum_B Q_B + \sum_L Q_L = 0$
        (sum(q_list))**2,
    ]

    return f


def Energy_density_Pressure(x, rho, theta13):
    r"""
    计算 β 平衡含 δ 介子物质的能量密度与压强。

    Args:
        x (array): $[\sigma, \omega, \rho_{03}, \delta, \mu_n, \mu_e]$
        rho (float): 重子数密度 (fm$^{-3}$)
        theta13 (array): 13 元参数向量

    Returns:
        list: $[\rho, \varepsilon, P, \mu_p, \mu_n, \mu_e, \mu_\mu, x_p]$
    """
    sigma, omega, rho_03, delta, mu_n, mu_e = x
    (m_sig, m_w, m_rho, m_delta, g_sigma, g_omega, g_rho,
     g_delta, kappa, lambda_0, zeta, Lambda_w, Lambda_sigma_delta) = _unpack_theta13(theta13)

    tau3 = np.array([1.0, -1.0])   # 论文式 (4.3a): 质子 +1, 中子 -1
    I3 = Matrix_b[:, 2]            # ρ 介子耦合: I₃ = ±1/2

    m_eff_p = m_n - g_sigma * sigma - g_delta * delta * tau3[0]
    m_eff_n = m_n - g_sigma * sigma - g_delta * delta * tau3[1]
    m_eff_list = [m_eff_p, m_eff_n]

    energy_b = 0.0
    energy_l = 0.0
    multi = 0.0

    composition = ["mu_p", "mu_n", "mu_e", "mu_mu", "proton_fraction"]

    # ---------- 重子 ----------
    for i in range(2):
        mf = m_eff_list[i]
        if mf <= 0.0:   # 负有效质量: 空费米海
            continue
        mu_b = Matrix_b[i, 0] * mu_n - Matrix_b[i, 1] * mu_e
        composition[i] = mu_b

        E_fb = mu_b - g_omega * omega - g_rho * rho_03 * I3[i]
        k_fb_sq = E_fb**2 - mf**2
        if k_fb_sq < 0:
            k_fb_sq = 0.0
            E_fb = mf
        k_fb = math.sqrt(k_fb_sq)

        rho_B = ((2.0 * J_B) + 1.0) * b_B * k_fb**3 / (6.0 * math.pi**2)

        if i == 0:
            composition[4] = rho_B / rho

        multi = multi + mu_b * rho_B

        energy_baryon = (1 / (8.0 * math.pi**2)) * (
            k_fb * E_fb**3 + k_fb**3 * E_fb
            - mf**4 * np.log((k_fb + E_fb) / mf)
        )
        energy_b = energy_b + energy_baryon

    # ---------- 轻子 ----------
    for j in range(2):
        mu_l = Matrix_l[j, 0] * mu_n - Matrix_l[j, 1] * mu_e
        composition[2 + j] = mu_l

        k_fl_sq = mu_l**2 - m_l[j]**2
        if k_fl_sq < 0.0:
            k_fl_sq = 0.0
        k_fl = math.sqrt(k_fl_sq)

        rho_l = k_fl**3 / (3.0 * math.pi**2)
        multi = multi + mu_l * rho_l

        energy_lepton = (1 / (8.0 * math.pi**2)) * (
            k_fl * mu_l**3 + mu_l * k_fl**3
            - m_l[j]**4 * np.log((k_fl + mu_l) / m_l[j])
        )
        energy_l = energy_l + energy_lepton

    # ---------- 介子场能量 ----------
    sigma_terms = (
        0.5 * (sigma * m_sig)**2
        + kappa * (g_sigma * sigma)**3 / 6
        + lambda_0 * (g_sigma * sigma)**4 / 24
        - 0.5 * Lambda_sigma_delta * (g_sigma * sigma)**2 * (g_delta * delta)**2
    )
    omega_terms = 0.5 * (omega * m_w)**2 + zeta * (g_omega * omega)**4 / 8
    rho_terms = 0.5 * (rho_03 * m_rho)**2 + 3 * Lambda_w * (g_rho * rho_03 * g_omega * omega)**2
    delta_terms = 0.5 * (delta * m_delta)**2

    energy_density = energy_b + energy_l + sigma_terms + omega_terms + rho_terms + delta_terms
    Pressure = multi - energy_density

    return [rho, energy_density, Pressure] + composition


def compute_EOS_delta(theta13, n_points=124, dt=0.05, rho_0=0.1505):
    r"""
    计算 β 平衡含 δ 介子的中子星物质状态方程。

    Args:
        theta13 (array): 13 元 RMF 参数向量
        n_points (int): 密度点数, 默认 124
        dt (float): 密度步长因子, 默认 0.05
        rho_0 (float): 参考饱和密度, 默认 0.1505 fm$^{-3}$

    Returns:
        ndarray: $(8, n_\text{points})$ 数组:
            [0] $\rho$ (fm$^{-3}$)
            [1] $\varepsilon$ (fm$^{-4}$)
            [2] $P$ (fm$^{-4}$)
            [3-6] $\mu_p, \mu_n, \mu_e, \mu_\mu$ (fm$^{-1}$)
            [7] 质子分数 $x_p$
    """
    theta13 = _validate_theta13(theta13)

    x_init = np.array(initial_values(0.1 * rho_0, theta13))

    EoS = [[] for _ in range(n_points)]

    for i in range(n_points):
        rho = 0.1 * rho_0 + dt * i * rho_0

        arg = np.array([*theta13, rho])

        sol = optimize.root(functie, x_init, method="lm", args=arg)

        if not sol.success:
            print(f"Warning: solver failed at i={i}, rho={rho:.4f}")

        Re = Energy_density_Pressure(sol.x, rho, theta13)
        EoS[i] = Re

        x_init = sol.x

    return np.array(EoS)


if __name__ == "__main__":
    from RMFCalculator.eos.parameters import get_theta

    theta13_test = get_theta("FSU_DELTA62")
    print("Computing beta-equilibrium EOS with FSU-δ6.2...")

    EoS_test = compute_EOS_delta(theta13_test, n_points=30)

    eps_test = EoS_test[:, 1] * fm_MeV
    P_test = EoS_test[:, 2] * fm_MeV
    rho_test = EoS_test[:, 0]
    xp_test = EoS_test[:, 7]

    print(f"{'rho(fm-3)':<12}{'eps(MeV/fm3)':<14}{'P(MeV/fm3)':<14}{'x_p':<10}")
    for k in range(0, 30, 3):
        print(f"{rho_test[k]:<12.4f}{eps_test[k]:<14.4f}{P_test[k]:<14.4f}{xp_test[k]:<10.4f}")