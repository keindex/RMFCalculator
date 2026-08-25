r"""
RMF_betaEOS.py - 电中性 $\beta$ 平衡中子星物质状态方程 (含/不含 $\delta$ 介子)

基于 FSUGold-$\delta$ 模型 (学位论文第四章: FSU-δ6.7, FSU-δ6.2),
在 β-平衡 (npeμ) 与电荷中性条件下求解:

$$\mu_n - \mu_p = \mu_e = \mu_\mu$$

场方程为静态均匀平均场近似 (与 ``RMF_INEOS`` 相同), 但质子分数 $x_p$
由化学势关系确定, 轻子 ($e^-, \mu^-$) 作为自由费米气体计入能量密度与压强。

本模块求解变量 $x = [\sigma, \omega, \rho_{03}, \delta, \mu_n, \mu_e]$ (6 个),
$\mu_p = \mu_n - \mu_e$, $\mu_\mu = \mu_e$。

**不含 $\delta$ 介子的模型 ($g_\delta = \Lambda_{\sigma\delta} = 0$) 是本模型的特例**:
此时 $\delta$ 场方程退化为 $\delta = 0$。因此本模块统一作为
两种模型的求解器: 统一接受 13 元参数向量 $\theta_{13}$。

约定:
- "大 $g_\rho$" 约定;
- 参数向量为 13 元 $\theta_{13}$ (见 ``RMF_INEOS``);
- 能量密度与压强输出为自然单位 (fm$^{-4}$), 通常需要乘以 $\hbar c$ 转换为 MeV/fm$^3$。
"""

import math

import numpy as np
from scipy import optimize

from RMFCalculator.eos.unit import (
    m_n, m_e, m_l, Matrix_b, Matrix_l, J_B, b_B, fm_MeV,
)
from RMFCalculator.eos.RMF_INEOS import (
    _validate_theta13, _unpack_theta13,
)


def initial_values(rho, theta):
    r"""
    在给定重子数密度 $\rho$ 下估算 6 个变量初值 (供求解器起步)。

    Args:
        rho (float): 重子数密度 (fm$^{-3}$)
        theta (array): 13 元 RMF 参数向量 $\theta_{13}$

    Returns:
        tuple: $(\sigma, \omega, \rho_{03}, \delta, \mu_n, \mu_e)$
    """
    theta13 = _validate_theta13(theta)
    (m_sig, m_w, m_rho, m_delta, g_sigma, g_omega, g_rho,
     g_delta, _kappa, _lambda_0, _zeta, Lambda_w, _Lambda_sd) = _unpack_theta13(theta13)

    rho_0 = 0.1505  # 参考饱和密度 (fm$^{-3}$)

    # $\sigma$ 场线性近似: $\sigma \approx g_\sigma \rho / m_\sigma^2$
    sigma = g_sigma * rho / (m_sig**2)
    # $\rho_{03}$ 场线性近似 (丰中子物质 $\rho_p - \rho_n < 0$):
    # $\rho_{03} \approx -g_\rho \rho / (2 m_\rho^2)$
    rho_03 = -g_rho * rho / (2.0 * m_rho**2)
    # $\omega$ 场线性近似 (含 $\Lambda_\omega$ 交叉耦合):
    # $\omega \approx \rho / [m_\omega^2/g_\omega + 2\Lambda_\omega (g_\rho\rho_{03})^2 g_\omega]$
    omega = rho * (
        (((m_w**2) / g_omega) + (2.0 * Lambda_w * ((g_rho * rho_03)**2) * g_omega)) ** (-1.0)
    )
    # $\delta$ 场线性近似 (丰中子物质 $n_p^s - n_n^s < 0$):
    # $\delta \approx -g_\delta \rho / (2 m_\delta^2)$
    delta = -g_delta * rho * 0.5 / (m_delta**2)

    # 核子有效质量 (对称近似): $m^* = m_n - g_\sigma\sigma$
    m_eff = m_n - g_sigma * sigma
    # 中子化学势初值: $\mu_n \approx m^* + g_\omega\omega + g_\rho\rho_{03} I_3^n$
    mu_n = m_eff + g_omega * omega + g_rho * rho_03 * Matrix_b[1, 2]
    # 电子化学势初值 (超相对论费米气体近似):
    # $\mu_e \approx 0.12 m_e (\rho/\rho_0)^{2/3}$
    mu_e = 0.12 * m_e * (rho / rho_0) ** (2 / 3.0)

    return sigma, omega, rho_03, delta, mu_n, mu_e


def functie(x, args):
    r"""
    RMF 自洽方程残差向量 $f(\sigma, \omega, \rho_{03}, \delta, \mu_n, \mu_e) = 0$。

    Args:
        x (array): $[\sigma, \omega, \rho_{03}, \delta, \mu_n, \mu_e]$
        args (array): theta13 13 元 + 密度 $\rho$ (共 14 元)

    Returns:
        list: 6 个原始残差 $f_i$ (供 LM 方法使用, LM 内部会平方)。
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
        # 核子化学势: $\mu_b = b_B \mu_n - q_B \mu_e$
        # 质子: $\mu_p = \mu_n - \mu_e$, 中子: $\mu_n = \mu_n$
        mu_b = Matrix_b[i, 0] * mu_n - Matrix_b[i, 1] * mu_e
        # 核子动能费米能: $E_F^b = \mu_b - g_\omega\omega - g_\rho\rho_{03} I_3^i$
        E_fb = mu_b - g_omega * omega - g_rho * rho_03 * I3[i]
        # 费米动量: $k_F^b = \sqrt{E_F^{b2} - M_i^{*2}}$
        k_fb_sq = E_fb**2 - mf**2
        if k_fb_sq < 0:
            k_fb_sq = np.clip(k_fb_sq, a_min=0.0, a_max=None)
            E_fb = mf
        k_fb = math.sqrt(k_fb_sq)

        # 重子数密度: $\rho_B = (2J_B+1) b_B k_F^3 / (6\pi^2)$
        rho_B = ((2 * J_B) + 1) * b_B * k_fb**3 / (6.0 * math.pi**2)
        rho_B_list.append(rho_B)

        # 标量密度: $n_J^s = \frac{M_J^*}{2\pi^2}\left[E_F k_F - M_J^{*2}\ln\frac{E_F + k_F}{M_J^*}\right]$
        log_arg = (E_fb + k_fb) / mf
        if log_arg < 1.0:
            log_arg = 1.0
        rho_SB = (mf / (2.0 * math.pi**2)) * (
            E_fb * k_fb - mf**2 * np.log(log_arg)
        )
        rho_SB_list.append(rho_SB)

        # 电荷密度: $Q_B = (2J_B+1) q_B k_F^3 / (6\pi^2)$
        Q_B = ((2.0 * J_B) + 1.0) * Matrix_b[i, 1] * k_fb**3 / (6.0 * math.pi**2)
        q_list.append(Q_B)

    rho_p = rho_B_list[0]
    rho_n = rho_B_list[1]
    rho_s_p = rho_SB_list[0]
    rho_s_n = rho_SB_list[1]
    rho_total = rho_p + rho_n

    # ---------- 轻子 (电子 j=0, μ子 j=1) ----------
    # 轻子作为自由费米气体处理, 化学势由 β 平衡条件确定:
    # $\mu_e = \mu_\mu = \mu_n - \mu_p$
    for j in range(2):
        # 轻子化学势: $\mu_l = b_l \mu_n - q_l \mu_e$
        # 电子/μ子: $b_l = 0, q_l = -1 \Rightarrow \mu_l = \mu_e$
        mu_l = Matrix_l[j, 0] * mu_n - Matrix_l[j, 1] * mu_e
        E_fl = mu_l
        # 轻子费米动量: $k_F^l = \sqrt{\mu_l^2 - m_l^2}$
        k_fl_sq = E_fl**2 - m_l[j]**2
        k_fl_sq = np.clip(k_fl_sq, a_min=0.0, a_max=None)
        k_fl = math.sqrt(k_fl_sq)

        # 轻子电荷密度: $Q_L = (2J_B+1) q_l k_F^3 / (6\pi^2)$
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
        ),
        # (2) ω 场方程
        (
            omega * m_w**2 / g_omega
            - rho_total
            + zeta * (g_omega * omega)**3 / 6
            + 2 * Lambda_w * g_omega * omega * (rho_03 * g_rho)**2
        ),
        # (3) ρ 场方程
        (
            rho_03 * m_rho**2 / g_rho
            - 0.5 * (rho_p - rho_n)
            + 2 * Lambda_w * g_rho * rho_03 * (omega * g_omega)**2
        ),
        # (4) δ 场方程
        # 乘以 $g_\delta$ 以避免 $g_\delta = 0$ (无 $\delta$ 介子) 时除零, 此时方程退化为
        # $\delta m_\delta^2 = 0 \Rightarrow \delta = 0$, 与无 $\delta$ 模型一致。
        (
            delta * m_delta**2
            - g_delta * (rho_s_p - rho_s_n)
            - Lambda_sigma_delta * g_delta * (g_delta * delta) * (g_sigma * sigma)**2
        ),
        # (5) 重子数守恒: $\rho = \rho_p + \rho_n$
        (rho - rho_total),
        # (6) 电荷中性: $\sum_B Q_B + \sum_L Q_L = 0$
        (sum(q_list)),
    ]

    return f


def Energy_density_Pressure(x, rho, theta):
    r"""
    计算 β 平衡物质 (含/不含 δ 介子) 的能量密度与压强。

    不含 $\delta$ 介子的情形由同一套代码处理 ($g_\delta = \Lambda_{\sigma\delta} = 0$)。

    Args:
        x (array): $[\sigma, \omega, \rho_{03}, \delta, \mu_n, \mu_e]$
        rho (float): 重子数密度 (fm$^{-3}$)
        theta (array): 13 元 RMF 参数向量 $\theta_{13}$

    Returns:
        list: $[\rho, \varepsilon, P, \mu_p, \mu_n, \mu_e, \mu_\mu, x_p]$
    """
    theta13 = _validate_theta13(theta)
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

    # ---------- 重子 (质子 i=0, 中子 i=1) ----------
    for i in range(2):
        mf = m_eff_list[i]
        if mf <= 0.0:   # 负有效质量: 空费米海
            continue
        # 核子化学势: $\mu_b = b_B \mu_n - q_B \mu_e$
        mu_b = Matrix_b[i, 0] * mu_n - Matrix_b[i, 1] * mu_e
        composition[i] = mu_b

        # 核子动能费米能: $E_F^b = \mu_b - g_\omega\omega - g_\rho\rho_{03} I_3^i$
        E_fb = mu_b - g_omega * omega - g_rho * rho_03 * I3[i]
        # 费米动量: $k_F^b = \sqrt{E_F^{b2} - M_i^{*2}}$
        k_fb_sq = E_fb**2 - mf**2
        if k_fb_sq < 0:
            k_fb_sq = 0.0
            E_fb = mf
        k_fb = math.sqrt(k_fb_sq)

        # 重子数密度: $\rho_B = (2J_B+1) b_B k_F^3 / (6\pi^2)$
        rho_B = ((2.0 * J_B) + 1.0) * b_B * k_fb**3 / (6.0 * math.pi**2)

        # 质子分数: $x_p = \rho_p / \rho$
        if i == 0:
            composition[4] = rho_B / rho

        # $\sum \mu_B \rho_B$ (用于压强计算)
        multi = multi + mu_b * rho_B

        # 核子动能密度:
        # $\varepsilon_B^{\text{kin}} = \frac{1}{8\pi^2}\left[k_F E_F^3 + k_F^3 E_F - M^{*4}\ln\frac{k_F + E_F}{M^*}\right]$
        energy_baryon = (1 / (8.0 * math.pi**2)) * (
            k_fb * E_fb**3 + k_fb**3 * E_fb
            - mf**4 * np.log((k_fb + E_fb) / mf)
        )
        energy_b = energy_b + energy_baryon

    # ---------- 轻子 (电子 j=0, μ子 j=1) ----------
    # 轻子作为自由费米气体, 化学势由 β 平衡条件 $\mu_e = \mu_\mu = \mu_n - \mu_p$ 确定
    for j in range(2):
        mu_l = Matrix_l[j, 0] * mu_n - Matrix_l[j, 1] * mu_e
        composition[2 + j] = mu_l

        # 轻子费米动量: $k_F^l = \sqrt{\mu_l^2 - m_l^2}$
        k_fl_sq = mu_l**2 - m_l[j]**2
        if k_fl_sq < 0.0:
            k_fl_sq = 0.0
        k_fl = math.sqrt(k_fl_sq)

        # 轻子数密度: $\rho_l = k_F^3 / (3\pi^2)$
        rho_l = k_fl**3 / (3.0 * math.pi**2)
        multi = multi + mu_l * rho_l

        # 轻子动能密度:
        # $\varepsilon_l^{\text{kin}} = \frac{1}{8\pi^2}\left[k_F \mu_l^3 + \mu_l k_F^3 - m_l^4\ln\frac{k_F + \mu_l}{m_l}\right]$
        energy_lepton = (1 / (8.0 * math.pi**2)) * (
            k_fl * mu_l**3 + mu_l * k_fl**3
            - m_l[j]**4 * np.log((k_fl + mu_l) / m_l[j])
        )
        energy_l = energy_l + energy_lepton

    # ---------- 介子场能量 ----------
    # $\sigma$ 场能量 (含 $\sigma$ 自相互作用及 $\sigma$-$\delta$ 交叉耦合):
    # $\mathcal{E}_\sigma = \frac12 m_\sigma^2\sigma^2 + \frac{\kappa}{6}(g_\sigma\sigma)^3 + \frac{\lambda_0}{24}(g_\sigma\sigma)^4
    #   - \frac12 \Lambda_{\sigma\delta} (g_\sigma\sigma)^2 (g_\delta\delta)^2$
    sigma_terms = (
        0.5 * (sigma * m_sig)**2
        + kappa * (g_sigma * sigma)**3 / 6
        + lambda_0 * (g_sigma * sigma)**4 / 24
        - 0.5 * Lambda_sigma_delta * (g_sigma * sigma)**2 * (g_delta * delta)**2
    )
    # $\omega$ 场能量: $\mathcal{E}_\omega = \frac12 m_\omega^2\omega^2 + \frac{\zeta}{8}(g_\omega\omega)^4$
    omega_terms = 0.5 * (omega * m_w)**2 + zeta * (g_omega * omega)**4 / 8
    # $\rho$ 场能量: $\mathcal{E}_\rho = \frac12 m_\rho^2\rho_{03}^2 + 3\Lambda_\omega (g_\rho\rho_{03} g_\omega\omega)^2$
    rho_terms = 0.5 * (rho_03 * m_rho)**2 + 3 * Lambda_w * (g_rho * rho_03 * g_omega * omega)**2
    # $\delta$ 场能量: $\mathcal{E}_\delta = \frac12 m_\delta^2 \delta^2$
    delta_terms = 0.5 * (delta * m_delta)**2

    # 总能量密度: $\mathcal{E} = \mathcal{E}_B^{\text{kin}} + \mathcal{E}_l^{\text{kin}} + \mathcal{E}_\sigma + \mathcal{E}_\omega + \mathcal{E}_\rho + \mathcal{E}_\delta$
    energy_density = energy_b + energy_l + sigma_terms + omega_terms + rho_terms + delta_terms
    # 压强: $P = \sum_B \mu_B \rho_B + \sum_l \mu_l \rho_l - \mathcal{E}$
    Pressure = multi - energy_density

    return [rho, energy_density, Pressure] + composition


def compute_EOS(theta, n_points=124, dt=0.05, rho_0=0.1505):
    r"""
    计算电中性 β 平衡 (npeμ) 中子星物质状态方程 (含/不含 δ 介子)。

    不含 $\delta$ 介子的模型 ($g_\delta = \Lambda_{\sigma\delta} = 0$) 是含 $\delta$ 介子模型
    的特例, 统一使用 13 元参数向量 $\theta_{13}$。

    Args:
        theta (array): 13 元 RMF 参数向量 $\theta_{13}$
        n_points (int): 密度点数, 默认 124
        dt (float): 密度步长因子, 默认 0.05
        rho_0 (float): 参考饱和密度, 默认 0.1505 fm$^{-3}$

    Returns:
        ndarray: $(n_\text{points}, 8)$ 数组:
            [0] $\rho$ (fm$^{-3}$)
            [1] $\varepsilon$ (fm$^{-4}$)
            [2] $P$ (fm$^{-4}$)
            [3-6] $\mu_p, \mu_n, \mu_e, \mu_\mu$ (fm$^{-1}$)
            [7] 质子分数 $x_p$
    """
    theta13 = _validate_theta13(theta)

    # 在最低密度点 $0.1\rho_0$ 处估算初值
    x_init = np.array(initial_values(0.1 * rho_0, theta13))

    EoS = [[] for _ in range(n_points)]

    # 从低密度到高密度逐点求解, 以上一点解作为下一点初值 (延续法)
    for i in range(n_points):
        # 密度网格: $\rho_i = 0.1\rho_0 + i \cdot dt \cdot \rho_0$
        rho = 0.1 * rho_0 + dt * i * rho_0

        # 参数向量: [theta13, rho] 共 14 元
        arg = np.array([*theta13, rho])

        # 使用 Levenberg-Marquardt 方法求解非线性方程组
        sol = optimize.root(functie, x_init, method="lm", args=arg)

        if not sol.success:
            print(f"Warning: solver failed at i={i}, rho={rho:.4f}")

        # 计算能量密度和压强
        Re = Energy_density_Pressure(sol.x, rho, theta13)
        EoS[i] = Re

        # 延续法: 当前解作为下一密度点的初值
        x_init = sol.x

    return np.array(EoS)


if __name__ == "__main__":
    from RMFCalculator.eos.parameters import get_theta

    theta13_test = get_theta("FSU_DELTA62")
    print("Computing beta-equilibrium EOS with FSU-δ6.2...")

    EoS_test = compute_EOS(theta13_test, n_points=30)

    eps_test = EoS_test[:, 1] * fm_MeV
    P_test = EoS_test[:, 2] * fm_MeV
    rho_test = EoS_test[:, 0]
    xp_test = EoS_test[:, 7]

    print(f"{'rho(fm-3)':<12}{'eps(MeV/fm3)':<14}{'P(MeV/fm3)':<14}{'x_p':<10}")
    for k in range(0, 30, 3):
        print(f"{rho_test[k]:<12.4f}{eps_test[k]:<14.4f}{P_test[k]:<14.4f}{xp_test[k]:<10.4f}")