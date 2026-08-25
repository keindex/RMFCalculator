r"""
RMF_INEOS.py - 无限核物质状态方程计算 (含/不含同位旋矢量-标量介子 $\delta$)

基于 FSUGold-δ 模型 (学位论文第四章 / 论文 arXiv:2207.04406 的 FSU-δ6.7, FSU-δ6.2),
在 FSUGold 基础上引入 $\delta$ 介子及 $\sigma$-$\delta$ 交叉耦合:

$$\mathcal{L}_{\delta} \supset g_\delta\, \vec{\delta}\cdot\bar\psi\vec{\tau}\psi
  + \tfrac12\left(\partial_\mu\vec{\delta}\cdot\partial^\mu\vec{\delta} - m_\delta^2\vec{\delta}\cdot\vec{\delta}\right)
  + \tfrac12 \Lambda_{\sigma\delta}\, (g_\sigma\sigma)^2 (g_\delta\vec{\delta})^2$$

静态均匀平均场下场方程 (论文式 4.8-4.11):

$$
\begin{aligned}
m_\sigma^2\bar\sigma &= g_\sigma\left[n^s - b_\sigma M(g_\sigma\bar\sigma)^2 - c_\sigma(g_\sigma\bar\sigma)^3
  + \Lambda_{\sigma\delta}(g_\sigma\bar\sigma)(g_\delta\bar\delta)^2\right] \\
m_\omega^2\bar\omega &= g_\omega\left[n - c_\omega(g_\omega\bar\omega)^3
  - \lambda_v(g_\omega\bar\omega)(g_\rho\bar\rho)^2\right] \\
m_\rho^2\bar\rho &= g_\rho\left[(n_p - n_n) - \lambda_v(g_\rho\bar\rho)(g_\omega\bar\omega)^2\right] \\
m_\delta^2\bar\delta &= g_\delta\left[(n_p^s - n_n^s)
  + \Lambda_{\sigma\delta}(g_\delta\bar\delta)(g_\sigma\bar\sigma)^2\right]
\end{aligned}
$$

核子 Dirac 有效质量劈裂:

$$M_J^* = M - g_\sigma\bar\sigma - g_\delta\bar\delta\, \tau_3^J, \qquad J = p, n$$

能量密度与压强包含 $\delta$ 场项 (论文式 4.12-4.13)。

本模块求解变量 $x = [\sigma, \omega, \rho_{03}, \delta, \mu_n, \mu_p]$ (6 个)。
**不含 $\delta$ 介子的模型 ($g_\delta = \Lambda_{\sigma\delta} = 0$) 是本模型的特例**:
此时 $\delta$ 场方程退化为 $\delta = 0$, 有效质量劈裂消失, 与无 $\delta$ 模型完全一致。
因此本模块统一作为两种模型的求解器, 统一接受 13 元参数向量 $\theta_{13}$:

$$\theta_{13} = [m_\sigma, m_\omega, m_\rho, m_\delta, g_\sigma, g_\omega, g_\rho, g_\delta, \kappa, \lambda_0, \zeta, \Lambda_\omega, \Lambda_{\sigma\delta}]$$

约定:
- 本模块采用与 ``RMF_betaEOS`` 相同的 "大 $g_\rho$" 约定,
  即 $\rho$ 场方程源项为 $\tfrac12(\rho_p - \rho_n)$;
- 论文式 (4.9)/(4.10) 中的 $\lambda_v$ 在此即 $\Lambda_\omega$。
"""

import math

import numpy as np
from scipy import optimize

from RMFCalculator.eos.unit import m_n, Matrix_b, fm_MeV


def _validate_theta13(theta):
    r"""
    校验并返回 13 元含 $\delta$ 介子的 RMF 参数向量 $\theta_{13}$。

    $$\theta_{13} = [m_\sigma, m_\omega, m_\rho, m_\delta, g_\sigma, g_\omega, g_\rho, g_\delta, \kappa, \lambda_0, \zeta, \Lambda_\omega, \Lambda_{\sigma\delta}]$$
    """
    theta = np.asarray(theta, dtype=np.float64)
    if theta.shape != (13,):
        raise TypeError(
            "RMF_INEOS expects a length-13 numeric RMF theta13."
        )
    return theta


def _unpack_theta13(theta):
    r"""解包 13 元参数向量为标量量, 返回:
    ``m_sig, m_w, m_rho, m_delta, g_sigma, g_omega, g_rho, g_delta, kappa, lambda_0, zeta, Lambda_w, Lambda_sigma_delta``
    """
    (m_sig, m_w, m_rho, m_delta, g_sigma, g_omega, g_rho,
     g_delta, kappa, lambda_0, zeta, Lambda_w, Lambda_sigma_delta) = theta
    return (m_sig, m_w, m_rho, m_delta, g_sigma, g_omega, g_rho,
            g_delta, kappa, lambda_0, zeta, Lambda_w, Lambda_sigma_delta)


def initial_values(rho, theta, proton_fraction=0.5):
    r"""
    在给定重子数密度 $\rho$ 下估算 6 个场/化学势初值 (线性化平均场近似)。

    对称核物质 ($x_p = 0.5$) 时 $\bar\delta = 0$, 与无 $\delta$ 情形一致。

    Args:
        rho (float): 重子数密度 (fm$^{-3}$)
        theta (array): 13 元 RMF 参数向量 $\theta_{13}$
        proton_fraction (float): 质子分数, 默认 0.5

    Returns:
        tuple: $(\sigma,\, \omega,\, \rho_{03},\, \delta,\, \mu_n,\, \mu_p)$
    """
    theta13 = _validate_theta13(theta)
    (m_sig, m_w, m_rho, m_delta, g_sigma, g_omega, g_rho,
     g_delta, _kappa, _lambda_0, _zeta, Lambda_w, _Lambda_sd) = _unpack_theta13(theta13)

    # $\sigma$ 场线性近似: $\sigma \approx g_\sigma \rho / m_\sigma^2$
    sigma = g_sigma * rho / (m_sig**2)
    # $\rho_{03}$ 场线性近似: $\rho_{03} \approx -g_\rho \rho (1/2 - x_p) / m_\rho^2$
    rho_03 = -g_rho * rho * (0.5 - proton_fraction) / (m_rho**2)
    # $\omega$ 场线性近似 (含 $\Lambda_\omega$ 交叉耦合):
    # $\omega \approx \rho / [m_\omega^2/g_\omega + 2\Lambda_\omega (g_\rho\rho_{03})^2 g_\omega]$
    omega = rho / (m_w**2 / g_omega + 2.0 * Lambda_w * (g_rho * rho_03)**2 * g_omega)
    # $\delta$ 场线性近似: $\delta \approx g_\delta (n_p^s - n_n^s) / m_\delta^2$
    # 在非对称核物质中 $n_p^s - n_n^s \approx \rho (2x_p - 1)/2$
    delta = g_delta * rho * (2 * proton_fraction - 1.0) * 0.5 / (m_delta**2)

    # 核子有效质量 (对称近似 $m^* = m_n - g_\sigma\sigma$)
    m_eff = m_n - g_sigma * sigma
    # 费米动量: $k_F = (3\pi^2 \rho)^{1/3}$, 单位: fm$^{-1}$
    k_f = (3 * math.pi**2 * rho)**(1 / 3)
    # 费米能量: $E_F = \sqrt{k_F^2 + m^{*2}}$, 单位: fm$^{-1}$
    E_F = math.sqrt(k_f**2 + m_eff**2)

    # 中子化学势: $\mu_n = E_F + g_\omega\omega + g_\rho\rho_{03} I_3^n$, $I_3^n = -1/2$
    mu_n = E_F + g_omega * omega + g_rho * rho_03 * Matrix_b[1, 2]
    # 质子化学势: $\mu_p = E_F + g_\omega\omega + g_\rho\rho_{03} I_3^p$, $I_3^p = +1/2$
    mu_p = E_F + g_omega * omega + g_rho * rho_03 * Matrix_b[0, 2]

    return sigma, omega, rho_03, delta, mu_n, mu_p


def functie(x, args):
    r"""
    RMF 平均场自洽方程残差向量 $f(\sigma, \omega, \rho_{03}, \delta, \mu_n, \mu_p) = 0$。

    Args:
        x (array): $[\sigma, \omega, \rho_{03}, \delta, \mu_n, \mu_p]$
        args (array): theta13 的 13 参数 + 密度 $\rho$ + 质子分数 $x_p$ (共 15 元)

    Returns:
        list: 6 个原始残差 $f_i$ (供 scipy.optimize.root 的 LM 方法使用)。
        LM 方法内部会对残差平方求和, 因此不能在此再平方。
    """
    (m_sig, m_w, m_rho, m_delta, g_sigma, g_omega, g_rho,
     g_delta, kappa, lambda_0, zeta, Lambda_w, Lambda_sigma_delta) = _unpack_theta13(args[:13])
    rho = args[13]
    x_p = args[14]

    sigma, omega, rho_03, delta, mu_n, mu_p = x
    mu_B_list = [mu_p, mu_n]

    # δ 介子耦合的同位旋因子: 论文式 (4.3a) 中 $\tau_3^J$, 质子 $+1$, 中子 $-1$
    # (注意: 这是 $\tau_3$ 而非 $I_3=\pm1/2$, 见论文式 4.3a/4.58)
    tau3 = np.array([1.0, -1.0])

    # 核子 Dirac 有效质量: $M_J^* = M - g_\sigma\sigma - g_\delta\delta\, \tau_3^J$
    m_eff_p = m_n - g_sigma * sigma - g_delta * delta * tau3[0]
    m_eff_n = m_n - g_sigma * sigma - g_delta * delta * tau3[1]
    m_eff_list = [m_eff_p, m_eff_n]

    rho_B_list = []
    rho_SB_list = []

    # ρ 介子矢量耦合仍用同位旋第三分量 $I_3 = \pm 1/2$ (与 RMF_INEOS 一致)
    I3 = Matrix_b[:, 2]

    for i in range(2):
        mf = m_eff_list[i]
        # 负有效质量分支 (Dirac 质量劈裂极端化, 如纯中子物质中的质子):
        # 无物理费米海, 贡献置 0, 避免 log(负值) 产生 NaN
        if mf <= 0.0:
            rho_B_list.append(0.0)
            rho_SB_list.append(0.0)
            continue
        E_fb = mu_B_list[i] - g_omega * omega - g_rho * rho_03 * I3[i]
        k_fb_sq = E_fb**2 - mf**2
        if k_fb_sq < 0:
            k_fb_sq = np.clip(k_fb_sq, a_min=0.0, a_max=None)
            E_fb = mf
        k_fb = math.sqrt(k_fb_sq)

        # 重子数密度: $\rho_B = k_F^3 / (3\pi^2)$, 单位: fm$^{-3}$
        rho_B = k_fb**3 / (3 * math.pi**2)
        rho_B_list.append(rho_B)

        # 标量密度: $n_J^s = \frac{M_J^*}{2\pi^2}\left[E_F k_F - M_J^{*2}\ln\frac{E_F + k_F}{M_J^*}\right]$
        rho_SB = (mf / (2 * math.pi**2)) * (
            E_fb * k_fb - mf**2 * np.log((E_fb + k_fb) / mf)
        )
        rho_SB_list.append(rho_SB)

    rho_p = rho_B_list[0]
    rho_n = rho_B_list[1]
    rho_s_p = rho_SB_list[0]
    rho_s_n = rho_SB_list[1]
    rho_total = rho_p + rho_n

    f = [
        # (1) $\sigma$ 场方程 (含 $\Lambda_{\sigma\delta}$ 项)
        (
            sigma * m_sig**2 / g_sigma
            - (rho_s_p + rho_s_n)
            + kappa * (g_sigma * sigma)**2 / 2
            + lambda_0 * (g_sigma * sigma)**3 / 6
            - Lambda_sigma_delta * (g_sigma * sigma) * (g_delta * delta)**2
        ),
        # (2) $\omega$ 场方程
        (
            omega * m_w**2 / g_omega
            - rho_total
            + zeta * (g_omega * omega)**3 / 6
            + 2 * Lambda_w * g_omega * omega * (rho_03 * g_rho)**2
        ),
        # (3) $\rho$ 场方程
        (
            rho_03 * m_rho**2 / g_rho
            - 0.5 * (rho_p - rho_n)
            + 2 * Lambda_w * g_rho * rho_03 * (omega * g_omega)**2
        ),
        # (4) $\delta$ 场方程 (含 $\Lambda_{\sigma\delta}$ 项)
        # 乘以 $g_\delta$ 以避免 $g_\delta = 0$ (无 $\delta$ 介子) 时除零, 此时方程退化为
        # $\delta m_\delta^2 = 0 \Rightarrow \delta = 0$, 与无 $\delta$ 模型一致。
        (
            delta * m_delta**2
            - g_delta * (rho_s_p - rho_s_n)
            - Lambda_sigma_delta * g_delta * (g_delta * delta) * (g_sigma * sigma)**2
        ),
        # (5) 重子数守恒: $\rho = \rho_p + \rho_n$
        (rho - rho_total),
        # (6) 质子分数约束: $x_p = \rho_p / \rho$
        (x_p - rho_p / rho),
    ]

    return f


def Energy_density_Pressure(x, rho, theta, x_p=0.5):
    r"""
    计算无限核物质(含/不含 $\delta$ 介子)的能量密度与压强。

    不含 $\delta$ 介子的情形 ($g_\delta = \Lambda_{\sigma\delta} = 0$) 由同一套代码处理。

    Args:
        x (array): $[\sigma, \omega, \rho_{03}, \delta, \mu_n, \mu_p]$
        rho (float): 重子数密度 (fm$^{-3}$)
        theta (array): 13 元 RMF 参数向量 $\theta_{13}$
        x_p (float): 质子分数, 默认 0.5

    Returns:
        list: $[\rho, \varepsilon, P, \mu_n, \mu_p, x_p]$
    """
    theta13 = _validate_theta13(theta)
    sigma, omega, rho_03, delta, mu_n, mu_p = x
    (m_sig, m_w, m_rho, m_delta, g_sigma, g_omega, g_rho,
     g_delta, kappa, lambda_0, zeta, Lambda_w, Lambda_sigma_delta) = _unpack_theta13(theta13)

    mu_B_list = [mu_p, mu_n]
    tau3 = np.array([1.0, -1.0])
    I3 = Matrix_b[:, 2]   # ρ 介子耦合: I₃ = ±1/2

    m_eff_p = m_n - g_sigma * sigma - g_delta * delta * tau3[0]
    m_eff_n = m_n - g_sigma * sigma - g_delta * delta * tau3[1]
    m_eff_list = [m_eff_p, m_eff_n]

    energy_b_list = []
    rho_B_list = []

    for i in range(2):
        mf = m_eff_list[i]
        if mf <= 0.0:   # 负有效质量: 空费米海
            rho_B_list.append(0.0)
            energy_b_list.append(0.0)
            continue
        E_fb = mu_B_list[i] - g_omega * omega - g_rho * rho_03 * I3[i]
        k_fb_sq = E_fb**2 - mf**2
        if k_fb_sq < 0:
            k_fb_sq = 0.0
            E_fb = mf
        k_fb = math.sqrt(k_fb_sq)

        # 重子数密度: $\rho_B = k_F^3 / (3\pi^2)$, 单位: fm$^{-3}$
        rho_B = k_fb**3 / (3 * math.pi**2)
        rho_B_list.append(rho_B)

        # 核子动能密度 (相对论费米气体):
        # $\varepsilon_B^{\text{kin}} = \frac{1}{8\pi^2}\left[k_F E_F^3 + k_F^3 E_F - M^{*4}\ln\frac{k_F + E_F}{M^*}\right]$
        energy_baryon = (1 / (8 * math.pi**2)) * (
            k_fb * E_fb**3 + k_fb**3 * E_fb - mf**4 * np.log((k_fb + E_fb) / mf)
        )
        energy_b_list.append(energy_baryon)

    # $\sigma$ 场能量 (含 $\sigma$ 自相互作用 $\kappa, \lambda_0$ 及 $\sigma$-$\delta$ 交叉耦合 $\Lambda_{\sigma\delta}$):
    
    # $\mathcal{E}_\sigma = \frac12 m_\sigma^2\sigma^2 + \frac{\kappa}{6}(g_\sigma\sigma)^3 + \frac{\lambda_0}{24}(g_\sigma\sigma)^4 - \frac12 \Lambda_{\sigma\delta} (g_\sigma\sigma)^2 (g_\delta\delta)^2$
    sigma_terms = (
        0.5 * (sigma * m_sig)**2
        + kappa * (g_sigma * sigma)**3 / 6
        + lambda_0 * (g_sigma * sigma)**4 / 24
        - 0.5 * Lambda_sigma_delta * (g_sigma * sigma)**2 * (g_delta * delta)**2
    )

    # $\omega$ 场能量 (含 $\omega$ 四次自耦合 $\zeta$):
    # $\mathcal{E}_\omega = \frac12 m_\omega^2\omega^2 + \frac{\zeta}{8}(g_\omega\omega)^4$
    omega_terms = 0.5 * (omega * m_w)**2 + zeta * (g_omega * omega)**4 / 8

    # $\rho$ 场能量 (含 $\omega$-$\rho$ 交叉耦合 $\Lambda_\omega$):
    # $\mathcal{E}_\rho = \frac12 m_\rho^2\rho_{03}^2 + 3\Lambda_\omega (g_\rho\rho_{03} g_\omega\omega)^2$
    rho_terms = 0.5 * (rho_03 * m_rho)**2 + 3 * Lambda_w * (g_rho * rho_03 * g_omega * omega)**2

    # $\delta$ 场能量: $\mathcal{E}_\delta = \frac12 m_\delta^2 \delta^2$
    delta_terms = 0.5 * (delta * m_delta)**2

    # 总能量密度: $\mathcal{E} = \sum_J \mathcal{E}_J^{\text{kin}} + \mathcal{E}_\sigma + \mathcal{E}_\omega + \mathcal{E}_\rho + \mathcal{E}_\delta$
    energy_density = sum(energy_b_list) + sigma_terms + omega_terms + rho_terms + delta_terms

    # 压强 (热力学关系): $P = \sum_J \mu_J \rho_J - \mathcal{E}$
    Pressure = sum(mu_B_list[i] * rho_B_list[i] for i in range(2)) - energy_density

    return [rho, energy_density, Pressure, mu_n, mu_p, x_p]


def compute_INEOS(theta, x_p=0.5, n_points=124, dt=0.05, rho_0=0.1505):
    r"""
    计算含/不含 $\delta$ 介子的无限核物质状态方程。

    不含 $\delta$ 介子的模型 ($g_\delta = \Lambda_{\sigma\delta} = 0$) 是含 $\delta$ 介子模型
    的特例, 统一使用 13 元参数向量 $\theta_{13}$。

    Args:
        theta (array): 13 元 RMF 参数向量 $\theta_{13}$
        x_p (float): 质子分数, 默认 0.5 (对称核物质)
        n_points (int): 密度点数量, 默认 124
        dt (float): 密度步长因子, 默认 0.05
        rho_0 (float): 参考饱和密度, 默认 0.1505 fm$^{-3}$

    Returns:
        ndarray: $(6, n_\text{points})$ 数组, 行:
            [0] 重子数密度 $\rho$ (fm$^{-3}$)
            [1] 能量密度 $\varepsilon$ (fm$^{-4}$, 自然单位)
            [2] 压强 $P$ (fm$^{-4}$, 自然单位)
            [3] 中子化学势 $\mu_n$ (fm$^{-1}$)
            [4] 质子化学势 $\mu_p$ (fm$^{-1}$)
            [5] 质子分数 $x_p$
        (乘以 ``fm_MeV`` 即得 MeV/fm$^3$)
    """
    theta13 = _validate_theta13(theta)

    # 在最低密度点 $0.1\rho_0$ 处估算初值
    x_init = np.array(initial_values(0.1 * rho_0, theta13, x_p))

    EoS = [[] for _ in range(n_points)]

    # 从低密度到高密度逐点求解, 以上一点解作为下一点初值 (延续法)
    for i in range(n_points):
        # 密度网格: $\rho_i = 0.1\rho_0 + i \cdot dt \cdot \rho_0$
        rho = 0.1 * rho_0 + dt * i * rho_0

        # 参数向量: [theta13, rho, x_p] 共 15 元
        arg = np.array([*theta13, rho, x_p])

        # 使用 Levenberg-Marquardt 方法求解非线性方程组
        sol = optimize.root(functie, x_init, method="lm", args=arg)

        if not sol.success:
            print(f"Warning: solver failed at i={i}, rho={rho:.4f}")

        # 计算能量密度和压强
        Re = Energy_density_Pressure(sol.x, rho, theta13, x_p)
        EoS[i] = Re

        # 延续法: 当前解作为下一密度点的初值
        x_init = sol.x

    return np.array(EoS)


if __name__ == "__main__":
    from RMFCalculator.eos.parameters import get_theta

    theta13_test = get_theta("FSU_DELTA67")
    print("theta13:", theta13_test)
    print("Computing symmetric nuclear matter EOS with FSU-δ6.7...")

    EoS_test = compute_INEOS(theta13_test, x_p=0.5, n_points=20, dt=0.05)

    eps_test = EoS_test[:, 1] * fm_MeV
    P_test = EoS_test[:, 2] * fm_MeV
    rho_test = EoS_test[:, 0]
    binding_test = eps_test / rho_test - m_n * fm_MeV
    print(f"{'rho(fm-3)':<12}{'eps(fm-4)':<14}{'EA(MeV)':<14}{'P(fm-4)':<14}")
    for k in range(10):
        print(f"{rho_test[k]:<12.4f}{eps_test[k]:<14.4f}{binding_test[k]:<14.4f}{P_test[k]:<14.4f}")