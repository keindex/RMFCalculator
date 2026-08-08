from RMFCalculator.tov.unit import g_cm_3, dyn_cm_2
from RMFCalculator.eos.unit import *
from scipy import optimize
import numpy as np
import math



def _validate_rmf_theta(theta):
    r"""
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

    $\sigma \approx \frac{g_\sigma \rho}{m_\sigma^2},\quad \rho_{03} \approx -\frac{g_\rho \rho}{2 m_\rho^2}$

    Args:
        rho (float): 重子数密度 $\rho$, 单位: fm$^{-3}$
        theta (array): RMF 模型 10 个参数
            $[m_\sigma, m_\omega, m_\rho, g_\sigma, g_\omega, g_\rho, \kappa, \lambda_0, \zeta, \Lambda_\omega]$

    Returns:
        tuple: $(\sigma, \omega, \rho_{03}, \mu_n, \mu_e)$
            - $\sigma$: 标量介子场
            - $\omega$: 矢量介子场
            - $\rho_{03}$: 同位旋矢量介子场第三分量
            - $\mu_n$: 中子化学势, 单位: fm$^{-1}$
            - $\mu_e$: 电子化学势, 单位: fm$^{-1}$
    """
    m_sig, m_w, m_rho, g_sigma, g_omega, g_rho, kappa, lambda_0, zeta, Lambda_w = theta

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

    sigma = x[0]
    omega = x[1]
    rho_03 = x[2]
    mu_n = x[3]
    mu_e = x[4]

    rho_B_list = []
    rho_SB_list = []
    q_list = []

    m_eff = m_n - (g_sigma * sigma)

    # -------- 重子部分 (质子 i=0, 中子 i=1) --------

    # 化学势: $\mu_b = b\mu_n - q\mu_e$ (b=1 for baryons)

    # 费米能量: $E_{fb} = \mu_b - g_\omega\omega - g_\rho\rho_{03}I_{3b}$

    # 费米动量: $k_{Fb} = \sqrt{E_{fb}^2 - m^{*2}}$

    # 重子数密度: $\rho_B = \frac{(2J_B+1)b_B k_{Fb}^3}{6\pi^2}$

    # 标量密度: $\rho_{SB} = \frac{m^*}{2\pi^2}[E_{Fb}k_{Fb} - m^{*2}\ln\frac{E_{Fb}+k_{Fb}}{m^*}]$

    # 电荷密度: $Q_B = \frac{(2J_B+1)q_B k_{Fb}^3}{6\pi^2}$

    for i in range(len(Matrix_b)):
        # 化学势计算: $\mu_b = \mu_n - q\mu_e$ (重子数 b=1)
        mu_b = Matrix_b[i, 0] * mu_n - Matrix_b[i, 1] * mu_e

        # 费米能量 (含介子场耦合)
        E_fb = mu_b - g_omega * omega - g_rho * rho_03 * Matrix_b[i, 2]

        # 费米动量平方: $k_{Fb}^2 = E_{fb}^2 - m^{*2}$
        k_fb_sq = E_fb**2 - m_eff**2
        if k_fb_sq < 0:
            k_fb_sq = np.clip(k_fb_sq, a_min=0.0, a_max=None)
            E_fb = m_eff

        k_fb = math.sqrt(k_fb_sq)

        # 重子数密度: $\rho_B = \frac{2 k_{Fb}^3}{6\pi^2} = \frac{k_{Fb}^3}{3\pi^2}$

        rho_B = ((2 * J_B) + 1) * b_B * k_fb**3 / (6.0 * math.pi**2)

        # 标量密度: $\rho_{SB} = \frac{m^*}{2\pi^2}[E_{Fb}k_{Fb} - m^{*2}\ln\frac{E_{Fb}+k_{Fb}}{m^*}]$

        rho_SB = (m_eff / (2.0 * math.pi**2)) * (
            E_fb * k_fb - (m_eff ** (2)) * np.log((E_fb + k_fb) / m_eff)
        )

        rho_B_list.append(rho_B)
        rho_SB_list.append(rho_SB)

        # 电荷密度: $Q_B = \frac{2 q_B k_{Fb}^3}{6\pi^2}$
        
        Q_B = ((2.0 * J_B) + 1.0) * Matrix_b[i, 1] * k_fb**3 / (6.0 * math.pi**2)
        q_list.append(Q_B)

    # -------- 轻子部分 (电子 j=0, μ子 j=1) --------

    # 化学势: $\mu_l = \mu_n - \mu_e$

    # 费米动量: $k_{Fl} = \sqrt{\mu_l^2 - m_l^2}$

    # 电荷密度: $Q_L = \frac{2 q_L k_{Fl}^3}{6\pi^2}$ (q_L = -1)

    for j in range(len(Matrix_l)):
        mu_l = Matrix_l[j, 0] * mu_n - Matrix_l[j, 1] * mu_e
        E_fl = mu_l

        k_fl_sq = E_fl**2 - m_l[j] ** 2
        k_fl_sq = np.clip(k_fl_sq, a_min=0.0, a_max=None)
        k_fl = math.sqrt(k_fl_sq)

        # 轻子电荷密度: $Q_L = -\frac{k_{Fl}^3}{3\pi^2}$
        
        Q_L = ((2.0 * J_B) + 1.0) * Matrix_l[j, 1] * (k_fl**3) / (6.0 * (math.pi**2))
        q_list.append(Q_L)

    # ============================================================

    # RMF 平均场自洽方程残差向量 $f_i^2$

    # 五个方程分别对应: σ场、ω场、ρ场方程, 重子数守恒, 电荷中性

    # ============================================================
    f = [
        # ---------- (1) σ 标量介子场方程 ----------

        # 方程来源:

        # $$\frac{m_\sigma^2 \sigma}{g_\sigma} = \rho_S - \frac{\kappa}{2}(g_\sigma\sigma)^2 - \frac{\lambda_0}{6}(g_\sigma\sigma)^3$$

        # 其中 $\rho_S = \sum_B \bar{u}_B\gamma^0 u_B = \sum_B \rho_{SB}$ 为标量密度

        (
            sigma * (m_sig**2) / g_sigma
            - sum(np.array(rho_SB_list))
            + (kappa * (g_sigma * sigma) ** 2) / 2.0
            + (lambda_0 * (g_sigma * sigma) ** 3) / 6.0
        )
        ** 2,

        # ---------- (2) ω 矢量介子场方程 ----------

        # 方程来源:

        # $$\frac{m_\omega^2 \omega}{g_\omega} = \rho_B - \frac{\zeta}{6}(g_\omega\omega)^3 - 2\Lambda_\omega g_\omega\omega(g_\rho\rho_{03})^2$$

        # 其中 $\rho_B = \sum_B b_B k_{Fb}^3/(6\pi^2)$ 为重子数密度


        # 末项 $-\Lambda_\omega g_\omega g_\rho^2 \omega \rho_{03}^2$ 为 ω-ρ 混合项
        (
            omega * (m_w**2) / g_omega
            - sum(np.array(rho_B_list) )
            + (zeta * (g_omega * omega) ** 3) / 6.0
            + 2.0 * Lambda_w * g_omega * omega * (rho_03 * g_rho) ** 2
        )
        ** 2,

        # ---------- (3) ρ 同位旋矢量介子场方程 ----------

        # 方程来源:

        # $$\frac{m_\rho^2 \rho_{03}}{g_\rho} = \sum_B \rho_B I_{3B} - 2\Lambda_\omega g_\rho\rho_{03}(g_\omega\omega)^2$$

        # 其中 $I_{3B}$ 为重子同位旋第三分量 (质子 $+1/2$, 中子 $-1/2$)

        (
            rho_03 * (m_rho**2) / g_rho
            - sum(np.array(rho_B_list) * Matrix_b[:, 2])
            + 2.0 * Lambda_w * g_rho * rho_03 * (omega * g_omega) ** 2
        )
        ** 2,

        # ---------- (4) 重子数守恒方程 ----------

        # 方程来源:

        # $$\rho = \sum_B \rho_B = \rho_p + \rho_n$$

        # 要求总重子数密度与输入密度一致
        (rho - sum(rho_B_list)) ** 2,

        # ---------- (5) 电荷中性条件 ----------

        # 方程来源:

        # $$\sum_B Q_B + \sum_L Q_L = 0$$

        # 即 $\rho_{Bp}q_p + \rho_{Bn}q_n + \rho_{Le}q_e + \rho_{L\mu}q_\mu = 0$
        (sum(q_list)) ** 2,
    ]

    return f


def Energy_density_Pressure(x, rho, theta):
    r"""
    计算 RMF 模型的状态方程 (能量密度和压强)。

    ---
    **重子能量密度:**

    单核子能量积分 (费米气体模型):

    $\varepsilon_b = \frac{1}{8\pi^2}\left[k_F(E_F^3 + k_F^3) - m^{*4}\ln\frac{k_F+E_F}{m^*}\right]$

    其中 $k_F$ 为费米动量, $E_F = \sqrt{k_F^2 + m^{*2}}$ 为费米能量。

    ---
    **轻子能量密度:**

    $\varepsilon_l = \frac{1}{8\pi^2}\left[k_l\mu_l^3 + k_l^3\mu_l - m_l^4\ln\frac{k_l+\mu_l}{m_l}\right]$

    ---
    **介子场能量项:**

    $\varepsilon_\sigma = \frac{1}{2}m_\sigma^2\sigma^2 + \frac{\kappa}{6}(g_\sigma\sigma)^3 + \frac{\lambda_0}{24}(g_\sigma\sigma)^4$

    $\varepsilon_\omega = \frac{1}{2}m_\omega^2\omega^2 + \frac{\zeta}{8}(g_\omega\omega)^4$

    $\varepsilon_\rho = \frac{1}{2}m_\rho^2\rho_{03}^2 + 3\Lambda_\omega(g_\rho g_\omega \omega\rho_{03})^2$

    ---
    **总能量密度与压强:**

    $\varepsilon = \varepsilon_b + \varepsilon_l + \varepsilon_\sigma + \varepsilon_\omega + \varepsilon_\rho$

    $P = \sum_f \mu_f \rho_f - \varepsilon$

    其中 $\sum_f \mu_f \rho_f$ 为化学势与粒子数密度之积的总和 (包括重子和轻子)。

    Args:
        x (array): $[\sigma, \omega, \rho_{03}, \mu_n, \mu_e]$
        rho (float): 重子数密度 $\rho$, 单位: fm$^{-3}$
        theta (array): 长度 10 的数值型 RMF 参数向量

    Returns:
        tuple 或 list: (能量密度, 压强) 或完整 EOS 组成
    """
    sigma, omega, rho_03, mu_n, mu_e = x

    m_sig, m_w, m_rho, g_sigma, g_omega, g_rho, kappa, lambda_0, zeta, Lambda_w = theta

    energy_b = 0
    energy_l = 0
    multi = 0

    Composition = ["mu_p", "mu_n", "mu_e", "mu_mu", "proton_fraction"]

    m_eff = m_n - (g_sigma * sigma)

    # -------- 重子能量计算 --------

    # 化学势关系: $\mu_b = \mu_n - q\mu_e$ (重子数 b=1)

    # 费米能量: $E_{fb} = \mu_b - g_\omega\omega - g_\rho\rho_{03}I_{3b}$

    # 动能项: $\varepsilon_b = \frac{1}{8\pi^2}[k_F(E_F^3 + k_F^3) - m^{*4}\ln\frac{k_F+E_F}{m^*}]$

    # 化学势×密度项: $\mu_b \rho_B$ 用于压强计算
    for i in range(len(Matrix_b)):
        # 化学势: $\mu_p = \mu_n - \mu_e$, $\mu_n = \mu_n$
        mu_b = Matrix_b[i, 0] * mu_n - Matrix_b[i, 1] * mu_e

        Composition[i] = mu_b

        # 费米能量: $E_{fb} = \mu_b - g_\omega\omega - g_\rho\rho_{03}I_{3b}$
        E_fb = mu_b - g_omega * omega - g_rho * rho_03 * Matrix_b[i, 2]

        # 费米动量: $k_{Fb} = \sqrt{E_{fb}^2 - m^{*2}}$
        k_fb_sq = E_fb**2 - m_eff**2
        if k_fb_sq < 0:
            k_fb_sq = 0.0
            E_fb = m_eff

        k_fb = math.sqrt(k_fb_sq)

        # 重子数密度: $\rho_B = \frac{2k_{Fb}^3}{6\pi^2} = \frac{k_{Fb}^3}{3\pi^2}$
        rho_B = ((2.0 * J_B) + 1.0) * b_B * (k_fb**3) / (6.0 * math.pi**2)

        if i == 0:
            # 质子分数: $x_p = \rho_p / \rho$
            Composition[4] = rho_B / rho

        # $\mu_b \rho_B$ 项, 贡献于压强: $P = \sum \mu_f \rho_f - \varepsilon$
        multi = multi + mu_b * rho_B

        # 重子能量密度 (费米气体积分):

        # $\varepsilon_b = \frac{1}{8\pi^2}[k_F(E_F^3 + k_F^3) - m^{*4}\ln\frac{k_F+E_F}{m^*}]$

        energy_baryon = (1 / (8.0 * (math.pi**2))) * (
            k_fb * (E_fb**3)
            + (k_fb**3) * E_fb
            - np.log((k_fb + E_fb) / m_eff) * m_eff**4
        )

        energy_b = energy_b + energy_baryon

    # -------- 轻子能量计算 --------

    # 轻子化学势: $\mu_l = \mu_n - \mu_e$ (电荷 q=-1)

    # 轻子数密度: $\rho_l = \frac{k_{Fl}^3}{3\pi^2}$

    # 轻子能量密度: $\varepsilon_l = \frac{1}{8\pi^2}[k_l\mu_l^3 + k_l^3\mu_l - m_l^4\ln\frac{k_l+\mu_l}{m_l}]$

    for j in range(len(Matrix_l)):
        mu_l = Matrix_l[j, 0] * mu_n - Matrix_l[j, 1] * mu_e

        Composition[2 + j] = mu_l

        k_fl_sq = mu_l**2 - m_l[j] ** 2
        if k_fl_sq < 0.0:
            k_fl_sq = 0.0
        k_fl = math.sqrt(k_fl_sq)

        # 轻子数密度: $\rho_l = \frac{k_{Fl}^3}{3\pi^2}$

        rho_l = k_fl**3 / (3.0 * math.pi**2)

        # $\mu_l \rho_l$ 项, 贡献于压强
        multi = multi + mu_l * rho_l

        # 轻子能量密度:

        # $\varepsilon_l = \frac{1}{8\pi^2}[k_l\mu_l^3 + k_l^3\mu_l - m_l^4\ln\frac{k_l+\mu_l}{m_l}]$

        energy_lepton = (1 / (8.0 * (math.pi**2))) * (
            k_fl * (mu_l**3)
            + mu_l * (k_fl**3)
            - (m_l[j] ** 4) * np.log((k_fl + mu_l) / m_l[j])
        )

        energy_l = energy_l + energy_lepton

    # -------- 介子场能量项 --------

    # σ 场能量:

    # $\varepsilon_\sigma = \frac{1}{2}m_\sigma^2\sigma^2 + \frac{\kappa}{6}(g_\sigma\sigma)^3 + \frac{\lambda_0}{24}(g_\sigma\sigma)^4$
    sigma_terms = (
        0.5 * ((sigma * m_sig) ** 2)
        + (kappa * ((g_sigma * sigma) ** 3)) / 6.0
        + (lambda_0 * ((g_sigma * sigma) ** 4)) / 24.0
    )

    # ω 场能量:

    # $\varepsilon_\omega = \frac{1}{2}m_\omega^2\omega^2 + \frac{\zeta}{8}(g_\omega\omega)^4$

    omega_terms = 0.5 * ((omega * m_w) ** 2) + (zeta * ((g_omega * omega) ** 4)) / 8.0

    # ρ 场能量 (含 ω-ρ 混合项):

    # $\varepsilon_\rho = \frac{1}{2}m_\rho^2\rho_{03}^2 + 3\Lambda_\omega(g_\rho g_\omega \omega\rho_{03})^2$
    
    rho_terms = 0.5 * ((rho_03 * m_rho) ** 2) + 3.0 * Lambda_w * (
        (g_rho * rho_03 * g_omega * omega) ** 2
    )

    # -------- 总能量密度与压强 --------

    # 总能量密度: $\varepsilon = \varepsilon_b + \varepsilon_l + \varepsilon_\sigma + \varepsilon_\omega + \varepsilon_\rho$

    energy_density = energy_b + energy_l + sigma_terms + omega_terms + rho_terms

    # 压强: $P = \sum_f \mu_f \rho_f - \varepsilon$

    # 其中 $\sum_f \mu_f \rho_f = \sum_B \mu_B\rho_B + \sum_L \mu_L\rho_L$

    Pressure = multi - energy_density

    EoS = [rho, energy_density, Pressure] + Composition
    return EoS


def compute_EOS(eps_crust, pres_crust, theta):
    r"""
    由 RMF 参数向量生成中子星内核状态方程, 并与壳层 EOS 拼接。

    ---
    **计算流程:**

    (1) 在饱和密度 $\rho_0 = 0.1505$ fm$^{-3}$ 附近取 124 个密度点:

    $\rho_i = i \cdot \Delta t \cdot \rho_0, \quad i = 1, 2, \ldots, 124$

    其中 $\Delta t = 0.05$。

    (2) 对每个密度点, 通过求解自洽方程获得介子场与化学势。

    (3) 计算能量密度与压强:

    $\varepsilon = \varepsilon_\mathrm{baryon} + \varepsilon_\mathrm{lepton} + \varepsilon_\mathrm{meson}$

    $P = \sum_f \mu_f \rho_f - \varepsilon$

    (4) 单位换算: MeV/fm³ → g/cm³, dyn/cm²:

    $\varepsilon [\mathrm{g/cm^3}] = \varepsilon [\mathrm{MeV/fm^3}] \times \frac{\hbar c}{1.7827\times 10^{12}}$

    $P [\mathrm{dyn/cm^2}] = P [\mathrm{MeV/fm^3}] \times \frac{\hbar c}{1.6022\times 10^{33}}$

    (5) 与壳层 EOS 拼接: 只保留能量密度高于壳层最大值的内核部分。

    Args:
        eps_crust (array): 壳层能量密度, 单位: g·cm$^{-3}$
        pres_crust (array): 壳层压强, 单位: dyn·cm$^{-2}$
        theta (array): 长度 10 的数值型 RMF 参数向量
            $[m_\sigma, m_\omega, m_\rho, g_\sigma, g_\omega, g_\rho, \kappa, \lambda_0, \zeta, \Lambda_\omega]$

    Returns:
        ndarray: (能量密度, 压强) 数组, 单位 g·cm$^{-3}$ / dyn·cm$^{-2}$
    """
    theta = _validate_rmf_theta(theta)
    dt = 0.05
    rho_0 = 0.1505

    x_init = np.array(initial_values(0.1 * rho_0, theta))

    EoS = [[] for i in range(124)]


    for i in range(1, 125):
        rho = i * dt * rho_0

        arg = np.append(theta, rho)
        sol = optimize.root(functie, x_init, method="lm", args=arg)

        Re = Energy_density_Pressure(sol.x, rho, theta)

        Re[1] = Re[1] * oneoverfm_MeV / gcm3_to_MeVfm3
        Re[2] = Re[2] * oneoverfm_MeV / dyncm2_to_MeVfm3

        EoS[i - 1] = Re

        x_init = sol.x

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
