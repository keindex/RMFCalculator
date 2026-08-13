r"""
parameters.py - RMF 模型参数集与参数向量构造

统一管理 RMF 模型的参数:

    $$\theta = [m_\sigma,\, m_\omega,\, m_\rho,\, g_\sigma,\, g_\omega,\, g_\rho,\, \kappa,\, \lambda_0,\, \zeta,\, \Lambda_\omega]$$

其中介子质量 $m_i$ 与 $\kappa$ 以 fm$^{-1}$ 为单位 (由 MeV 除以 $\hbar c$ 得到),
耦合常数 $g_i,\, \lambda_0,\, \zeta,\, \Lambda_\omega$ 为无量纲。

    含 $\delta$ 介子的模型额外使用参数向量

    $$\theta_\delta = [m_\delta,\, g_\delta,\, \Lambda_{\sigma\delta}]$$

    共 13 元 ($\theta$ + $\theta_\delta$)。

"""

import numpy as np

from RMFCalculator.eos.unit import fm_MeV


def build_theta_from_params(m_sigma, m_omega, m_rho, g_sigma, g_omega, g_rho,
                            kappa, lambda_0, zeta, Lambda_w):
    r"""
    由 MeV 单位参数构造 RMF 参数向量 $\theta$, 单位: fm$^{-1}$。

    Args:
        各介子质量 (MeV) 与无量纲耦合常数。

    Returns:
        ndarray: 长度 10 的 $\theta$:
            $[m_\sigma,\, m_\omega,\, m_\rho,\, g_\sigma,\, g_\omega,\, g_\rho,\, \kappa,\, \lambda_0,\, \zeta,\, \Lambda_\omega]$
    """
    MeV_to_fm = 1.0 / fm_MeV
    return np.array([
        m_sigma * MeV_to_fm,   # $m_\sigma$ (fm$^{-1}$)
        m_omega * MeV_to_fm,   # $m_\omega$ (fm$^{-1}$)
        m_rho * MeV_to_fm,     # $m_\rho$   (fm$^{-1}$)
        g_sigma,               # $g_\sigma$
        g_omega,               # $g_\omega$
        g_rho,                 # $g_\rho$
        kappa * MeV_to_fm,     # $\kappa$   (MeV → fm$^{-1}$)
        lambda_0,              # $\lambda_0$
        zeta,                  # $\zeta$
        Lambda_w,              # $\Lambda_\omega$
    ])


# ==================== 参数集 ====================

def build_theta_delta_from_params(m_delta, g_delta, Lambda_sigma_delta):
    r"""
    由 MeV 单位参数构造 $\delta$ 介子参数向量 $\theta_\delta$ (fm$^{-1}$/无量纲)。

    Args:
        m_delta (float): $\delta$ 介子质量 $m_\delta$ (MeV)
        g_delta (float): 核子-$\delta$ 耦合常数 $g_\delta$
        Lambda_sigma_delta (float): $\sigma$-$\delta$ 交叉耦合 $\Lambda_{\sigma\delta}$

    Returns:
        ndarray: 长度 3 的 $\theta_\delta = [m_\delta,\, g_\delta,\, \Lambda_{\sigma\delta}]$,
        其中 $m_\delta$ 以 fm$^{-1}$ 为单位。
    """
    MeV_to_fm = 1.0 / fm_MeV
    return np.array([
        m_delta * MeV_to_fm,   # $m_\delta$ (fm$^{-1}$)
        g_delta,               # $g_\delta$
        Lambda_sigma_delta,    # $\Lambda_{\sigma\delta}$
    ])


def build_theta_13_from_params(m_sigma, m_omega, m_rho, m_delta,
                               g_sigma, g_omega, g_rho, g_delta,
                               kappa, lambda_0, zeta, Lambda_w,
                               Lambda_sigma_delta):
    r"""
    由 MeV 单位参数构造含 $\delta$ 介子的 13 元参数向量 (fm$^{-1}$/无量纲)。

    $$\theta_{13} = [m_\sigma,\, m_\omega,\, m_\rho,\, m_\delta,\, g_\sigma,\, g_\omega,\, g_\rho,\, g_\delta,\, \kappa,\, \lambda_0,\, \zeta,\, \Lambda_\omega,\, \Lambda_{\sigma\delta}]$$

    新引入的 $\delta$ 相关参数位于下列位置:

    - ``index 3``:  $m_\delta$ (fm$^{-1}$)
    - ``index 7``:  $g_\delta$
    - ``index 12``: $\Lambda_{\sigma\delta}$

    Args:
        各介子质量 (MeV) 与无量纲耦合常数。

    Returns:
        ndarray: 长度 13 的 $\theta_{13}$。
    """
    MeV_to_fm = 1.0 / fm_MeV
    return np.array([
        m_sigma * MeV_to_fm,   # $m_\sigma$  (fm$^{-1}$)
        m_omega * MeV_to_fm,   # $m_\omega$  (fm$^{-1}$)
        m_rho * MeV_to_fm,     # $m_\rho$    (fm$^{-1}$)
        m_delta * MeV_to_fm,   # $m_\delta$  (fm$^{-1}$)
        g_sigma,               # $g_\sigma$
        g_omega,               # $g_\omega$
        g_rho,                 # $g_\rho$
        g_delta,               # $g_\delta$
        kappa * MeV_to_fm,     # $\kappa$    (MeV → fm$^{-1}$)
        lambda_0,              # $\lambda_0$
        zeta,                  # $\zeta$
        Lambda_w,              # $\Lambda_\omega$
        Lambda_sigma_delta,    # $\Lambda_{\sigma\delta}$
    ])


# FSUGold 参数 (学位论文表 4-2), 单位: MeV / 无量纲。
# 注意: $g_\sigma^2 = 112.1996$, 对应 $g_\sigma = \sqrt{112.1996} \approx 10.5924$。
# 早期代码误用 $g_\sigma^2 = 107.5751$ (NL3 的 $g_\sigma$ 平方), 会导致
# $K_0 \approx 169$ MeV、$n_0 \approx 0.134$ fm$^{-3}$ 等偏离文献的值。
FSU_PARAMS_MEV = {
    'm_sigma': 491.5,          # $m_\sigma$ (MeV)
    'm_omega': 782.5,          # $m_\omega$ (MeV)
    'm_rho': 763.0,            # $m_\rho$   (MeV)
    'g_sigma': np.sqrt(112.1996),   # $g_\sigma = \sqrt{112.1996}$
    'g_omega': np.sqrt(204.5469),   # $g_\omega = \sqrt{204.5469}$
    'g_rho': np.sqrt(138.4701),     # $g_\rho   = \sqrt{138.4701}$
    'kappa': 1.4203,           # $\kappa$ (MeV)
    'lambda_0': 0.023762,      # $\lambda_0$
    'zeta': 0.06,              # $\zeta$
    'Lambda_w': 0.03,          # $\Lambda_\omega$
}


# FSU-δ6.7 / FSU-δ6.2 参数 (学位论文表 4-2), 基于 FSU-J0 + δ 介子 + σ-δ 耦合。
# 注意: FSU-J0 / FSU-δ 系列的真参数与 FSUGold 不同:
#   g_σ = 10.2143, g_ω = 13.4353, b_σ = 1.59524e-3, c_σ = 0.540269e-3, c_ω = 0.528340e-2。
# 代码约定 (与 FSU_PARAMS_MEV 一致):
#   g_ρ^code = 2·g_ρ^paper (论文小 g_ρ 约定), Λ_ω^code = Λ_V^paper/8,
#   κ = 2·b_σ·M, λ_0 = 6·c_σ, ζ = 6·c_ω (M = m_n = 938.92 MeV)。
# 验证 (FSUGold): 2×0.756283e-3×938.92 ≈ 1.4203 ✓, 6×3.96033e-3 = 0.023762 ✓,
#   6×1.00e-2 = 0.06 ✓, 0.24/8 = 0.03 ✓, 2×5.88367 ≈ 11.767 ✓。
FSU_DELTA_PARAMS_MEV = {
    'm_sigma': 491.5,          # $m_\sigma$ (MeV)
    'm_omega': 782.5,          # $m_\omega$ (MeV)
    'm_rho': 763.0,            # $m_\rho$   (MeV)
    'g_sigma': 10.2143,        # $g_\sigma$ (表 4-2)
    'g_omega': 13.4353,        # $g_\omega$ (表 4-2)
    'kappa': 2 * 1.59524e-3 * 938.92,   # $\kappa = 2 b_\sigma M$ ≈ 2.9956 MeV
    'lambda_0': 6 * 0.540269e-3,        # $\lambda_0 = 6 c_\sigma$ ≈ 0.0032416
    'zeta': 6 * 0.528340e-2,            # $\zeta = 6 c_\omega$ ≈ 0.0317004
    # g_rho / Lambda_w 各模型不同, 覆盖于下方:
    'g_rho': None,
    'Lambda_w': None,
}


# FSU-δ6.7: g_δ = 6.7, Λ_σδ = 0.0385; g_ρ = 7.26866 (小) → 14.5373 (代码), Λ_V = 0.0214 → Λ_ω = 0.002675
FSU_DELTA67 = {
    **FSU_DELTA_PARAMS_MEV,
    'g_rho': 2 * 7.26866,               # $g_\rho$ = 14.5373
    'Lambda_w': 0.0214 / 8,             # $\Lambda_\omega$ = 0.002675
    'm_delta': 980.0,                   # $m_\delta$ (MeV)
    'g_delta': 6.7,                     # $g_\delta$
    'Lambda_sigma_delta': 0.0385,       # $\Lambda_{\sigma\delta} = C_{\delta\sigma}$
}


# FSU-δ6.2: g_δ = 6.2, Λ_σδ = 0.054; g_ρ = 6.98518 (小) → 13.97036 (代码), Λ_V = 0.0200112 → Λ_ω = 0.0025014
FSU_DELTA62 = {
    **FSU_DELTA_PARAMS_MEV,
    'g_rho': 2 * 6.98518,               # $g_\rho$ = 13.97036
    'Lambda_w': 0.0200112 / 8,          # $\Lambda_\omega$ = 0.0025014
    'm_delta': 980.0,                   # $m_\delta$ (MeV)
    'g_delta': 6.2,                     # $g_\delta$
    'Lambda_sigma_delta': 0.054,        # $\Lambda_{\sigma\delta} = C_{\delta\sigma}$
}


def get_theta(name="FSU"):
    r"""
    返回命名参数集的 RMF 参数向量 (fm$^{-1}$ 单位)。

    Args:
        name (str): 参数集名称:
            - "FSU"       → 长度 10 的 $\theta$ (无 $\delta$ 介子)
            - "FSU_DELTA67" / "FSU_DELTA62" → 长度 13 的 $\theta_{13}$ (含 $\delta$ 介子)

    Returns:
        ndarray: 长度 10 或 13 的参数向量。
    """
    key = name.upper()
    if key == "FSU":
        return build_theta_from_params(**FSU_PARAMS_MEV)
    if key in ("FSU_DELTA67", "FSU_DELTA6.7", "FSU_δ67", "FSUΔ67"):
        return build_theta_13_from_params(**FSU_DELTA67)
    if key in ("FSU_DELTA62", "FSU_DELTA6.2", "FSU_δ62", "FSUΔ62"):
        return build_theta_13_from_params(**FSU_DELTA62)
    raise ValueError(f"未知参数集: {name!r}, 可用: ['FSU', 'FSU_DELTA67', 'FSU_DELTA62']")