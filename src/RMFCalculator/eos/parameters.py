r"""
parameters.py - RMF 模型参数集与参数向量构造

统一管理 RMF 模型的参数:

    $$\theta = [m_\sigma,\, m_\omega,\, m_\rho,\, g_\sigma,\, g_\omega,\, g_\rho,\, \kappa,\, \lambda_0,\, \zeta,\, \Lambda_\omega]$$

其中介子质量 $m_i$ 与 $\kappa$ 以 fm$^{-1}$ 为单位 (由 MeV 除以 $\hbar c$ 得到),
耦合常数 $g_i,\, \lambda_0,\, \zeta,\, \Lambda_\omega$ 为无量纲。

所有脚本 (quick_start / compute_saturation / compute_quantities) 统一从这里取参数,
避免各自复制参数导致不一致。
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


def get_theta(name="FSU"):
    r"""
    返回命名参数集的 RMF 参数向量 $\theta$ (fm$^{-1}$ 单位)。

    Args:
        name (str): 参数集名称, 目前仅支持 "FSU"。

    Returns:
        ndarray: 长度 10 的 $\theta$。
    """
    if name.upper() == "FSU":
        return build_theta_from_params(**FSU_PARAMS_MEV)
    raise ValueError(f"未知参数集: {name!r}, 可用: ['FSU']")