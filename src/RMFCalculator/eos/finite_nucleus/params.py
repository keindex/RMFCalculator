r"""
params.py - 有限核 RMF 模型参数集

基于学位论文第二章 (2.6节) 的理论框架，定义有限核计算的参数集。
"""

import math
from dataclasses import dataclass
from typing import Dict

from RMFCalculator.eos.unit import fm_MeV


@dataclass(frozen=True)
class NucleusParams:
    """有限核 RMF 参数集.

    参数对应拉格朗日量中的系数:
    - m_sigma: $m_\sigma$ (sigma meson mass)
    - m_omega: $m_\omega$ (omega meson mass)
    - m_rho: $m_\rho$ (rho meson mass)
    - m_delta: $m_\delta$ (delta meson mass)
    - g_sigma: $g_\sigma$ (sigma-nucleon coupling)
    - g_omega: $g_\omega$ (omega-nucleon coupling)
    - g_rho: $g_\rho$ (rho-nucleon coupling)
    - g_delta: $g_\delta$ (delta-nucleon coupling)
    - kappa: $\kappa$ (sigma cubic self-coupling)
    - lambda0: $\lambda_0$ (sigma quartic self-coupling)
    - zeta: $\zeta$ (omega quartic self-coupling)
    - Lambda_w: $\Lambda_\omega$ (omega-rho cross coupling)
    - Lambda_sd: $\Lambda_{\sigma\delta}$ (sigma-delta cross coupling)
    """
    m_sigma: float
    m_omega: float
    m_rho: float
    m_delta: float
    g_sigma: float
    g_omega: float
    g_rho: float
    g_delta: float
    kappa: float
    lambda0: float
    zeta: float
    Lambda_w: float
    Lambda_sd: float


FSU_GOLD_FN: NucleusParams = NucleusParams(
    m_sigma=491.5, m_omega=782.5, m_rho=763.0, m_delta=980.0,
    g_sigma=math.sqrt(112.1996), g_omega=math.sqrt(204.5469),
    g_rho=math.sqrt(138.4701), g_delta=0.0,
    kappa=1.4203, lambda0=0.023762, zeta=0.06,
    Lambda_w=0.03, Lambda_sd=0.0,
)

FSU_DELTA67_FN: NucleusParams = NucleusParams(
    m_sigma=491.5, m_omega=782.5, m_rho=763.0, m_delta=980.0,
    g_sigma=10.2143, g_omega=13.4353, g_rho=11.767, g_delta=4.5348,
    kappa=2 * 1.59524e-3 * 938.92, lambda0=6 * 0.540269e-3,
    zeta=6 * 0.528340e-2, Lambda_w=0.03, Lambda_sd=-14.06,
)

PARAMSETS: Dict[str, NucleusParams] = {
    "FSUGold": FSU_GOLD_FN,
    "FSU_Delta67": FSU_DELTA67_FN,
}