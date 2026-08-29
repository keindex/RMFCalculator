r"""
binding_energy.py - 结合能计算

计算有限核的总结合能，包括液滴模型部分和 RMF 场能部分。
"""

import math

import numpy as np

from RMFCalculator.eos.unit import fm_MeV
from .params import NucleusParams


def compute_binding_energy(
    Z: int,
    N: int,
    sigma: np.ndarray,
    omega: np.ndarray,
    rho03: np.ndarray,
    delta_f: np.ndarray,
    A0: np.ndarray,
    params: NucleusParams,
    r: np.ndarray,
    dr: float,
) -> dict:
    r"""计算总结合能.

    Args:
        Z: 质子数。
        N: 中子数。
        sigma: $\sigma$ 场分布。
        omega: $\omega$ 场分布。
        rho03: $\rho_0^3$ 场分布。
        delta_f: $\delta$ 场分布。
        A0: 光子场 $A_0$ 分布。
        params: NucleusParams 参数集。
        r: 径向网格 (fm)。
        dr: 网格步长 (fm)。

    Returns:
        dict: 包含各项能量贡献和总结合能。
    """
    g_s, g_w, g_r, g_d = params.g_sigma, params.g_omega, params.g_rho, params.g_delta
    # 介子质量用 fm⁻¹ (natural units)
    m_sig, m_w, m_rho, m_d = (
        params.m_sigma / fm_MeV,  # $m_\sigma^* = m_\sigma / (\hbar c)$
        params.m_omega / fm_MeV,  # $m_\omega^* = m_\omega / (\hbar c)$
        params.m_rho / fm_MeV,    # $m_\rho^* = m_\rho / (\hbar c)$
        params.m_delta / fm_MeV,  # $m_\delta^* = m_\delta / (\hbar c)$
    )
    # 自耦合系数
    kappa_c, lam0, zeta_c = params.kappa / fm_MeV, params.lambda0, params.zeta  # $\kappa^* = \kappa/(\hbar c)$
    Lw, Lsd = params.Lambda_w, params.Lambda_sd

    A = Z + N
    weight = 4.0 * math.pi * r ** 2 * dr  # fm³, $dV = 4\pi r^2 dr$

    # ========== 液滴模型部分 (MeV) ==========
    # $E_{LD} = a_v A - a_s A^{2/3} - a_c Z^2/A^{1/3} - a_{sym} (N-Z)^2/A + \delta_{pair} + E_{CM}$
    a_v, a_s, a_c, a_sym = 15.5, 16.8, 0.72, 23.0  # MeV
    E_ld = (
        a_v * A  # $a_v A$ 体积项
        - a_s * A ** (2.0 / 3.0)  # $-a_s A^{2/3}$ 表面项
        - a_c * Z ** 2 / A ** (1.0 / 3.0)  # $-a_c Z^2/A^{1/3}$ 库仑项
        - a_sym * (N - Z) ** 2 / A  # $-a_{sym} (N-Z)^2/A$ 对称能项
    )
    # Pairing: $\delta_{pair} = +12/\sqrt{A}$ (even-even), $-12/\sqrt{A}$ (odd-odd), 0 (否则)
    delta_pair = 12.0 / math.sqrt(A) if (Z % 2 == 0 and N % 2 == 0) else 0.0
    if Z % 2 != 0 and N % 2 != 0:
        delta_pair = -delta_pair
    E_ld += delta_pair
    # 质心修正: $E_{CM} = -0.75 \times 41 \times A^{-1/3}$ MeV
    E_CM = -0.75 * 41.0 * A ** (-1.0 / 3.0)

    # ========== RMF 场能 (自然单位 fm⁻⁴) ==========
    # 能量密度 $\mathcal{E} = \int d^3r \, \varepsilon(r)$
    # 其中 $\varepsilon(r)$ 为各场的能量密度

    # σ 场能: $\varepsilon_\sigma = \frac{1}{2} (\nabla\sigma)^2 + \frac{1}{2} m_\sigma^{*2} \sigma^2 + \frac{\kappa}{6} (g_\sigma \sigma)^3 + \frac{\lambda_0}{24} (g_\sigma \sigma)^4$
    d_sigma_dr = np.gradient(sigma, dr)  # $\nabla\sigma$ (radial)
    eps_sigma = (
        0.5 * d_sigma_dr ** 2  # $\frac{1}{2} (\nabla\sigma)^2$
        + 0.5 * m_sig ** 2 * sigma ** 2  # $\frac{1}{2} m_\sigma^{*2} \sigma^2$
        + kappa_c / 6.0 * (g_s * sigma) ** 3  # $\frac{\kappa}{6} (g_\sigma \sigma)^3$
        + lam0 / 24.0 * (g_s * sigma) ** 4  # $\frac{\lambda_0}{24} (g_\sigma \sigma)^4$
    )
    E_sigma = np.sum(eps_sigma * weight) * fm_MeV  # $\int \varepsilon_\sigma d^3r \times (\hbar c)$ → MeV

    # ω 场能: $\varepsilon_\omega = -\frac{1}{2} (\nabla\omega)^2 - \frac{1}{2} m_\omega^{*2} \omega^2 - \frac{\zeta}{8} (g_\omega \omega)^4 - \Lambda_\omega (g_\rho \rho_{03})^2 (g_\omega \omega)^2$
    d_omega_dr = np.gradient(omega, dr)  # $\nabla\omega$
    eps_omega = (
        -d_omega_dr ** 2  # $-\frac{1}{2} (\nabla\omega)^2$ (注意符号)
        - 0.5 * m_w ** 2 * omega ** 2  # $-\frac{1}{2} m_\omega^{*2} \omega^2$
        - zeta_c / 8.0 * (g_w * omega) ** 4  # $-\frac{\zeta}{8} (g_\omega \omega)^4$
        - Lw * (g_r * rho03) ** 2 * (g_w * omega) ** 2  # $-\Lambda_\omega (g_\rho \rho_{03})^2 (g_\omega \omega)^2$
    )
    E_omega = np.sum(eps_omega * weight) * fm_MeV

    # ρ 场能: $\varepsilon_\rho = -\frac{1}{2} (\nabla\rho_{03})^2 - \frac{1}{2} m_\rho^{*2} \rho_{03}^2$
    d_rho_dr = np.gradient(rho03, dr)  # $\nabla\rho_{03}$
    eps_rho = -d_rho_dr ** 2 - 0.5 * m_rho ** 2 * rho03 ** 2  # $-\frac{1}{2} (\nabla\rho_{03})^2 - \frac{1}{2} m_\rho^{*2} \rho_{03}^2$
    E_rho = np.sum(eps_rho * weight) * fm_MeV

    # δ 场能: $\varepsilon_\delta = -\frac{1}{2} (\nabla\delta)^2 - \frac{1}{2} m_\delta^{*2} \delta^2$ (仅当 $g_\delta > 0$)
    if g_d > 0:
        d_delta_dr = np.gradient(delta_f, dr)  # $\nabla\delta$
        eps_delta = -d_delta_dr ** 2 - 0.5 * m_d ** 2 * delta_f ** 2  # $-\frac{1}{2} (\nabla\delta)^2 - \frac{1}{2} m_\delta^{*2} \delta^2$
        E_delta = np.sum(eps_delta * weight) * fm_MeV
    else:
        E_delta = 0.0

    # Coulomb 能: $\varepsilon_C = -\frac{1}{2} (\nabla A_0)^2$
    d_A0_dr = np.gradient(A0, dr)  # $\nabla A_0$
    eps_c = -0.5 * d_A0_dr ** 2  # $-\frac{1}{2} (\nabla A_0)^2$
    E_c_field = np.sum(eps_c * weight) * fm_MeV

    # 总结合能: $E_{tot} = E_{LD} + E_\sigma + E_\omega + E_\rho + E_\delta + E_C + E_{CM}$
    E_total = E_ld + E_sigma + E_omega + E_rho + E_delta + E_c_field + E_CM

    return {
        "E_ld": E_ld,
        "E_sigma": E_sigma,
        "E_omega": E_omega,
        "E_rho": E_rho,
        "E_delta": E_delta,
        "E_c_field": E_c_field,
        "E_CM": E_CM,
        "E_total": E_total,
        "E_per_nucleon": E_total / A if A > 0 else 0.0,  # $E_{tot}/A$
    }