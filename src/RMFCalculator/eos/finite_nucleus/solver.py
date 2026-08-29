r"""
solver.py - 有限核 RMF 自洽求解器

主求解器：对给定原子核 (Z, N) 进行 RMF 自洽迭代计算。

理论基础:
- 介子场方程: $(-\nabla^2 + m_i^2) \phi_i = g_i \rho_i$ (sigma, omega, rho, delta)
- 库仑场: $\nabla^2 A_0 = -e^2 \rho_p$
- 标量密度: $n_s = \langle \bar\psi\psi \rangle$
- 矢量密度: $n_v = \langle \psi^\dagger\psi \rangle$
- 同位旋矢量密度: $n_3 = \langle \psi^\dagger\tau_3\psi \rangle$
- 同位旋赝标量密度: $n_{3s} = \langle \bar\psi i\gamma_5\tau_3\psi \rangle$
- 质子密度: $n_\gamma = \langle \psi_p^\dagger\psi_p \rangle$
"""

import math

import numpy as np

from RMFCalculator.eos.unit import fm_MeV
from .binding_energy import compute_binding_energy
from .density import initial_densities, nuclear_radius
from .green_function import solve_field_1d
from .params import FSU_GOLD_FN, NucleusParams

# 物理常数
E_SQ = 4.0 * math.pi / 137.036  # $e^2$ in fm$^{-1}$, $e^2 = 4\pi\alpha$


def compute_finite_nucleus(
    Z: int,
    N: int,
    params: NucleusParams = FSU_GOLD_FN,
    max_iter: int = 20,
    conv_tol: float = 1e-2,
) -> dict:
    r"""
    对给定原子核 (Z, N) 进行 RMF 自洽计算.

    Args:
        Z: 质子数。
        N: 中子数。
        params: NucleusParams 参数集。
        max_iter: 最大迭代次数。
        conv_tol: 收敛阈值 (MeV)。

    Returns:
        dict: 包含场分布、密度、结合能等。
    """
    g_s, g_w, g_r, g_d = params.g_sigma, params.g_omega, params.g_rho, params.g_delta
    m_sig = params.m_sigma * fm_MeV  # $m_\sigma^* = m_\sigma / (\hbar c)$ (fm$^{-1}$)
    m_w = params.m_omega * fm_MeV    # $m_\omega^* = m_\omega / (\hbar c)$
    m_rho = params.m_rho * fm_MeV    # $m_\rho^* = m_\rho / (\hbar c)$
    m_d = params.m_delta * fm_MeV    # $m_\delta^* = m_\delta / (\hbar c)$

    kappa_c = params.kappa * fm_MeV  # $\kappa^* = \kappa / (\hbar c)$
    lam0, zeta_c = params.lambda0, params.zeta
    Lw, Lsd = params.Lambda_w, params.Lambda_sd

    A = Z + N
    R = nuclear_radius(A)  # $R = 1.2 \times A^{1/3}$ fm

    # 径向网格
    r = np.linspace(0.05, R * 2.5, 200)  # 0.05 fm 到 2.5R
    dr = r[1] - r[0]  # 网格步长

    # 初始密度 (Woods-Saxon)
    dens0 = initial_densities(r, A, Z, N, R)
    n_v0 = dens0["n_v"]  # 矢量密度 $n_v(r)$
    n_s0 = dens0["n_s"]  # 标量密度 $n_s(r)$
    n_30 = dens0["n_3"]  # 同位旋矢量密度 $n_3(r)$
    n_3s0 = dens0["n_3s"]  # 同位旋赝标量密度 $n_{3s}(r)$
    n_gamma0 = dens0["n_gamma"]  # 质子密度 $n_\gamma(r)$

    # 初始场 (局部近似 $\phi \approx \rho / m^2$)
    # $\sigma_0 = g_\sigma n_v / m_\sigma^{*2}$
    sigma = np.full_like(r, g_s * n_v0[0] / (m_sig / fm_MeV) ** 2)  # $g_s n_v0 / m_sig^{*2}$
    # $\omega_0 = g_\omega n_v / m_\omega^{*2}$
    omega = np.full_like(r, g_w * n_v0[0] / (m_w / fm_MeV) ** 2)  # $g_w n_v0 / m_w^{*2}$
    # $\rho_0^3 = -g_\rho n_3 / (2 m_\rho^{*2})$ (因子 2 来自等式 $(-\nabla^2 + m^2)\rho = -g_\rho n_3$)
    rho03 = np.full_like(
        r,
        -g_r * n_30[0] / (2.0 * (m_rho / fm_MeV) ** 2) if n_30[0] != 0 else 0.0,  # $-g_r n_30 / (2 m_rho^{*2})$
    )
    # $\delta_0 = g_\delta n_{3s} / (2 m_\delta^{*2})$ (类似 rho 场)
    delta_f = np.full_like(
        r,
        g_d * n_3s0[0] / (2.0 * (m_d / fm_MeV) ** 2)
        if g_d > 0 and n_3s0[0] != 0
        else 0.0,  # $g_d n_{3s0} / (2 m_d^{*2})$
    )
    # 库仑场: $A_0(r) = Z e^2 / r$ (单点近似，实际应为 $Q(r)/r$)
    A0 = np.full_like(r, Z * E_SQ / r[-1] if Z > 0 else 0.0)  # $Z e^2 / R_{max}$

    prev_E = None

    for it in range(max_iter):
        # 计算源项 (右端项 $s_i = g_i \rho_i$ 及自耦合修正)
        # $\sigma$ 场源: $s_\sigma = -g_\sigma [ n_s - \frac{\kappa}{2} (g_\sigma \sigma)^2 - \frac{\lambda_0}{6} (g_\sigma \sigma)^3 ]$
        s_sigma = -g_s * (
            n_s0
            - kappa_c / (2.0 * fm_MeV ** 3) * (g_s * sigma) ** 2  # $\frac{\kappa}{2} (g_\sigma \sigma)^2$
            - lam0 / (6.0 * fm_MeV ** 4) * (g_s * sigma) ** 3  # $\frac{\lambda_0}{6} (g_\sigma \sigma)^3$
        )
        # $\omega$ 场源: $s_\omega = g_\omega [ n_v - \frac{\zeta}{6} (g_\omega \omega)^3 - \Lambda_\omega g_\omega \omega (g_\rho \rho_{03})^2 ]$
        s_omega = g_w * (
            n_v0
            - zeta_c / (6.0 * fm_MeV ** 4) * (g_w * omega) ** 3  # $\frac{\zeta}{6} (g_\omega \omega)^3$
            - Lw * (g_w * omega) * (g_r * rho03) ** 2 / fm_MeV ** 4  # $\Lambda_\omega g_\omega \omega (g_\rho \rho_{03})^2$
        )
        # $\rho$ 场源: $s_\rho = g_\rho [ n_3 - \Lambda_\omega g_\rho \rho_{03} (g_\omega \omega)^2 ]$
        s_rho = g_r * (
            n_30
            - Lw * (g_r * rho03) * (g_w * omega) ** 2 / fm_MeV ** 4  # $\Lambda_\omega g_\rho \rho_{03} (g_\omega \omega)^2$
        )
        # $\delta$ 场源: $s_\delta = g_\delta n_{3s}$ (无自耦合)
        s_delta = g_d * n_3s0 if g_d > 0 else np.zeros_like(r)  # $g_d n_{3s}$
        # 库仑场源: $s_A = e^2 n_\gamma$
        s_A0 = E_SQ * n_gamma0  # $e^2 n_\gamma$

        # Green 函数求解 (求解 $(-\nabla^2 + m^2) \phi = s$)
        sigma_new = solve_field_1d(s_sigma, r, dr, m_sig / fm_MeV)  # $m_\sigma^* = m_\sigma / (\hbar c)$
        omega_new = solve_field_1d(s_omega, r, dr, m_w / fm_MeV)   # $m_\omega^* = m_\omega / (\hbar c)$
        rho03_new = solve_field_1d(s_rho, r, dr, m_rho / fm_MeV)   # $m_\rho^* = m_\rho / (\hbar c)$
        delta_new = (
            solve_field_1d(s_delta, r, dr, m_d / fm_MeV)
            if g_d > 0
            else np.zeros_like(r)
        )  # $m_\delta^* = m_\delta / (\hbar c)$
        A0_new = solve_field_1d(s_A0, r, dr, 0.0, is_photon=True)  # $m=0$ 光子场

        # 阻尼混合 (防止振荡)
        alpha = 0.5
        sigma = alpha * sigma_new + (1 - alpha) * sigma  # $\sigma \leftarrow 0.5\sigma_{new} + 0.5\sigma_{old}$
        omega = alpha * omega_new + (1 - alpha) * omega
        rho03 = alpha * rho03_new + (1 - alpha) * rho03
        delta_f = alpha * delta_new + (1 - alpha) * delta_f
        A0 = alpha * A0_new + (1 - alpha) * A0

        # 更新密度 (保持 Woods-Saxon 形状但更新归一化)
        dens0 = initial_densities(r, A, Z, N, R)
        n_v0 = dens0["n_v"]
        n_s0 = dens0["n_s"]
        n_30 = dens0["n_3"]
        n_3s0 = dens0["n_3s"]
        n_gamma0 = dens0["n_gamma"]

        # 计算能量
        ens = compute_binding_energy(
            Z, N, sigma, omega, rho03, delta_f, A0, params, r, dr
        )
        E_total = ens["E_total"]

        if prev_E is not None and abs(E_total - prev_E) < conv_tol:
            print(
                f"  Converged after {it + 1} iterations "
                f"(ΔE = {abs(E_total - prev_E):.2e} MeV)"
            )
            break
        prev_E = E_total
    else:
        print(f"  Warning: did not converge within {max_iter} iterations")

    # 最终结果
    ens = compute_binding_energy(
        Z, N, sigma, omega, rho03, delta_f, A0, params, r, dr
    )

    # 均方根半径
    weight_r = 4.0 * math.pi * r ** 4 * dr  # $d\langle r^2 \rangle = 4\pi r^4 dr$
    r_p_sq = np.sum(weight_r * n_gamma0) / Z if Z > 0 else 0.0  # $\langle r_p^2 \rangle = \int r^2 n_\gamma d^3r / Z$
    r_n_sq = np.sum(weight_r * n_v0) / N if N > 0 else 0.0  # $\langle r_n^2 \rangle = \int r^2 n_v d^3r / N$

    return {
        "Z": Z,
        "N": N,
        "A": A,
        "r": r,
        "sigma": sigma,
        "omega": omega,
        "rho03": rho03,
        "delta": delta_f,
        "A0": A0,
        "n_s": n_s0,
        "n_v": n_v0,
        "n_3": n_30,
        "n_3s": n_3s0,
        "n_gamma": n_gamma0,
        "E_total": E_total,
        "E_per_nucleon": ens["E_per_nucleon"],  # $E_{tot}/A$
        "r_n": math.sqrt(max(r_n_sq, 0.0)),  # $\sqrt{\langle r_n^2 \rangle}$
        "r_p": math.sqrt(max(r_p_sq, 0.0)),  # $\sqrt{\langle r_p^2 \rangle}$
        "r_charge": math.sqrt(max(r_p_sq, 0.0) + 0.8 ** 2),  # $\sqrt{\langle r_p^2 \rangle + 0.8^2}$ (加入质子电荷半径)
        "converged": prev_E is not None and abs(E_total - prev_E) < conv_tol,
    }


def compute_Pb208(params_name: str = "FSUGold") -> dict:
    """计算 $^{208}$Pb."""
    from .params import PARAMSETS

    params = PARAMSETS.get(params_name, FSU_GOLD_FN)
    return compute_finite_nucleus(Z=82, N=126, params=params)


def compute_Sn132(params_name: str = "FSUGold") -> dict:
    """计算 $^{132}$Sn."""
    from .params import PARAMSETS

    params = PARAMSETS.get(params_name, FSU_GOLD_FN)
    return compute_finite_nucleus(Z=50, N=82, params=params)
