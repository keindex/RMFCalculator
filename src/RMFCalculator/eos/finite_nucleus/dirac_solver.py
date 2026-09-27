r"""
dirac_solver.py - 球对称相对论平均场的径向 Dirac 方程求解。

理论基础:
- 采用球对称单粒子形式
  $$
  \psi_{n\kappa m}(\mathbf{r}) = \frac{1}{r}
  \begin{pmatrix}
  G_{n\kappa}(r) \Omega_{\kappa m}(\hat{r}) \\
  i F_{n\kappa}(r) \Omega_{-\kappa m}(\hat{r})
  \end{pmatrix},
  $$
  其中 $G(r)$ 和 $F(r)$ 分别是径向大、小分量。
- 径向 Dirac 方程为
  $$
  \begin{pmatrix}
  V_q + M_q^* & -\frac{d}{dr} + \frac{\kappa}{r} \\
  \frac{d}{dr} + \frac{\kappa}{r} & V_q - M_q^*
  \end{pmatrix}
  \binom{G}{F} = \varepsilon \binom{G}{F},
  $$
  其中 $M_q^* = M - g_\sigma\sigma - g_\delta\delta\,\tau_3^q$。
- 归一化条件为

  $$\int_0^\infty \left[G^2(r) + F^2(r)\right] dr = 1.$$ 
  
- 球对称密度满足
  $$n_v(r)=\sum_{occ}\frac{G^2+F^2}{4\pi r^2},\qquad
    n_s(r)=\sum_{occ}\frac{G^2-F^2}{4\pi r^2}.$$ 

本模块在给定平均场 $(\sigma, \omega, \rho_{03}, \delta, A_0)$ 时，求解各个
$\kappa$ 通道的径向 Dirac 本征态，并由占据轨道重建局域密度。
"""

from dataclasses import dataclass

import numpy as np
from scipy.sparse import bmat, diags
from scipy.sparse.linalg import eigsh

from RMFCalculator.eos.unit import fm_MeV
from .params import NucleusParams


@dataclass
class DiracOrbit:
    r"""记录已占据的单粒子 Dirac 轨道。"""

    species: str
    kappa: int
    energy: float
    degeneracy: int
    occupancy: int
    upper: np.ndarray
    lower: np.ndarray


def _hamiltonian(
    r: np.ndarray,
    dr: float,
    mass: np.ndarray,
    vector: np.ndarray,
    kappa: int,
):
    r"""构造径向 Dirac Hamiltonian 矩阵.

    采用离散差分近似
    $$
    \frac{d}{dr} \approx \frac{f(r+dr)-f(r-dr)}{2dr},
    $$
    因此矩阵保持 Hermitian 形式，便于使用 `eigsh` 求本征态。
    """
    interior = slice(1, -1)
    ri = r[interior]
    n = len(ri)
 
    derivative = diags(
        [-np.ones(n - 1), np.ones(n - 1)], [-1, 1], shape=(n, n), format="csc"
    ) / (2.0 * dr)
    k_over_r = diags(kappa / ri, format="csc")
    upper_diag = diags(vector[interior] + mass[interior], format="csc")
    lower_diag = diags(vector[interior] - mass[interior], format="csc")

    return bmat(
        [[upper_diag, -derivative + k_over_r], [derivative + k_over_r, lower_diag]],
        format="csc",
    )


def _channel_states(
    r: np.ndarray,
    dr: float,
    mass: np.ndarray,
    vector: np.ndarray,
    kappa: int,
    number: int = 8,
) -> list[tuple[float, np.ndarray, np.ndarray]]:
    r"""返回单个 $\kappa$ 通道中的低能正能态。"""
    hamiltonian = _hamiltonian(r, dr, mass, vector, kappa)
    reference = float(np.max(mass))
    count = min(number, hamiltonian.shape[0] - 2)

    try:
        energies, vectors = eigsh(
            hamiltonian, k=count, sigma=reference - 0.15, which="LM"
        )
    except (RuntimeError, ValueError):
        return []

    states: list[tuple[float, np.ndarray, np.ndarray]] = []
    grid = r[1:-1]

    for energy, vector_state in zip(energies, vectors.T):
        # 在很轻的核中，有限盒内可能没有明显束缚态，
        # 但仍可使用低于自由质量的正能盒态作为迭代起点。
        if energy <= 0.0:
            continue

        upper = np.zeros_like(r)
        lower = np.zeros_like(r)
        upper[1:-1] = vector_state[: len(grid)]
        lower[1:-1] = vector_state[len(grid) :]

        norm = np.sqrt(np.trapz(upper * upper + lower * lower, r))
        if norm > 0.0:
            states.append((float(energy), upper / norm, lower / norm))

    return states


def solve_dirac_densities(
    r: np.ndarray,
    dr: float,
    sigma: np.ndarray,
    omega: np.ndarray,
    rho03: np.ndarray,
    delta: np.ndarray,
    A0: np.ndarray,
    Z: int,
    N: int,
    params: NucleusParams,
    kappa_max: int = 8,
    states_per_channel: int = 8,
) -> tuple[dict[str, np.ndarray], list[DiracOrbit]]:
    r"""在给定平均场下求解 Dirac 轨道并重建核子密度.

    采用同位旋约定
    $$\tau_3^p = +1, \qquad \tau_3^n = -1,$$
    并且
    $$n_3(r) = n_n(r) - n_p(r),\qquad n_{3s}(r) = n_n^s(r) - n_p^s(r).$$

    由于球对称径向波函数满足
    $\int_0^\infty (G^2 + F^2)dr = 1$，因此局域密度应使用
    $1/(4\pi r^2)$ 这一体积因子。
    """
    g_s, g_w, g_r, g_d = (
        params.g_sigma,
        params.g_omega,
        params.g_rho,
        params.g_delta,
    )
    nucleon_mass = 939.0 / fm_MeV
    species_data = {
        "p": (+1.0, Z),
        "n": (-1.0, N),
    }
    orbitals: list[DiracOrbit] = []
    density = {
        key: np.zeros_like(r)
        for key in ("n_s", "n_v", "n_3", "n_3s", "n_gamma")
    }

    r_safe = np.where(r > 0.0, r, 1.0e-12)

    for species, (tau3, particle_number) in species_data.items():
        if particle_number == 0:
            continue

        # 有效质量与矢量势:
        # $M_q^* = M - g_\sigma \sigma - g_\delta \delta \tau_3^q$
        # $V_q = g_\omega \omega + \frac{1}{2} g_\rho \rho_{03}\tau_3^q + eA_0\delta_{qp}$
        effective_mass = nucleon_mass - g_s * sigma - g_d * delta * tau3
        vector = g_w * omega + 0.5 * g_r * rho03 * tau3
        if species == "p":
            vector = vector + A0

        states: list[tuple[float, int, np.ndarray, np.ndarray]] = []
        for kappa in range(-kappa_max, kappa_max + 1):
            if kappa == 0:
                continue
            for energy, upper, lower in _channel_states(
                r, dr, effective_mass, vector, kappa, states_per_channel
            ):
                states.append((energy, kappa, upper, lower))

        states.sort(key=lambda item: item[0])
        remaining = particle_number

        for energy, kappa, upper, lower in states:
            degeneracy = 2 * abs(kappa)
            occupancy = min(remaining, degeneracy)
            if occupancy <= 0:
                continue

            orbitals.append(
                DiracOrbit(
                    species,
                    kappa,
                    energy * fm_MeV,
                    degeneracy,
                    occupancy,
                    upper,
                    lower,
                )
            )

            # 球对称体积因子:
            # $n_v(r) = \sum_{occ} (G^2 + F^2)/(4\pi r^2)$
            # $n_s(r) = \sum_{occ} (G^2 - F^2)/(4\pi r^2)$
            vector_density = occupancy * (upper * upper + lower * lower) / (
                4.0 * np.pi * r_safe ** 2
            )
            scalar_density = occupancy * (upper * upper - lower * lower) / (
                4.0 * np.pi * r_safe ** 2
            )

            sign = 1.0 if species == "n" else -1.0
            density["n_v"] += vector_density
            density["n_s"] += scalar_density
            density["n_3"] += sign * vector_density
            density["n_3s"] += sign * scalar_density
            if species == "p":
                density["n_gamma"] += vector_density

            remaining -= occupancy
            if remaining == 0:
                break

        if remaining:
            raise RuntimeError(
                f"Dirac basis cannot accommodate {remaining} {species} particles; "
                "increase kappa_max or states_per_channel"
            )

    return density, orbitals
