r"""
green_function.py - 介子场 Green 函数求解

求解一维微分方程 $- \nabla^2 \phi + m^2 \phi = s(r)$，
边界条件 $\phi(0)$ 有限、$\phi(\infty) \to 0$。
"""

import math

import numpy as np


def solve_field_1d(
    source: np.ndarray,
    r: np.ndarray,
    dr: float,
    mass: float,
    is_photon: bool = False,
) -> np.ndarray:
    r"""
    一维微分方程求解: $-\phi'' + m^2 \phi = s(r)$.

    使用边界条件 $\phi(0)$ 有限, $\phi(\infty) \to 0$.

    球对称下 $\nabla^2 \phi = \frac{1}{r^2} \frac{d}{dr} \left( r^2 \frac{d\phi}{dr} \right)$
    近似为一维: $-\phi'' + m^2 \phi = s(r)$

    Args:
        source: 源项数组 $s(r)$。
        r: 径向网格 (fm)。
        dr: 网格步长 (fm)。
        mass: 介子质量 (fm$^{-1}$)。
        is_photon: 是否为光子场 (Coulomb)。

    Returns:
        与 ``r`` 同形状的场数组 $\phi(r)$。
    """
    N = len(r)
    phi = np.zeros(N)

    if is_photon:
        # 光子场: $\nabla^2 A_0 = -\rho \Rightarrow A_0(r) = Q(r)/r$ (球对称)
        # 累积电荷 $Q(r) = \int_0^r 4\pi r'^2 \rho(r') dr'$
        charge = np.cumsum(4.0 * math.pi * r[:-1] ** 2 * source[:-1] * dr)  # $Q(r) = \int 4\pi r^2 \rho dr$
        # $A_0(r) = Q(r)/r$ 近似
        phi[:-1] = charge / r[:-1]  # $A_0 = Q/r$
        phi[-1] = phi[-2] if N > 1 else 0.0
    else:
        # 有质量场: 对球对称场使用 $u=r\phi$，则
        # $-u'' + m^2 u = r s(r)$，并满足 $u(0)=u(R)=0$。
        # 该形式同时保留了球对称 Laplace 算子的 $2\phi'/r$ 项。
        r_full = np.concatenate(([0.0], r))
        source_full = np.concatenate(([source[0]], source))
        interior = slice(1, N)
        diagonal = 2.0 / dr ** 2 + mass ** 2
        off_diagonal = -1.0 / dr ** 2

        matrix = np.diag(np.full(N - 1, diagonal))
        matrix += np.diag(np.full(N - 2, off_diagonal), 1)
        matrix += np.diag(np.full(N - 2, off_diagonal), -1)
        rhs = r_full[interior] * source_full[interior]
        u = np.zeros(N + 1)
        u[interior] = np.linalg.solve(matrix, rhs)
        phi = u[1:] / r

    return phi
