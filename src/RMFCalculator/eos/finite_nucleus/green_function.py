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
        # 有质量场: 使用简单迭代法
        # $\phi(r) \approx s(r) / m^2$ (局部近似, 适用于缓变场)
        # 加上边界修正
        phi_approx = source / (mass ** 2 + 1e-10)  # $\phi \approx s/m^2$

        # 平滑处理 (迭代求解 Helmholtz 方程)
        phi = phi_approx.copy()
        for _ in range(5):
            phi_new = phi.copy()
            # 五点差分算子: $\phi'' \approx (\phi_{i+1} - 2\phi_i + \phi_{i-1})/dr^2$
            for i in range(1, N - 1):
                d2phi = (phi[i + 1] - 2 * phi[i] + phi[i - 1]) / (dr ** 2)  # $\phi''$
                phi_new[i] = (d2phi - source[i]) / (mass ** 2)  # $\phi = (\phi'' - s)/m^2$
            # 边界
            phi_new[0] = phi_new[1]  # 原点正则: $\phi'(0) = 0$
            phi_new[-1] = 0.0  # 无穷远衰减: $\phi(\infty) = 0$
            phi = 0.5 * phi + 0.5 * phi_new  # 阻尼更新

        # 确保正值
        phi = np.maximum(phi, 0.0)

    return phi
