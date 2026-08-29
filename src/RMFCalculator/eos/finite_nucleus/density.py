r"""
density.py - 核密度分布

提供 Woods-Saxon 密度分布近似及核半径计算。
"""

import math

import numpy as np


def woods_saxon(r: np.ndarray, R: float, a: float, C: float = 1.0) -> np.ndarray:
    r"""Woods-Saxon 密度分布.

    $$\rho(r) = \frac{C}{1 + \exp\left(\frac{r - R}{a}\right)}$$

    Args:
        r: 径向坐标数组 (fm)。
        R: 核半径 (fm)。
        a: 表面厚度参数 (fm)。
        C: 归一化常数。

    Returns:
        与 ``r`` 同形状的密度数组。
    """
    x = (r - R) / a  # $x = (r - R)/a$
    # 避免溢出
    with np.errstate(over='ignore', invalid='ignore'):
        exp_x = np.exp(np.clip(x, -500, 500))  # $\exp(x)$ with clipping
    return C / (1.0 + exp_x)  # $\rho = C / (1 + e^x)$


def nuclear_radius(A: int) -> float:
    r"""核半径.

    $$R = r_0 A^{1/3}, \quad r_0 \approx 1.2 \text{ fm}$$

    Args:
        A: 质量数。

    Returns:
        核半径 (fm)。
    """
    return 1.2 * A ** (1.0 / 3.0)  # $R = 1.2 \times A^{1/3}$


def initial_densities(
    r: np.ndarray,
    A: int,
    Z: int,
    N: int,
    R: float,
    a: float = 0.5,
) -> dict:
    r"""构造初始核子密度分布.

    返回 Woods-Saxon 形状的标量密度 $n_s$、矢量密度 $n_v$、同位旋矢量密度 $n_3$、
    同位旋赝标量密度 $n_{3s}$、质子密度 $n_\gamma$。

    归一化: $\int n_v(r) d^3r = A$
    $$n_v(r) = \rho_{WS}(r) \cdot \frac{3A}{4\pi R^3}$$
    $$n_s(r) = n_v(r) \quad \text{(近似)}$$
    $$n_3(r) = n_v(r) \frac{N-Z}{A}$$
    $$n_{3s}(r) = n_s(r) \frac{N-Z}{A}$$
    $$n_\gamma(r) = n_v(r) \frac{Z}{A}$$

    Args:
        r: 径向网格 (fm)。
        A: 质量数。
        Z: 质子数。
        N: 中子数。
        R: 核半径 (fm)。
        a: 表面厚度 (fm)。

    Returns:
        dict: 包含 ``n_s``, ``n_v``, ``n_3``, ``n_3s``, ``n_gamma`` 的字典。
    """
    n_v0 = woods_saxon(r, R, a) * (3.0 * A / (4.0 * math.pi * R ** 3))  # $n_v = \rho_{WS} \cdot 3A/(4\pi R^3)$
    n_s0 = n_v0.copy()  # $n_s \approx n_v$
    n_30 = n_v0 * (N - Z) / A  # $n_3 = n_v (N-Z)/A$
    n_3s0 = n_s0 * (N - Z) / A  # $n_{3s} = n_s (N-Z)/A$
    n_gamma0 = n_v0 * Z / A if Z > 0 else np.zeros_like(n_v0)  # $n_\gamma = n_v Z/A$
    return {
        "n_s": n_s0,
        "n_v": n_v0,
        "n_3": n_30,
        "n_3s": n_3s0,
        "n_gamma": n_gamma0,
    }
