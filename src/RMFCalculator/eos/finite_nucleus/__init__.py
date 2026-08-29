r"""
finite_nucleus 子包 - 球对称有限原子核基态的 RMF 模型计算

基于学位论文第二章 (2.6节) 的理论框架，采用简化平均场近似。
适用于教学和初步研究，完整计算需使用更精确的数值方法。

主要特点:
- 使用 Woods-Saxon 密度分布近似
- 通过 Green 函数求解 Boson 场
- 支持含/不含 $\delta$ 介子的模型

模块结构:
- :mod:`.params`         - 参数集定义
- :mod:`.density`        - 核密度分布 (Woods-Saxon)
- :mod:`.green_function` - 介子场 Green 函数求解
- :mod:`.binding_energy` - 结合能计算
- :mod:`.solver`         - 主自洽求解器

用法:
    from RMFCalculator.eos.finite_nucleus import compute_finite_nucleus

    result = compute_finite_nucleus(Z=82, N=126)
    print(f"Binding/A = {result['E_per_nucleon']:.2f} MeV")
"""

from .binding_energy import compute_binding_energy
from .density import initial_densities, nuclear_radius, woods_saxon
from .green_function import solve_field_1d
from .params import (
    FSU_DELTA67_FN,
    FSU_GOLD_FN,
    NucleusParams,
    PARAMSETS,
)
from .solver import compute_finite_nucleus, compute_Pb208, compute_Sn132

__all__ = [
    # 参数
    "NucleusParams",
    "FSU_GOLD_FN",
    "FSU_DELTA67_FN",
    "PARAMSETS",
    # 密度
    "woods_saxon",
    "nuclear_radius",
    "initial_densities",
    # Green 函数
    "solve_field_1d",
    # 结合能
    "compute_binding_energy",
    # 求解器
    "compute_finite_nucleus",
    "compute_Pb208",
    "compute_Sn132",
]
