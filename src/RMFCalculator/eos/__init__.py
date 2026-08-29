r"""
eos 子包 - RMF 状态方程生成模块

提供 RMF (Relativistic Mean-Field) 模型下核物质状态方程 (EOS) 的计算工具:

- 无限核物质 (对称核物质 SNM / 纯中子物质 PNM): ``RMF_INEOS``
- $\beta$ 平衡中子星物质 (npeμ): ``RMF_betaEOS``
- 饱和性质 ($K_0, J_0, L$ 等): ``saturation``
- 壳层 EOS 拼接: ``crust_EOS``
- 参数管理: ``parameters``
- 物理常数与单位换算: ``unit``

含/不含 $\delta$ 介子的模型统一由 ``RMF_INEOS`` / ``RMF_betaEOS`` 模块处理
($g_\delta = \Lambda_{\sigma\delta} = 0$ 退化为无 $\delta$ 情形)。
"""

from .crust_EOS import PolyInterpolate
from .parameters import (
    build_theta_13_from_params,
    FSU_PARAMS_MEV,
    FSU_DELTA_PARAMS_MEV,
    FSU_J0_PARAMS_MEV,
    FSU_DELTA67,
    FSU_DELTA62,
    get_theta,
)
from .RMF_INEOS import (
    compute_INEOS,
    Energy_density_Pressure as Energy_density_Pressure_delta,
    initial_values as initial_values_ineos,
    functie as functie_ineos,
)
from .RMF_betaEOS import (
    compute_EOS,
    initial_values as initial_values_beta,
    functie as functie_beta,
    Energy_density_Pressure as Energy_density_Pressure_beta,
)
from .saturation import (
    fluid_energy,
    find_saturation_density,
    compute_saturation_properties,
)
from .finite_nucleus import (
    compute_finite_nucleus,
    compute_Pb208,
    compute_Sn132,
    PARAMSETS,
    NucleusParams,
    FSU_GOLD_FN,
    FSU_DELTA67_FN,
)
