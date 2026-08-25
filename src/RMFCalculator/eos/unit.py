r"""
unit.py - 物理常数与单位换算

提供 RMF 计算所需的物理常数 (自然单位 fm$^{-1}$) 和单位换算因子。

自然单位制 ($\hbar = c = 1$):
- 质量、动量、能量均以 fm$^{-1}$ 为单位
- 换算关系: $1\ \mathrm{fm}^{-1} = \hbar c\ \mathrm{MeV} = 197.327\ \mathrm{MeV}$
- 能量密度: fm$^{-4}$ (自然单位) $\times \hbar c =$ MeV/fm$^3$
- 压强: fm$^{-4}$ (自然单位) $\times \hbar c =$ MeV/fm$^3$

粒子量子数矩阵:
- ``Matrix_b``: 重子 (质子/中子) 的 [重子数, 电荷, 同位旋第三分量 $I_3$]
- ``Matrix_l``: 轻子 (电子/μ子) 的 [重子数, 电荷, 自旋]
"""

import numpy as np


c = 2.99792458e10

# 引力常数 $G$, 单位: cm³/(g·s²)
# $$G = 6.67428\times 10^{-8} \ \mathrm{cm^3/(g\cdot s^2)}$$
G = 6.67428e-8

# 太阳质量 $M_\odot$, 单位: g
# $$M_\odot = 1.989\times 10^{33} \ \mathrm{g}$$
Msun = 1.989e33

# 压强单位换算: dyn/cm² → MeV/fm³
# $$1 \ \mathrm{dyn/cm^2} = \frac{1}{1.6022\times 10^{33}} \ \mathrm{MeV/fm^3}$$
dyncm2_to_MeVfm3 = 1.0 / (1.6022e33)

# 能量密度单位换算: g/cm³ → MeV/fm³
# $$1 \ \mathrm{g/cm^3} = \frac{1}{1.7827\times 10^{12}} \ \mathrm{MeV/fm^3}$$
gcm3_to_MeVfm3 = 1.0 / (1.7827e12)

# ħc 换算因子: $ħc \approx 197.327$ MeV·fm, 用于 fm⁻¹ ↔ MeV
# $$\hbar c = 197.327053 \ \mathrm{MeV\cdot fm}$$
fm_MeV = 197.327053

# ==================== 粒子质量 (自然单位: fm⁻¹) ====================
# 电子质量 $m_e \approx 0.511$ MeV → fm⁻¹
# $$m_e = 0.511 \ \mathrm{MeV} \times \frac{1}{197.327} \ \mathrm{fm^{-1}} = 2.5896\times 10^{-3} \ \mathrm{fm^{-1}}$$
m_e = 2.5896041 * 10**-3

# μ 子质量 $m_\mu \approx 105.66$ MeV → fm⁻¹
# $$m_\mu = 105.66 \ \mathrm{MeV} \times \frac{1}{197.327} \ \mathrm{fm^{-1}} = 0.5354 \ \mathrm{fm^{-1}}$$
m_mu = 0.5354479981

# 中子质量 $m_n \approx 938.92$ MeV → fm⁻¹
# $$m_n = 938.92 \ \mathrm{MeV} \times \frac{1}{197.327} \ \mathrm{fm^{-1}} = 4.7584 \ \mathrm{fm^{-1}}$$
m_n = 4.7583690772

# 质子质量 $m_p$ (RMF 中通常取 $m_p = m_n$)
# $$m_p \approx m_n = 4.7584 \ \mathrm{fm^{-1}}$$
m_p = 4.7583690772

# 核子自旋 $J_B = 1/2$, 简并度因子 $(2J_B+1) = 2$
# $$(2J_B + 1) = 2\times\frac{1}{2} + 1 = 2$$
J_B = 1 / 2.0

# 重子数 $b_B = 1$ (核子)
b_B = 1

# 轻子质量列表: $[m_e, m_\mu]$
m_l = [m_e, m_mu]

# 核子质量列表: $[m_p, m_n]$
m_b = [m_p, m_n]

# 重子量子数矩阵 Matrix_b[i]: 行 i=0 为质子, i=1 为中子
# 列含义: [重子数 $b$, 电荷 $q$, 同位旋第三分量 $I_3$, σ耦合, ω耦合, ρ耦合]
# 质子: $b=1, q=+1, I_3=+1/2$; 中子: $b=1, q=0, I_3=-1/2$

# $$\mathrm{Matrix\_b} = \begin{pmatrix} 1 & 1 & +1/2  \\ 1 & 0 & -1/2  \end{pmatrix}$$
Matrix_b = np.array(
    [[1.0, 1.0, 1 / 2.0], [1.0, 0.0, -1 / 2.0]]
)

# 轻子量子数矩阵 Matrix_l[j]: 行 j=0 为电子, j=1 为 μ 子
# 列含义: [重子数 $b=0$, 电荷 $q=-1$, 自旋 $J=1/2$]

# $$\mathrm{Matrix\_l} = \begin{pmatrix} 0 & -1 & 1/2 \\ 0 & -1 & 1/2 \end{pmatrix}$$
Matrix_l = np.array([[0.0, -1.0, 1 / 2.0], [0.0, -1.0, 1 / 2.0]])
