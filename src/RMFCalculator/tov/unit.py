# TOVsolver/unit.py
# ============================================================================
# 单位换算模块
# ============================================================================
#
# 本模块定义了中子星物理中常用的单位系统, 主要包括:
#
# 1. 几何单位制: $G = c = 1$
#    - 长度单位: fm (飞米, $10^{-15}$ m)
#    - 质量单位: MeV (兆电子伏特)
#    - 时间单位: 通过 $c$ 与长度单位关联
#
# 2. cgs 单位制 (厘米-克-秒):
#    - 长度: cm (厘米)
#    - 质量: g (克)
#    - 时间: s (秒)
#
# 3. 天文单位:
#    - 质量: $M_\odot$ (太阳质量)
#    - 长度: km (千米)
#
# 单位换算关系:
# - $1 \text{ fm} = 10^{-13} \text{ cm}$
# - $1 \text{ MeV}^{-1} \approx 1.973 \times 10^{-11} \text{ cm}$ (约化康普顿波长)
# - $1 M_\odot \approx 1.989 \times 10^{33} \text{ g}$
# - $1 \text{ km} = 10^5 \text{ cm}$
#
# 物理常数:
# - $c \approx 2.998 \times 10^{10} \text{ cm/s}$
# - $G \approx 6.674 \times 10^{-8} \text{ cm}^3/(\text{g} \cdot \text{s}^2)$
# - $\hbar c \approx 197.327 \text{ MeV} \cdot \text{fm}$
#
# ============================================================================

fm = 1
hbarc = 0.197327053
c = 1  # speed of light
hbar = 1  # reduced Planck constant

GeV = 1 / hbarc  # Giga-electronvolt
MeV = 1e-3 * GeV  # Mega-electronvolt

g = 5.625e26 * MeV  # gram
kg = 1e3 * g  # kilogram
cm = 1e13 * fm  # centimeter
m = 100 * cm  # meter
km = 1e5 * cm  # kilometer
s = 3e10 * cm  # second

dyn = g * cm / s**2  # dyne
dyn_cm_2 = dyn / cm**2  # dyne / cm^2
g_cm_3 = g / cm**3  # gram / cm^3
erg = dyn * cm  # ἐργον energy unit

m_n = 939.565 * MeV  # mass of neutron
n0 = 0.16 / fm**3  # saturation density

e0 = m_n * n0  # saturation energy density
G = 6.6743e-8 * dyn * cm**2 / g**2  # gravitational constant
Msun = 1.989e33 * g  # mass of sun
