"""
compute_quantities.py - 计算核物质饱和性质 K0, J0, L

基于论文第一章的公式:

对称核物质的每核子结合能 E_0(n) 在饱和密度 n_0 附近展开:

    E_0(n) = E_0(n_0) + K_0/(2!) χ² + J_0/(3!) χ³ + O(χ⁴),  χ ≡ (n-n_0)/(3n_0)

其中:
    K_0 = 9 n² ∂²E_0/∂n² |_{n=n_0}    (不可压缩系数)
    J_0 = 27 n³ ∂³E_0/∂n³ |_{n=n_0}  (偏斜系数)

对称能在密度 n_r 处展开:

    E_sym(n) = E_sym(n_r) + L χ_r + K_sym/(2!) χ_r² + O(χ_r³),  χ_r ≡ (n-n_r)/(3n_r)

其中:
    L = 3 n_r ∂E_sym/∂n |_{n=n_r}    (对称能斜率系数)

对称能通过 E_sym(n) = E(n, α=1) - E(n, α=0) 计算,
其中 α=1 为纯中子物质 (PNM), α=0 为对称核物质 (SNM)。

用法:
    python compute_quantities.py
"""

from RMFCalculator.eos import compute_saturation_properties, get_theta


def main():

    # 使用 FSUGold 参数 (集中定义于 RMFCalculator.eos.parameters.FSU_PARAMS_MEV)。
    # 注意: 质量与 kappa 以 MeV 原值存储, get_theta 内部完成 /fm_MeV 换算,
    # 无需在此再除以 fm_MeV (否则双重换算, 参数失真)。
    theta = get_theta("FSU")

    print("计算 FSU 参数下的核物质饱和性质...\n")
    q = compute_saturation_properties(theta)

    print("=" * 55)
    print(f"  饱和密度  rho_0   = {q['rho_0']:.4f} fm^-3")
    print(f"  结合能    E0      = {q['E0']:.2f} MeV")
    print(f"  不可压缩  K0      = {q['K0']:.1f} MeV")
    print(f"  偏斜系数  J0      = {q['J0']:.1f} MeV")
    print(f"  对称能    E_sym(n0) = {q['Esym0']:.2f} MeV")
    print(f"  对称能斜率 L(n0)    = {q['L0']:.1f} MeV")
    print(f"  对称能曲率 K_sym(0.1)= {q['Ksym']:.1f} MeV")
    print("=" * 55)

    print("\n文献参考值:")
    print("  K0 ≈ 240 ± 20 MeV")
    print("  E_sym(n_0) ≈ 31.6 ± 0.92 MeV")
    print("  L(n_0) ≈ 58.9 ± 16.5 MeV")
    print("  FSUGold (学位论文表 4-4): K0=229.2, J0=-521.6, E_sym=32.59, L=60.50")
    print(f"  本脚本: K0={q['K0']:.1f}, J0={q['J0']:.1f}, E_sym(n0)={q['Esym0']:.2f}, L(n0)={q['L0']:.1f}")


if __name__ == "__main__":
    main()