"""
compute_saturation.py - 计算核饱和密度

饱和密度 n_0 的定义: 对称核物质 (x_p = 0.5) 在零温下压强为零的密度, 即

    P(n_0) = n^2 ∂E_0(n)/∂n |_{n=n_0} = 0

本脚本通过 RMF 模型计算对称核物质的状态方程 (能量密度 ε 与压强 P),
并数值求解 P = 0 的密度点, 同时输出每核子结合能 E/A 等饱和性质。

用法:
    python compute_saturation.py
"""

import argparse
from pathlib import Path

from RMFCalculator.eos import find_saturation_density, get_theta
from RMFCalculator.eos.saturation import fluid_energy

IMG_DIR = Path(__file__).resolve().parent.parent / "img"


def main():
    parser = argparse.ArgumentParser(description="计算核饱和密度")
    parser.add_argument("--plot", action="store_true", help="绘制饱和密度图")
    parser.add_argument("--dt", type=float, default=0.05, help="密度步长因子 (默认 0.05)")
    args = parser.parse_args()

    # 使用 FSUGold 参数 (集中定义于 RMFCalculator.eos.parameters.FSU_PARAMS_MEV)
    theta = get_theta("FSU")

    print("使用 FSU 参数计算对称核物质饱和密度...\n")

    result = find_saturation_density(theta, dt=args.dt)

    print("=" * 50)
    print(f"  饱和密度  n_0   = {result['rho_0']:.4f} fm^-3")
    print(f"  每核子结合能 E/A = {result['E_A']:.2f} MeV")
    print(f"  饱和处能量密度   = {result['energy_dens']:.2f} MeV/fm^3")
    print("=" * 50)

    # 与文献值对比
    print("\n文献参考值:")
    print("  n_0 ≈ 0.16 ± 0.01 fm^-3")
    print("  E/A ≈ -16 ± 1 MeV")
    print(f"  本脚本结果: n_0 = {result['rho_0']:.4f} fm^-3, E/A = {result['E_A']:.2f} MeV")

    if args.plot:
        plot_saturation(theta, result['rho_0'], dt=args.dt)


def plot_saturation(theta, rho_sat, dt=0.05):
    """绘制对称核物质 E/A(ρ) 曲线并标记饱和密度。"""
    import matplotlib.pyplot as plt

    rho, binding, _ = fluid_energy(theta, 0.5, n_points=124, dt=dt)

    plt.figure(figsize=(7, 5))
    plt.plot(rho, binding, 'o-', label='E/A(n)')
    plt.axvline(rho_sat, color='r', ls='--', label=f'$n_0$={rho_sat:.3f} fm$^{{-3}}$')
    plt.axhline(0, color='k', lw=0.8)
    plt.xlabel(r'$\rho$ (fm$^{-3}$)')
    plt.ylabel(r'$E/A$ (MeV)')
    plt.title("Binding energy of symmetric nuclear matter")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(IMG_DIR / "saturation.png", dpi=150)
    print(f"\nSaved figure: {IMG_DIR / 'saturation.png'}")


if __name__ == "__main__":
    main()