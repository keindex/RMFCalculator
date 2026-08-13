r"""
compute_delta_EOS.py - 计算含 $\delta$ 介子的核物质 / 中子星 EOS

基于 FSUGold-$\delta$ 模型 (FSU-δ6.7, FSU-δ6.2, 学位论文第四章):

- 对称核物质 (SNM, $x_p = 0.5$) 与纯中子物质 (PNM, $x_p = 0$):
  E/A($\rho$)、$P(\rho)$、对称能 $E_{\rm sym}(\rho)$;
- β 平衡物质: 能量密度、压强、质子分数;
- 饱和性质: $n_0, E_0, K_0, K_{\rm sym}, L$ 等.

用法:
    python compute_delta_EOS.py                     # 默认 FSU-δ6.7
    python compute_delta_EOS.py --model FSU_DELTA62 # FSU-δ6.2
    python compute_delta_EOS.py --plot              # 绘图
    python compute_delta_EOS.py --write out.csv     # 输出 EOS 到 csv
"""

import argparse

import numpy as np
from scipy.signal import savgol_filter

from RMFCalculator.eos.unit import fm_MeV, m_n
from RMFCalculator.eos.parameters import get_theta
from RMFCalculator.eos.RMF_INEOS_delta import compute_INEOS_delta
from RMFCalculator.eos.RMF_betaEOS_delta import compute_EOS_delta


def fluiddelta(theta13, x_p, n_points=200, dt=0.005, rho_0=0.1505, filter_unphysical=True):
    r"""
    计算给定质子分数 $x_p$ 的 E/A 与 P (过滤非物理解)。

    与 ``saturation.fluid_energy`` 类似, 但使用含 $\delta$ 介子的 EOS。

    Returns:
        tuple: $(\rho,\, E/A,\, P)$
    """
    EoS = compute_INEOS_delta(theta13, x_p=x_p, n_points=n_points, dt=dt, rho_0=rho_0)
    rho = EoS[:, 0]
    eps = EoS[:, 1] * fm_MeV
    P = EoS[:, 2] * fm_MeV
    binding = eps / rho - m_n * fm_MeV

    if filter_unphysical:
        mask = binding > -25.0
        if x_p > 0.0:
            mask &= binding < 5.0
        return rho[mask], binding[mask], P[mask]
    return rho, binding, P


def symmetry_energy(theta13, n_r=0.1):
    r"""
    计算对称能 $E_{\rm sym}(n) = E_{\rm PNM}(n) - E_{\rm SNM}(n)$。

    Args:
        theta13 (array): 13 元参数向量
        n_r (float): 参考密度 (fm$^{-3}$)

    Returns:
        dict: {'rho': 公共密度网格, 'E_sym': 对称能数组 (MeV),
               'Esym_c': 交叉密度 n_c 处对称能, 'L_c': L(n_c)}
    """
    rho_s, e_s, _ = fluiddelta(theta13, 0.5)
    rho_p, e_p, _ = fluiddelta(theta13, 0.0)

    lo = max(rho_s.min(), rho_p.min())
    hi = min(rho_s.max(), rho_p.max())
    rho_c = rho_s[(rho_s >= lo) & (rho_s <= hi)]
    e_snm = np.interp(rho_c, rho_s, e_s)
    e_pnm = np.interp(rho_c, rho_p, e_p)
    e_sym = e_pnm - e_snm

    # L(n) = 3n dE_sym/dn
    de_sym = np.gradient(e_sym, rho_c)

    # 交叉密度 n_c ≈ 0.11/0.16 n_0 ≈ 0.102 fm^-3 (论文式)
    n_c = n_r
    Esym_c = np.interp(n_c, rho_c, e_sym)
    L_c = 3.0 * n_c * np.interp(n_c, rho_c, de_sym)

    return {'rho': rho_c, 'E_sym': e_sym, 'Esym_c': Esym_c, 'L_c': L_c}


def saturation_delta(theta13):
    r"""
    计算对称核物质饱和密度与饱和性质。

    Returns:
        dict: {'rho_0', 'E0', 'K0'}
    """
    rho, e, P = fluiddelta(theta13, 0.5, n_points=300, dt=0.004)

    # 饱和: P = n²dE/dn = 0
    de_dn = np.gradient(e, rho)
    crossings = np.where(np.diff(np.sign(de_dn)))[0]
    if len(crossings) == 0:
        raise RuntimeError("未找到饱和密度 (dE/dn 过零点)。")
    idx = crossings[0]
    n0 = np.interp(0.0, de_dn[idx:idx + 2], rho[idx:idx + 2])
    E0 = np.interp(n0, rho, e)

    # K0 = 9 dP/dn (Savitzky-Golay 平滑求导, 抑制数值噪声)
    h = rho[1] - rho[0]
    wl = max(21, int(0.02 / h) | 1)
    dP_dn = savgol_filter(P, wl, 5, deriv=1, delta=h)
    K0 = 9.0 * np.interp(n0, rho, dP_dn)

    return {'rho_0': float(n0), 'E0': float(E0), 'K0': float(K0)}


def main():
    parser = argparse.ArgumentParser(description="计算含 δ 介子的 EOS (FSUGold-δ)")
    parser.add_argument("--model", type=str, default="FSU_DELTA67",
                        help="模型: FSU_DELTA67 / FSU_DELTA62 (默认 FSU_DELTA67)")
    parser.add_argument("--plot", action="store_true", help="绘图")
    parser.add_argument("--write", type=str, default="", help="输出 beta-EOS csv 路径 (可选)")
    parser.add_argument("--n_points", type=int, default=124, help="密度点数")
    args = parser.parse_args()

    theta13 = get_theta(args.model)

    print(f"模型: {args.model}, θ13 =")
    print(f"  m_δ = {theta13[3] * fm_MeV:.1f} MeV, g_δ = {theta13[7]:.4f}, "
          f"Λ_σδ = {theta13[12]:.6f}")

    # ---- 饱和性质 ----
    sat = saturation_delta(theta13)
    print(f"\n饱和密度  n_0     = {sat['rho_0']:.4f} fm^-3")
    print(f"结合能    E0      = {sat['E0']:.3f} MeV")
    print(f"不可压缩  K0      = {sat['K0']:.1f} MeV")

    # ---- 对称能 ----
    sym = symmetry_energy(theta13)
    print(f"对称能    E_sym(n_c) = {sym['Esym_c']:.2f} MeV")
    print(f"对称能斜率 L(n_c)    = {sym['L_c']:.2f} MeV")

    # ---- β 平衡 EOS ----
    beta = compute_EOS_delta(theta13, n_points=args.n_points)
    rho_b = beta[:, 0]
    eps_b = beta[:, 1] * fm_MeV
    P_b = beta[:, 2] * fm_MeV
    xp_b = beta[:, 7]

    print(f"\nβ 平衡 EOS (前 10 点, {args.n_points} 点):")
    print(f"{'rho(fm-3)':<12}{'eps(MeV/fm3)':<16}{'P(MeV/fm3)':<16}{'x_p':<10}")
    for i in range(min(10, len(beta))):
        print(f"{rho_b[i]:<12.4f}{eps_b[i]:<16.4f}{P_b[i]:<16.4f}{xp_b[i]:<10.4f}")

    if args.plot:
        _plot(args.model, theta13, sat, sym, beta)

    if args.write:
        out = np.column_stack([rho_b, eps_b, P_b, xp_b])
        np.savetxt(args.write, out, delimiter=",",
                   header="rho(fm-3),eps(MeV/fm3),P(MeV/fm3),x_p",
                   comments="")
        print(f"\n已输出 β 平衡 EOS: {args.write}")


def _plot(model, theta13, sat, sym, beta):
    import matplotlib.pyplot as plt

    rho_s, e_s, _ = fluiddelta(theta13, 0.5)
    rho_p, e_p, _ = fluiddelta(theta13, 0.0)

    eps_b = beta[:, 1] * fm_MeV
    P_b = beta[:, 2] * fm_MeV

    fig, axes = plt.subplots(2, 2, figsize=(11, 9))

    # (a) E/A vs ρ
    ax = axes[0, 0]
    ax.plot(rho_s, e_s, 'b-', lw=1.5, label='SNM')
    ax.plot(rho_p, e_p, 'r-', lw=1.5, label='PNM')
    ax.axvline(sat['rho_0'], color='k', ls='--')
    ax.axhline(0, color='k', lw=0.6)
    ax.set_xlabel(r'$\rho$ (fm$^{-3}$)')
    ax.set_ylabel(r'$E/A$ (MeV)')
    ax.legend(); ax.grid(alpha=0.3)
    ax.set_title(f'{model}: E/A(ρ)')

    # (b) 对称能
    ax = axes[0, 1]
    ax.plot(sym['rho'], sym['E_sym'], 'g-', lw=1.5)
    ax.set_xlabel(r'$\rho$ (fm$^{-3}$)')
    ax.set_ylabel(r'$E_{\rm sym}$ (MeV)')
    ax.grid(alpha=0.3)
    ax.set_title('Symmetry energy')

    # (c) β 平衡 P vs ε
    ax = axes[1, 0]
    ax.plot(eps_b, P_b, 'm-', lw=1.5)
    ax.set_xlabel(r'$\varepsilon$ (MeV/fm$^3$)')
    ax.set_ylabel(r'$P$ (MeV/fm$^3$)')
    ax.grid(alpha=0.3)
    ax.set_title('β-equilibrium EOS')

    # (d) x_p vs ρ
    ax = axes[1, 1]
    ax.plot(beta[:, 0], beta[:, 7], 'c-', lw=1.5)
    ax.set_xlabel(r'$\rho$ (fm$^{-3}$)')
    ax.set_ylabel(r'$x_p$')
    ax.grid(alpha=0.3)
    ax.set_title('Proton fraction (β-equilibrium)')

    fig.tight_layout()
    fname = f"delta_eos_{model.lower()}.png"
    fig.savefig(fname, dpi=150)
    print(f"\nSaved figure: {fname}")


if __name__ == "__main__":
    main()