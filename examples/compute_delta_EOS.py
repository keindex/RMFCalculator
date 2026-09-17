r"""
compute_delta_EOS.py - 计算 RMF 模型 (FSUGold / FSU-J0 / FSU-δ6.7 / FSU-δ6.2) 的核物质 / 中子星 EOS

基于学位论文第四章表 4-2 的四组参数:

- 对称核物质 (SNM, $x_p = 0.5$) 与纯中子物质 (PNM, $x_p = 0$):
  E/A($\rho$)、$P(\rho)$、对称能 $E_{\rm sym}(\rho)$;
- β 平衡物质 (npeμ): 能量密度、压强、质子分数;
- 饱和性质: $n_0, E_0, K_0$;
- 对比绘图: 对称核物质 EOS、对称能、β 平衡 EOS (论文图 4-22 / 4-23 / 4-28 风格)。

无 $\delta$ 介子的模型 (FSUGold, FSU-J0) 使用 13 元参数向量 ($g_\delta = \Lambda_{\sigma\delta} = 0$);
含 $\delta$ 介子的模型 (FSU-δ6.7, FSU-δ6.2) 使用 13 元参数向量与
``RMF_INEOS`` / ``RMF_betaEOS``。

用法:
    python compute_delta_EOS.py                # 全部 4 个模型
    python compute_delta_EOS.py --model FSU    # 仅单个模型
    python compute_delta_EOS.py --plot         # 对比绘图
    python compute_delta_EOS.py --write out.csv  # 输出 β 平衡 EOS 到 csv (单个模型)
"""

import argparse
from pathlib import Path

import numpy as np
from scipy.signal import savgol_filter

from RMFCalculator.eos.unit import fm_MeV, m_n
from RMFCalculator.eos.parameters import get_theta
from RMFCalculator.eos.RMF_INEOS import compute_INEOS
from RMFCalculator.eos.RMF_betaEOS import compute_EOS

IMG_DIR = Path(__file__).resolve().parent.parent / "img"

# 论文图 4-22/4-23/4-28 中的线型: 红虚线 FSUGold, 绿点虚线 FSU-J0,
# 蓝实线 FSU-δ6.7, 紫点线 FSU-δ6.2。
MODEL_STYLES = {
    "FSU":         dict(color="red",    ls="--",  label="FSUGold"),
    "FSU_J0":      dict(color="green",  ls="-.",  label="FSU-J0"),
    "FSU_DELTA67": dict(color="blue",   ls="-",   label="FSU-$\\delta$6.7"),
    "FSU_DELTA62": dict(color="purple", ls=":",   label="FSU-$\\delta$6.2"),
}
MODEL_ORDER = ["FSU", "FSU_J0", "FSU_DELTA67", "FSU_DELTA62"]


def _is_delta(theta):
    """含 δ 介子模型为 g_delta != 0 或 Lambda_sigma_delta != 0。"""
    theta = np.asarray(theta)
    # 13 元向量中 index 7 = g_delta, index 12 = Lambda_sigma_delta
    return (theta.shape == (13,)) and (theta[7] != 0.0 or theta[12] != 0.0)


def _ineos(theta, x_p, n_points=200, dt=0.005, rho_0=0.1505):
    """无限核物质求解器 (含/不含 δ 介子由同一套代码处理)。"""
    return compute_INEOS(theta, x_p=x_p, n_points=n_points, dt=dt, rho_0=rho_0)


def fluiddelta(theta, x_p, n_points=200, dt=0.005, rho_0=0.1505, rho_max=0.9, filter_unphysical=True):
    r"""
    计算给定质子分数 $x_p$ 的 E/A 与 P (过滤非物理解)。

    Args:
        theta (array): 10 元或 13 元 RMF 参数向量
        x_p (float): 质子分数
        n_points (int): 密度点数下限 (覆盖不足 rho_max 时自动加密)
        dt (float): 密度步长因子
        rho_0 (float): 参考饱和密度 (fm$^{-3}$)
        rho_max (float): 最大密度 (fm$^{-3}$), 默认 0.9
        filter_unphysical (bool): 是否过滤非物理解

    Returns:
        tuple: $(\rho,\, E/A,\, P)$
    """
    n_points = max(n_points, int((rho_max - 0.1 * rho_0) / (dt * rho_0)) + 1)
    EoS = _ineos(theta, x_p=x_p, n_points=n_points, dt=dt, rho_0=rho_0)
    rho = EoS[:, 0]
    eps = EoS[:, 1] * fm_MeV
    P = EoS[:, 2] * fm_MeV
    binding = eps / rho - m_n * fm_MeV

    if filter_unphysical:
        # 仅过滤求解器崩溃产生的非物理解 (E/A 异常负值, 如 -460 MeV)。
        # 不设上限: 高密度 (ρ > 0.3 fm⁻³) 时 E/A 物理上单调增长,
        # SNM 可达 ~10 MeV, PNM 可达 ~70 MeV, 旧版 `binding < 5.0` 会误删这些点。
        mask = np.isfinite(binding) & (binding > -25.0)
        return rho[mask], binding[mask], P[mask]
    return rho, binding, P


def symmetry_energy(theta, n_r=0.1):
    r"""
    计算对称能 $E_{\rm sym}(n) = E_{\rm PNM}(n) - E_{\rm SNM}(n)$ 及交叉密度处斜率。

    Args:
        theta (array): 10 元或 13 元 RMF 参数向量
        n_r (float): 参考密度 (fm$^{-3}$), 默认 0.1

    Returns:
        dict: {'rho', 'E_sym', 'Esym_c', 'L_c'}
    """
    rho_s, e_s, _ = fluiddelta(theta, 0.5)
    rho_p, e_p, _ = fluiddelta(theta, 0.0)

    lo = max(rho_s.min(), rho_p.min())
    hi = min(rho_s.max(), rho_p.max())
    rho_c = rho_s[(rho_s >= lo) & (rho_s <= hi)]
    e_snm = np.interp(rho_c, rho_s, e_s)
    e_pnm = np.interp(rho_c, rho_p, e_p)
    e_sym = e_pnm - e_snm

    de_sym = np.gradient(e_sym, rho_c)
    Esym_c = np.interp(n_r, rho_c, e_sym)
    L_c = 3.0 * n_r * np.interp(n_r, rho_c, de_sym)

    return {'rho': rho_c, 'E_sym': e_sym, 'Esym_c': float(Esym_c), 'L_c': float(L_c)}


def saturation_delta(theta):
    r"""
    计算对称核物质饱和密度与饱和性质 ($n_0, E_0, K_0$)。

    Returns:
        dict: {'rho_0', 'E0', 'K0'}
    """
    rho, e, P = fluiddelta(theta, 0.5, n_points=300, dt=0.004, rho_max=0.6)

    de_dn = np.gradient(e, rho)
    crossings = np.where(np.diff(np.sign(de_dn)))[0]
    if len(crossings) == 0:
        raise RuntimeError("未找到饱和密度 (dE/dn 过零点)。")
    idx = crossings[0]
    n0 = np.interp(0.0, de_dn[idx:idx + 2], rho[idx:idx + 2])
    E0 = np.interp(n0, rho, e)

    h = rho[1] - rho[0]
    wl = max(21, int(0.02 / h) | 1)
    dP_dn = savgol_filter(P, wl, 5, deriv=1, delta=h)
    K0 = 9.0 * np.interp(n0, rho, dP_dn)

    return {'rho_0': float(n0), 'E0': float(E0), 'K0': float(K0)}


def beta_eos(theta, n_points=124, dt=0.05, rho_0=0.1505):
    r"""
    计算 β 平衡 (npeμ) 中子星物质 EOS。

    含/不含 $\delta$ 介子模型均由 ``RMF_betaEOS.compute_EOS`` 统一处理
    (不含 $\delta$ 时 10 元 $\theta$ 补齐后走同一套代码)。

    Returns:
        ndarray: (n_points, 8) 数组:
            [0] $\rho$ (fm$^{-3}$), [1] $\varepsilon$ (fm$^{-4}$), [2] $P$ (fm$^{-4}$),
            [3-6] $\mu_p, \mu_n, \mu_e, \mu_\mu$ (fm$^{-1}$), [7] 质子分数 $x_p$
    """
    return compute_EOS(theta, n_points=n_points, dt=dt, rho_0=rho_0)


def report_model(name, theta):
    """计算并打印单个模型的性质。"""
    print("=" * 62)
    if _is_delta(theta):
        print(f"模型: {name}   (含 δ 介子, θ13, "
              f"g_δ={theta[7]:.4f}, Λ_σδ={theta[12]:.6f})")
    else:
        print(f"模型: {name}   (无 δ 介子, θ13, "
              f"g_ρ={theta[5]:.4f}, Λ_ω={theta[9]:.6f})")
    print("=" * 62)

    sat = saturation_delta(theta)
    print(f"饱和密度  n_0       = {sat['rho_0']:.4f} fm^-3")
    print(f"结合能    E_0       = {sat['E0']:.3f} MeV")
    print(f"不可压缩  K_0       = {sat['K0']:.1f} MeV")

    sym = symmetry_energy(theta)
    print(f"对称能    E_sym(n_c) = {sym['Esym_c']:.2f} MeV   (n_c = 0.1 fm^-3)")
    print(f"对称能斜率 L(n_c)    = {sym['L_c']:.2f} MeV")

    return sat, sym


def main():
    parser = argparse.ArgumentParser(description="计算 FSUGold / FSU-J0 / FSU-δ6.7 / FSU-δ6.2 EOS")
    parser.add_argument("--model", type=str, default="ALL",
                        help="模型: FSU / FSU_J0 / FSU_DELTA67 / FSU_DELTA62 / ALL (默认 ALL)")
    parser.add_argument("--plot", action="store_true", help="绘图 (全部模型对比)")
    parser.add_argument("--write", type=str, default="", help="输出 beta-EOS csv 路径 (仅单个模型时)")
    parser.add_argument("--n_points", type=int, default=124, help="β EOS 密度点数")
    args = parser.parse_args()

    if args.model.upper() == "ALL":
        names = MODEL_ORDER
    else:
        names = [args.model.upper()]

    if args.write and len(names) != 1:
        parser.error("--write 需要同时指定 --model (单个模型)")

    theta_sets = {n: get_theta(n) for n in names}
    results = {}

    for n in names:
        theta = theta_sets[n]
        sat, sym = report_model(n, theta)

        beta = beta_eos(theta, n_points=args.n_points)
        rho_b = beta[:, 0]
        eps_b = beta[:, 1] * fm_MeV
        P_b = beta[:, 2] * fm_MeV
        xp_b = beta[:, 7]

        print(f"\nβ 平衡 EOS ({len(beta)} 点):")
        print(f"{'rho(fm-3)':<12}{'eps(MeV/fm3)':<16}{'P(MeV/fm3)':<16}{'x_p':<10}")
        for i in range(min(8, len(beta))):
            print(f"{rho_b[i]:<12.4f}{eps_b[i]:<16.4f}{P_b[i]:<16.4f}{xp_b[i]:<10.4f}")

        results[n] = {
            'theta': theta, 'sat': sat, 'sym': sym,
            'beta': beta, 'rho': rho_b, 'eps': eps_b, 'P': P_b,
        }

        if args.write:
            out = np.column_stack([rho_b, eps_b, P_b, xp_b])
            np.savetxt(args.write, out, delimiter=",",
                       header=f"{n}: rho(fm-3),eps(MeV/fm3),P(MeV/fm3),x_p",
                       comments="")
            print(f"\n已输出 β 平衡 EOS: {args.write}")

    if args.plot:
        plot_compare(results)
        print(f"\n已保存 {IMG_DIR / 'delta_eos_compare.png'}")


def _cs2(results):
    r"""
    计算 β 平衡物质的声速平方 $C_s^2 = dP/d\varepsilon$ (论文式 4.70)。

    用 Savitzky-Golay 平滑求导抑制数值噪声 (与饱和性质 K0 处理一致)。
    """
    cs2 = {}
    for n, r in results.items():
        eps = r['eps']
        P = r['P']
        # 只在使用共同能量密度区间内求导, 避免边界外推噪声
        cs2[n] = np.gradient(P, eps)
    return cs2


def plot_compare(results):
    r"""
    对比四组模型 EOS (论文图 4-22/4-23/4-28 风格):

    (a) 对称核物质 E/A($\rho$);      (b) 对称能 $E_{\rm sym}(\rho)$;
    (c) β 平衡物质 $P(\varepsilon)$;  (d) β 平衡 $P(\rho)$;
    (e) 质子分数 $x_p(\rho)$;         (f) 声速平方 $C_s^2(\rho)$。
    """
    import matplotlib.pyplot as plt

    styles = {n: MODEL_STYLES[n] for n in results}
    cs2 = _cs2(results)

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))

    # (a) SNM E/A vs ρ
    ax = axes[0, 0]
    for n, r in results.items():
        rho, e, _ = fluiddelta(r['theta'], 0.5)
        ax.plot(rho, e, ls=styles[n]["ls"], color=styles[n]["color"],
                lw=1.6, label=styles[n]["label"])
    ax.set_xlabel(r'$\rho$ (fm$^{-3}$)')
    ax.set_ylabel(r'$E/A$ (MeV)')
    ax.axhline(0, color='k', lw=0.6)
    ax.legend(); ax.grid(alpha=0.3)
    ax.set_title('(a) SNM $E/A(\\rho)$')

    # (b) 对称能
    ax = axes[0, 1]
    for n, r in results.items():
        ax.plot(r['sym']['rho'], r['sym']['E_sym'], ls=styles[n]["ls"], color=styles[n]["color"],
                lw=1.6, label=styles[n]["label"])
    ax.axvline(0.1, color='k', ls='--', lw=0.8, alpha=0.6)
    ax.set_xlabel(r'$\rho$ (fm$^{-3}$)')
    ax.set_ylabel(r'$E_{\rm sym}$ (MeV)')
    ax.legend(); ax.grid(alpha=0.3)
    ax.set_title('(b) Symmetry energy')

    # (c) β 平衡 P vs ε
    ax = axes[0, 2]
    for n, r in results.items():
        ax.plot(r['eps'], r['P'], ls=styles[n]["ls"], color=styles[n]["color"],
                lw=1.6, label=styles[n]["label"])
    ax.set_xlabel(r'$\varepsilon$ (MeV/fm$^3$)')
    ax.set_ylabel(r'$P$ (MeV/fm$^3$)')
    ax.legend(); ax.grid(alpha=0.3)
    ax.set_title('(c) β-equilibrium EOS $P(\\varepsilon)$')

    # (d) β 平衡 P vs ρ
    ax = axes[1, 0]
    for n, r in results.items():
        ax.plot(r['rho'], r['P'], ls=styles[n]["ls"], color=styles[n]["color"],
                lw=1.6, label=styles[n]["label"])
    ax.set_xlabel(r'$\rho$ (fm$^{-3}$)')
    ax.set_ylabel(r'$P$ (MeV/fm$^3$)')
    ax.legend(); ax.grid(alpha=0.3)
    ax.set_title('(d) β-equilibrium $P(\\rho)$')

    # (e) 质子分数 x_p vs ρ
    ax = axes[1, 1]
    for n, r in results.items():
        ax.plot(r['rho'], r['beta'][:, 7], ls=styles[n]["ls"], color=styles[n]["color"],
                lw=1.6, label=styles[n]["label"])
    ax.set_xlabel(r'$\rho$ (fm$^{-3}$)')
    ax.set_ylabel(r'$x_p$')
    ax.legend(); ax.grid(alpha=0.3)
    ax.set_title('(e) Proton fraction (β-equilibrium)')

    # (f) 声速平方 C_s^2 vs ρ
    ax = axes[1, 2]
    for n, r in results.items():
        ax.plot(r['rho'], cs2[n], ls=styles[n]["ls"], color=styles[n]["color"],
                lw=1.6, label=styles[n]["label"])
    ax.axhline(1 / 3.0, color='orange', ls='-', lw=1.2,
               label=r'Conformal $C_s^2=1/3$')
    ax.set_xlabel(r'$\rho$ (fm$^{-3}$)')
    ax.set_ylabel(r'$C_s^2$')
    ax.legend(); ax.grid(alpha=0.3)
    ax.set_title('(f) Speed of sound squared')

    fig.suptitle('EOS comparison: FSUGold / FSU-J0 / FSU-$\\delta$6.7 / FSU-$\\delta$6.2',
                 fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(IMG_DIR / "delta_eos_compare.png", dpi=150)


if __name__ == "__main__":
    main()