import numpy as np
from RMFCalculator.eos import compute_EOS_delta, PolyInterpolate, get_theta
from RMFCalculator.tov import OutputMR
from RMFCalculator.tov.unit import km, Msun
from RMFCalculator.eos.unit import fm_MeV, gcm3_to_MeVfm3, dyncm2_to_MeVfm3

# 加载真实的壳层 EOS (Tolos_crust_out.txt 第 4/5 列已是 cgs 单位: g/cm^3, dyn/cm^2)
Tolos_crust = np.loadtxt(r'Tolos_crust_out.txt')
eps_crust_T = Tolos_crust[:, 3]      # g/cm^3
pres_crust_T = Tolos_crust[:, 4]     # dyn/cm^2
eps_com, pres_com = PolyInterpolate(eps_crust_T, pres_crust_T)


# FSUGold 参数: 统一从 RMFCalculator.eos.parameters.get_theta 获取 (fm^-1 单位)。
# 注意: 早期代码在此手动除以 fm_MeV 并误用 g_sigma^2 = 107.5751,
# 现已统一为正确的 g_sigma^2 = 112.1996 (见学位论文表 4-2)。
theta = get_theta("FSU_DELTA62")
print(f'使用参数: {theta}')
# 计算内核 EOS (β 平衡, 输出 MeV/fm^3, 需换算回 cgs 才能与壳层连接)
x = compute_EOS_delta(theta, n_points=200)
eps_kernel_MeVfm3 = x[:, 1] * fm_MeV          # MeV/fm^3
pres_kernel_MeVfm3 = x[:, 2] * fm_MeV         # MeV/fm^3
eps_kernel = eps_kernel_MeVfm3 / gcm3_to_MeVfm3   # g/cm^3
pres_kernel = pres_kernel_MeVfm3 / dyncm2_to_MeVfm3  # dyn/cm^2
print(f'EOS 计算完成: {len(eps_kernel)} 个点')

# 连接壳层和内核 (仅保留压强高于壳层末端的核心点, 保证单调递增)
mask = pres_kernel > pres_com[-1]
eps_kernel = eps_kernel[mask]
pres_kernel = pres_kernel[mask]
eps_total = np.array([*eps_com, *eps_kernel])
pres_total = np.array([*pres_com, *pres_kernel])
print(f'壳层 {len(eps_com)} 点 + 内核 {len(eps_kernel)} 点 = {len(eps_total)} 点')

# 计算 MR
MR = OutputMR('', eps_total, pres_total)
print(f'MR 计算完成: {MR.shape}')

# 提取有效数据
arr = np.asarray(MR, dtype=float)
radius = arr[:, 1] / km
mass = arr[:, 0] / Msun
valid = np.isfinite(radius) & np.isfinite(mass) & (radius > 4) & (radius < 20) & (mass > 0.1) & (mass < 3)

print(f'最大质量: {np.max(mass[valid]):.3f} M_sun')
print(f'对应半径: {radius[valid][np.argmax(mass[valid])]:.3f} km')

# 绘制质量半径关系图
import matplotlib.pyplot as plt
plt.figure(figsize=(8, 6))
plt.scatter(radius[valid], mass[valid], c='steelblue', s=30, alpha=0.8)
plt.xlabel('Radius (km)', fontsize=12)
plt.ylabel('Mass (M_sun)', fontsize=12)
plt.title('Mass-Radius Relation', fontsize=14)
plt.xlim(4, 20)
plt.ylim(0, 3)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('mass_radius.png', dpi=150)
plt.show()
print('图表已保存为 mass_radius.png')