import numpy as np
from RMFCalculator.eos import compute_EOS, PolyInterpolate
from RMFCalculator.tov import OutputMR
from RMFCalculator.tov.unit import g_cm_3, dyn_cm_2, km, Msun, MeV, fm

# 加载真实的壳层 EOS
Tolos_crust = np.loadtxt(r'd:\dev\CompactObject\Test_Case\Tolos_crust_out.txt')
eps_crust_T = Tolos_crust[:, 3] * g_cm_3
pres_crust_T = Tolos_crust[:, 4] * dyn_cm_2
eps_com, pres_com = PolyInterpolate(eps_crust_T, pres_crust_T)

# FSU 参数
MeV_to_fm = 1.0 / 197.327
FSU_params = {
    'm_sigma': 491.5 * MeV_to_fm,
    'm_omega': 782.5 * MeV_to_fm,
    'm_rho': 763.0 * MeV_to_fm,
    'g_sigma': 10.217,
    'g_omega': 13.584,
    'g_rho': 8.919,
    'kappa': 1.950e-4,
    'lambda_0': 5.651e-3,
    'zeta': 3.483e-3,
    'Lambda_w': 1.864e-2,
}

theta = np.array([
    FSU_params['m_sigma'], FSU_params['m_omega'], FSU_params['m_rho'],
    FSU_params['g_sigma'], FSU_params['g_omega'], FSU_params['g_rho'],
    FSU_params['kappa'], FSU_params['lambda_0'], FSU_params['zeta'], FSU_params['Lambda_w']
])

# 计算 EOS
eps_kernel, pres_kernel = compute_EOS(eps_com, pres_com, theta)
print(f'EOS 计算完成: {len(eps_kernel)} 个点')

# 连接壳层和内核
eps_total = np.array([*eps_com, *eps_kernel])
pres_total = np.array([*pres_com, *pres_kernel])

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