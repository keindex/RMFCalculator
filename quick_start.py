import math

import numpy as np
from RMFCalculator.eos import compute_EOS, PolyInterpolate
from RMFCalculator.tov import OutputMR
from RMFCalculator.tov.unit import g_cm_3, dyn_cm_2, km, Msun, MeV, fm

# 加载真实的壳层 EOS
Tolos_crust = np.loadtxt(r'Tolos_crust_out.txt')
eps_crust_T = Tolos_crust[:, 3] * g_cm_3
pres_crust_T = Tolos_crust[:, 4] * dyn_cm_2
eps_com, pres_com = PolyInterpolate(eps_crust_T, pres_crust_T)

# FSU 参数
MeV_to_fm = 1.0 / 197.327
FSU_params = {
    'm_sigma': 491.5 * MeV_to_fm,
    'm_omega': 782.5 * MeV_to_fm,
    'm_rho': 763.0 * MeV_to_fm,
    'g_sigma': math.sqrt(107.5751),
    'g_omega': math.sqrt(204.5469),
    'g_rho': math.sqrt(138.4701),
    'kappa': 1.4203 * MeV_to_fm,
    'lambda_0': 0.023762,
    'zeta':  0.06,
    'Lambda_w':  0.03,
}


theta = np.array([
    FSU_params['m_sigma'], FSU_params['m_omega'], FSU_params['m_rho'],
    FSU_params['g_sigma'], FSU_params['g_omega'], FSU_params['g_rho'],
    FSU_params['kappa'], FSU_params['lambda_0'], FSU_params['zeta'], FSU_params['Lambda_w']
])

# 计算 EOS
x = compute_EOS(eps_com, pres_com, theta)
eps_kernel, pres_kernel = x[1], x[2]
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