"""

DMDモジュールの使用例

"""

import numpy as np
import matplotlib.pyplot as plt
import DMDmod as dmd

x = np.linspace(-5, 5, 100)   # 空間
t = np.linspace(0, 5, 200)    # 時間 
dt = t[1] - t[0]

xx, tt = np.meshgrid(x, t, indexing='ij')

# モード1 : 空間構造：np.exp(-(xx + 2)**2)
# 成長率 : 減衰　np.exp(-0.5 * tt)
# 固有振動数 : 2Hz 
mode1 = np.exp(-(xx + 2)**2) * np.exp(-0.5 * tt) * np.cos(4 * np.pi * tt)

# モード2 : 空間構造：np.exp(-(xx - 2)**2)
# 成長率 : 成長　np.exp(0.3 * tt)
# 固有振動数 : 5Hz 
mode2 = np.exp(-(xx - 2)**2) * np.exp(0.3 * tt) * np.cos(10 * np.pi * tt)

# 観測データ（2つのモードの足し合わせ）
X_data = mode1 + mode2

fig = plt.figure()
ax1 = fig.add_subplot(111)
# 元データのホフメラー図
cont1 = ax1.contourf(tt,xx,X_data)
plt.colorbar(cont1,ax=ax1)

exact_dmd = dmd.DMD(dt = dt, rank = 10)

Phi,eigvals,omega,b = exact_dmd.perform_dmd(X_data)

fig = plt.figure(figsize=(12,5))
ax1 = fig.add_subplot(121)
ax2 = fig.add_subplot(122)

# モードのエネルギー = 特異値の2乗
energy = exact_dmd.S**2

ax1.plot(energy)
ax1.set_yscale('log')
ax1.set_title('POD Spectrum')
ax1.set_xlabel('Mode Index')
ax1.set_ylabel('Energy')

dmd_model = dmd.windowedDMD(dt=dt,rank = 4)

reconstructed = dmd_model.run_pipeline(
    X = X_data,
    window = 50,
    overlap = 25,
    n_clusters = 2
)

plt.show()
