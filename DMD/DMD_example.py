"""

DMDモジュールの使用例

PODスペクトルからランク数を決定し、固有値をプロットする

"""

import numpy as np
import matplotlib.pyplot as plt
import DMDmod as dmd

# 仮想データの作成
x = np.linspace(-5, 5, 100)   # 空間
t = np.linspace(0, 5, 200)    # 時間 
dt = t[1] - t[0]

xx, tt = np.meshgrid(x, t, indexing='ij')

# モード1 : 空間構造：np.exp(-(xx + 2)**2)
# 成長率 : 減衰　np.exp(-0.5 * tt)
# 固有振動数 : 0.4Hz 
mode1 = 5*np.exp(-(xx + 2)**2) * np.exp(-0.2 * tt) * np.cos(0.8 * np.pi * tt - 2 * xx)

# モード2 : 空間構造：np.exp(-(xx - 2)**2)
# 成長率 : 成長　np.exp(0.3 * tt)
# 固有振動数 : 5Hz 
mode2 = np.exp(-(xx - 2)**2) * np.exp(0.3 * tt) * np.cos(10 * np.pi * tt - 3 * xx)

# ガウシアンノイズ
noise_level = 0.1  # 元の信号に対するノイズの強さ（0.1 = 10%）
noise = noise_level * np.random.standard_normal(size=mode1.shape)

# データ
X_data = mode1 + mode2 + noise

exact_dmd = dmd.DMD(dt = dt, rank = 4)

# DMDの実行
Phi,eigvals,omega,b = exact_dmd.perform_dmd(X_data)

fig = plt.figure(figsize=(12,5))
ax1 = fig.add_subplot(121)
ax2 = fig.add_subplot(122)

# モードのエネルギー = 特異値の2乗(PODスペクトル)
# 有効なモード数(ランク)を決定
energy = exact_dmd.S**2

ax1.plot(energy)
ax1.set_xscale('log')
ax1.set_yscale('log')
ax1.set_title('POD Spectrum')
ax1.set_xlabel('Mode Index')
ax1.set_ylabel('Energy')

#単位円を描画
theta = np.linspace(0, 2*np.pi, 100)
ax2.plot(np.cos(theta), np.sin(theta), linestyle='--', color='gray', label='Unit Circle')

#単位円の内側　→　減衰
#単位円の外側　→　成長
ax2.scatter(eigvals.real,eigvals.imag,color = 'blue', marker = 'o',label = 'eigenvalues')
ax2.set_title('Eigenvalues')
ax2.set_xlabel('Real part')
ax2.set_ylabel('Imaginary part')
ax2.set_aspect('equal',adjustable='box')
ax2.set_xlim(0.5,1.1)
ax2.legend()

# DMDによるスケール分離
dmd_model = dmd.windowedDMD(dt=dt,rank = 4)

# クラスター数(いくつのスケールに分離するか)　…　経験的に決定
n_clusters = 2
reconstructed = dmd_model.run_pipeline(
    X = X_data,
    window = 50,
    overlap = 45,
    n_clusters = n_clusters
)

growth_rate = dmd_model.growth_rate
flat_label  = dmd_model.flat_label
times = dmd_model.times
omega = dmd_model.omega

fig = plt.figure()
ax1 = fig.add_subplot(111)
cmap  =plt.get_cmap("tab10")
colors = [cmap(i) for i in range(n_clusters)]

# クラスターと成長率に応じた特徴ベクトル(振動数の絶対値)

eps = 1.e-2 # 成長・減衰の閾値(適当)
for n in range(n_clusters):
    growth_rate_n = growth_rate[flat_label == n]
    is_grow    = growth_rate_n > eps
    is_decay   = growth_rate_n < -eps
    is_neutral = np.abs(growth_rate_n) <= eps
    time_n    = times[flat_label==n]
    feature_n = omega[flat_label==n]
    ax1.scatter(time_n[is_grow]   ,feature_n[is_grow]/(2*np.pi)    ,color=colors[n],marker='^',label=str(n)+'_grow')
    ax1.scatter(time_n[is_decay]  ,feature_n[is_decay]/(2*np.pi)   ,color=colors[n],marker='v',label=str(n)+'_decay')
    ax1.scatter(time_n[is_neutral],feature_n[is_neutral]/(2*np.pi) ,color=colors[n],marker='o',label=str(n)+'_neutral')

ax1.set_title('Clusterwd Feature vector')
ax1.set_xlabel('time[s]')
ax1.set_ylabel('$|\omega|$[Hz]')
plt.legend()

fig = plt.figure()
ax1 = fig.add_subplot(121)
ax2 = fig.add_subplot(122)
# 元データのホフメラー図
cont1 = ax1.contourf(tt,xx,X_data)
ax1.set_xlim(0,times.max()*dt)
ax1.set_title('Original Data')
ax1.set_xlabel('time[s]')
ax1.set_ylabel('x')

# 再構成したデータのホフメラー図
reconed = np.zeros_like(reconstructed[0])
for n in range(n_clusters):
    reconed += reconstructed[n]
cont2 = ax2.contourf(tt,xx,reconed)
ax2.set_xlim(0,times.max()*dt)
ax2.set_title('Reconstructed Data')
ax2.set_xlabel('time[s]')
ax2.set_ylabel('x')

plt.colorbar(cont1,ax=ax1)
plt.colorbar(cont2,ax=ax2)

# 各クラスター(スケール)のホフメラー図
for n in range(n_clusters):
    reconed = reconstructed[n]
    ax =plt.figure().add_subplot(111)
    cont = ax.contourf(tt,xx,reconed) 
    ax.set_title("cluster #"+str(n))
    ax.set_xlim(0,times.max()*dt)
    plt.colorbar(cont,ax=ax)
plt.show()
