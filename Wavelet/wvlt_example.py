"""
    Wavelet解析の例
    非定常性とノイズを持つ時系列に対するWavelet解析

"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import wvlt_mod as wvlt

#　サンプルデータ作成
dt = 0.01                  # サンプリング間隔 
tm = np.arange(0, 10, dt)  # 時間軸

# 1. 局所的(ガウシアン型)な低周波成分
# 周波数 2Hz
#t=3秒付近で振幅が最大

signal1 = np.sin(2.0 * np.pi * 2.0 * tm) * np.exp(-((tm - 3.0) ** 2) / 2.0)

# 2. 局所的(ステップ関数型)高周波成分
# 周波数 10Hz 
# t=6秒からt=8秒の区間のみに存在する
signal2 = np.sin(2.0 * np.pi * 10.0 * tm) * ((tm >= 6.0) & (tm <= 8.0))

# 3. ホワイトノイズ　平均0、標準偏差0.5のガウシアンノイズ
noise = np.random.normal(loc=0.0, scale=0.5, size=len(tm))

# 合成データ
target = signal1 + signal2 + noise

# マザーウェーブレット : Morlet
morlet = wvlt.Morlet(omega0=6)

# マザーウェーブレット : Paul
Paul = wvlt.Paul(m=4)

# マザーウェーブレット : DOG
DOG = wvlt.DOG(m=6)

freq_cwt,cwt_morlet,COI_morlet = wvlt.cwt(tm,dt,target,wavelet=morlet,dj=0.1)
freq_cwt,cwt_paul,COI_paul    = wvlt.cwt(tm,dt,target,wavelet=Paul,dj=0.1)
freq_cwt,cwt_dog,COI_dog     = wvlt.cwt(tm,dt,target,wavelet=DOG,dj=0.1)

morlet_power = np.abs(cwt_morlet)**2
morlet_paul  = np.abs(cwt_paul)**2
morlet_dog   = np.abs(cwt_dog)**2

mpl.rcParams.update({
    "axes.labelsize" : 15,
    "xtick.labelsize":15,
    "ytick.labelsize":15,
    "axes.titlesize":15,
    "legend.fontsize": 10
    # "colorbar.labelsize":15,
    # "colorbar.tick.labelsize":10
})

# 合成データの描画 ウェーブレット係数 & ウェーブレットパワー
fig = plt.figure(figsize=(12,8))
ax1 = fig.add_subplot(111)
ax1.plot(tm,target)
ax1.set_xlabel("time")
ax1.set_ylabel("Signal")
ax1.set_title("Raw timeseries")

fig = plt.figure(figsize=(23,15))
ax1 = fig.add_subplot(321)
ax2 = fig.add_subplot(322)
ax3 = fig.add_subplot(323)
ax4 = fig.add_subplot(324)
ax5 = fig.add_subplot(325)
ax6 = fig.add_subplot(326)

levels = 11
tt,ff = np.meshgrid(tm,freq_cwt,indexing='ij')
cont1 = ax1.contourf(tt,ff,cwt_morlet.T,cmap = 'RdBu',levels=levels)
ax1.fill_between(tm,COI_morlet,hatch='X',alpha=0.5)
ax1.set_ylabel("frequency")
ax1.set_title("Morlet coef")
ax1.set_ylim(0,15)

cont2 = ax2.contourf(tt,ff,np.log10(morlet_power).T,cmap='plasma',levels = levels)
ax2.fill_between(tm,COI_morlet,hatch='X',alpha=0.5)
ax2.set_ylim(0,15)
ax2.set_title("Morlet power")

cont3 = ax3.contourf(tt,ff,cwt_paul.T,cmap = 'RdBu',levels=levels)
ax3.fill_between(tm,COI_paul,hatch='X',alpha=0.5)
ax3.set_ylim(0,15)
ax3.set_ylabel("frequency")
ax3.set_title("Paul coef")


cont4 = ax4.contourf(tt,ff,np.log10(morlet_paul).T,cmap='plasma',levels = levels)
ax4.set_title("Paul power")
ax4.fill_between(tm,COI_paul,hatch='X',alpha=0.5)
ax4.set_ylim(0,15)

cont5 = ax5.contourf(tt,ff,cwt_dog.T,cmap = 'RdBu',levels=levels)
ax5.fill_between(tm,COI_dog,hatch='X',alpha=0.5)
ax5.set_ylim(0,15)
ax5.set_xlabel("time")
ax5.set_ylabel("frequency")
ax5.set_title("DOG coef")


cont6 = ax6.contourf(tt,ff,np.log10(morlet_dog).T,cmap='plasma',levels = levels)
ax6.fill_between(tm,COI_dog,hatch='X',alpha=0.5)
ax6.set_ylim(0,15)
ax6.set_xlabel("time")
ax6.set_title("DOG power")

plt.colorbar(cont1,ax = ax1)
plt.colorbar(cont2,ax = ax2)
plt.colorbar(cont3,ax = ax3)
plt.colorbar(cont4,ax = ax4)
plt.colorbar(cont5,ax = ax5)
plt.colorbar(cont6,ax = ax6)
plt.tight_layout
plt.legend()
plt.show()