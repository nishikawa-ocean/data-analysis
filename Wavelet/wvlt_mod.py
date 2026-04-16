import numpy as np
import math

class Morlet:

    def __init__(self, omega0=6.0):
        self.omega0 = omega0

    def get_s_to_f(self):
        
        """
        スケールを振動数に変換
        """

        return (self.omega0 + np.sqrt(2 + self.omega0**2)) / (4.0 * np.pi)

    def get_coi_factor(self):

        """
        スケールからCOIを計算するための係数
        """

        return np.sqrt(2)

    def psi_ft(self, s, freq):

        """
        フーリエ空間でのウェーブレット関数
        """

        Hev = np.array(freq > 0.0)
        sw = s * freq
        Psi_0 = (np.pi**(-0.25)) * np.exp(-(sw - self.omega0)**2 / 2.0) * Hev
        return Psi_0

class Paul:

    def __init__(self, m=4):
        self.m = m

    def get_s_to_f(self):
        
        """
        スケールを振動数に変換
        """

        return (2.0 * self.m + 1.0) / (4.0 * np.pi)

    def get_coi_factor(self):

        """
        スケールからCOIを計算するための係数
        """

        return 1.0 / np.sqrt(2)

    def psi_ft(self, s, freq):

        """
        フーリエ空間でのウェーブレット関数
        """

        Hev = np.array(freq > 0.0)
        const = (2**self.m) / np.sqrt(self.m * math.factorial(2 * self.m - 1))
        sw = s * freq
        return const * (sw**self.m) * np.exp(-sw) * Hev
    
class DOG:

    def __init__(self, m=2):
        self.m = m

    def get_s_to_f(self):
       
        """
        スケールを振動数に変換
        """
        return np.sqrt(self.m + 0.5) / (2.0 * np.pi)

    def get_coi_factor(self):

        """
        スケールからCOIを計算するための係数
        """

        return np.sqrt(2)

    def psi_ft(self, s, freq):

        """
        フーリエ空間でのウェーブレット関数
        """

        const = -(1j**self.m) / np.sqrt(math.gamma(self.m + 0.5))
        sw = s * freq
        return const * (sw**self.m) * np.exp(-(sw**2) / 2.0)    
    
import numpy as np
import math

class Morlet:
    """Morletウェーブレット"""
    def __init__(self, omega0=6.0):
        self.omega0 = omega0

    def get_s_to_f(self):
        return (self.omega0 + np.sqrt(2 + self.omega0**2)) / (4.0 * np.pi)

    def get_coi_factor(self):
        return np.sqrt(2)

    def psi_ft(self, s, freq):
        Hev = np.array(freq > 0.0)
        Psi_0 = (np.pi**(-0.25)) * np.exp(-(s * freq - self.omega0)**2 / 2.0) * Hev
        return Psi_0

class Paul:
    """Paulウェーブレット"""
    def __init__(self, m=4):
        self.m = m

    def get_s_to_f(self):
        return (2.0 * self.m + 1.0) / (4.0 * np.pi)

    def get_coi_factor(self):
        return 1.0 / np.sqrt(2)

    def psi_ft(self, s, freq):
        Hev = np.array(freq > 0.0)
        const = (2**self.m) * (1j**self.m) * math.factorial(self.m) / np.sqrt(np.pi * math.factorial(2 * self.m))
        sw = s * freq
        return const * (sw**self.m) * np.exp(-sw) * Hev

class DOG:
    """DOG (Derivative of Gaussian) ウェーブレット"""
    def __init__(self, m=2):
        self.m = m

    def get_s_to_f(self):
        return np.sqrt(self.m + 0.5) / (2.0 * np.pi)

    def get_coi_factor(self):
        return np.sqrt(2)

    def psi_ft(self, s, freq):
        const = -(1j**self.m) / np.sqrt(math.gamma(self.m + 0.5))
        sw = s * freq
        return const * (sw**self.m) * np.exp(-(sw**2) / 2.0)

class GMW:
    """ Generalized Morse Wavelet """
    def __init__(self, gamma=3.0, beta=20.0):
        self.gamma = gamma
        self.beta = beta

    def get_s_to_f(self):
        """
        スケールを振動数に変換
        """
        # フーリエ空間におけるウェーブレット関数のピーク周波数から変換係数を算出
        omega_peak = (self.beta / self.gamma)**(1.0 / self.gamma)

        self.omega_peak = omega_peak

        return omega_peak / (2.0 * np.pi)

    def get_coi_factor(self):
        """
        スケールからCOIを計算するための係数
        """
        return np.sqrt(2.0 * self.gamma * self.beta) / self.omega_peak

    def psi_ft(self, s, freq):
        """
        フーリエ空間でのウェーブレット関数
        """
        Hev = freq > 0.0
        Psi_0 = np.zeros_like(freq, dtype=float)
        
        sw = s * freq[Hev]
        
        # エネルギー正規化のための定数計算
        gamma_arg = (2.0 * self.beta + 1.0) / self.gamma
        const = np.sqrt((2.0 * self.gamma * (2.0**gamma_arg)) / math.gamma(gamma_arg))
        
        Psi_0[Hev] = const * (sw**self.beta) * np.exp(-(sw**self.gamma))
        return Psi_0

def cwt(tm, dt, target, wavelet, dj=0.25):
    
    """

    連続ウェーブレット変換

    参考:Torrence and compo(1998)
    
    args
    ============================
    tm      : 時間軸
    dt      : 時間刻み幅
    target  : 解析する一次元時系列
    wavelet : マザーウェーブレットのインスタンス
    dj      : スケールの解像度

    returns
    ============================
    freq_cwt : 振動数軸
    cwtu     : ウェーブレット係数(t,omega)
    COI      : Cone of Influence
    
    """
    N = len(target)
    # データ長を2のべき乗にする
    n_cwt = int(2**(np.ceil(np.log2(N)))) 
    s0 = 2.0 * dt
    J = int(np.log2(n_cwt * dt / s0) / dj)
    pi = np.pi

    s = s0 * 2.0**(dj * np.arange(0, J + 1, 1))
    x = np.zeros(n_cwt,dtype = np.complex128) 

    # 2のべき乗長にゼロパディング
    x[0:len(target)] = target[0:len(target)] - target.mean() 

    freq = 2.0 * pi * np.fft.fftfreq(n_cwt, dt)
    X = np.fft.fft(x)
    cwtu = np.zeros((J + 1, n_cwt), dtype=complex)

    for j in range(J + 1):
        
        # 各クラス固有の関数を呼び出し
        Psi_0 = wavelet.psi_ft(s[j], freq)
        Psi = np.sqrt(2.0 * pi * s[j] / dt) * Psi_0
        
        #ウェーブレット関数とデータの畳み込み　= フーリエ空間での積
        cwtu[j, :] = np.fft.ifft(X * np.conjugate(Psi)) 

    #スケールから振動数への変換
    s_to_f = wavelet.get_s_to_f() 
    freq_cwt = s_to_f / s
    
    cwtu = cwtu[:, 0:len(tm)]

    COI = np.zeros_like(tm)
    COI[0] = 0
    coi_factor = wavelet.get_coi_factor()
    COI[1:len(tm)//2] = coi_factor * s_to_f / tm[1:len(tm)//2]
    COI[len(tm)//2:-1] = coi_factor * s_to_f / (tm[-1] - tm[len(tm)//2:-1])
    COI[-1] = 0


    return freq_cwt, cwtu, COI