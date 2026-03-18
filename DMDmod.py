# Dynamic mode decomposition
import numpy as np
from scipy.linalg import svd,eig
from sklearn.cluster import KMeans

def create_snapshots(x,delay):
    ' create Hankel matrix from 1d data x'
    N = len(x) - delay
    X = np.array([x[i:i+delay] for i in range(N)]).T

    return X[:,:-1], X[:,1:]

class windowedDMD:

    """

    二次元データ(空間×時間)に対する動的モード分解(DMD)による
    振動数特性(変動の時間スケール)に応じたデータの再構成

    """

    def __init__(self,dt,rank):
        
        self.dt   = dt
        self.rank = rank

        self.data         = None
        self.dmd_features = None
        self.features     = None
        self.labels       = None

    def set_data(self,X):
        self.X = X

    def perform_dmd(self,Xw):

        """

        動的モード分解の実行(各窓に対し行う)

        args
        ===================================
        X   : 解析対象(二次元を想定)
        rank: モード数(別途解析により決定)
        dt  : 時間刻幅(データのサンプリング間隔)
        
        returns
        ===================================
        Phi     : モード(空間構造)
        eigvals : 固有値
        omega   : 複素振動数
                    実部：モード成長率
                    虚部：振動数
        b       : モード振幅

        """
        r  = self.rank
        dt = self.dt 
        
        X1 = Xw[:,:-1]
        X2 = Xw[:,1:]
        
        U,S,Vh = svd(X1,full_matrices=False)  # U = U, S = Sigma, Vh = V^T in CD2025 # rank truncation r (tuning parameter?)
        # print("energy ratio:",np.sum(S[:r]**2)/np.sum(S**2))

        U_r,S_r,V_r = U[:,:r], np.diag(S[:r]), Vh[:r,:] 
        
        A_tilde = U_r.T @ X2 @ V_r.T @ np.linalg.inv(S_r) #(6) in CD2025
        
        eigvals, W = eig(A_tilde)
        eigvals = eigvals[np.abs(eigvals)>1.e-12]
        Phi = X2 @ V_r.T @ np.linalg.inv(S_r) @ W # (7) in CD2025
        omega = np.log(eigvals) / dt #/ (2j*np.pi) # (10) in CD2025
        
        b = np.linalg.lstsq(Phi, X1[:,0], rcond=None)[0]
        
        # dynamics = np.array([b*eigvals**i for i in range(X1.shape[1])]).T
        # X_dmd    = (Phi @ dynamics).real
        
        return Phi,eigvals,omega,b
    
    def perform_windowed_dmd(self,window,overlap):
        
        """
        各窓に対してDMDを実行

        args
        ===================================
        X   : 解析対象(二次元を想定)
        rank: モード数(別途解析により決定)
        dt  : 時間刻幅(データのサンプリング間隔)
        
        returns
        ===================================
        self.window : 窓の長さ
        self.dmd_results : {
                   "start_time": 時間窓の始点,
                   "end_time"  : 時間窓の終点,
                   "Phi"       : モード,
                   "eigvals"   : 固有値,
                   "omega"     : 複素振動数,
                   "b"         : 初期振幅
                 }

        """

        X  = self.X
        dmd_results = []
        Nb = int(X.shape[1]/(window-overlap))
        X = X - np.mean(X,axis=1,keepdims=True)
        print("number of blocks",Nb)
        for i in range(Nb-1):
            if i*(window-overlap)+window > X.shape[1]: 
                break

            X_windowed = X[:,i*(window-overlap):i*(window-overlap)+window] 
            Phi,eigvals,omega,b = self.perform_dmd(X_windowed)

            result = {
            "start_time": i*(window-overlap),
            "end_time"  : (i+1)*(window-overlap),
            "Phi"       : Phi,
            "eigvals"   : eigvals,
            "omega"     : omega,
            "b"         : b
            }

            dmd_results.append(result)
        self.window = window
        self.dmd_results = dmd_results

    def make_frequency_features(self):

        """

        振動数から特徴ベクトルを作成(ここでは絶対値)

        returns
        =================================
        self.omega       : 振動数の絶対値をベクトル化して並べたもの
        self.growth_rate : モードの成長率   

        """

        dmd_results = self.dmd_results
        features_omega = []
        growth_rate = []
        times    = np.array([])
        for result in dmd_results:
            time_center = (result["start_time"]+result["end_time"])/2
            times = np.append(times,(np.ones_like(result["omega"])*time_center))
            for om in result["omega"]:
                features_omega.append(np.abs(np.imag(om)))
                growth_rate.append((np.real(om)))
        
        self.times = np.array(times)
        self.omega = np.array(features_omega)
        self.growth_rate = np.array(growth_rate) 
    
    def cluster_freq(self,n_clusters=2):

        """
        
        特徴ベクトルをクラスタリング
        
        args
        =================================
        n_clusters : クラスター数

        returns
        =================================
        
        self.labels  : クラスター番号(0からn_cluster-1)
        self.kmeans  : KMeansで得られるオブジェクト
        self.labeled : ベクトル化した振動数の各成分のラベル
       
        """

        dmd_results = self.dmd_results
        self.omega[self.omega == 0] = 1.e-9
        # log_features = np.log10(self.omega)
        kmeans = KMeans(n_clusters=n_clusters,random_state=0)
        # labels = kmeans.fit_predict(log_features.reshape(-1,1))
        labels = kmeans.fit_predict(self.omega.reshape(-1,1))
        
        labeled = []
        idx = 0

        for results in dmd_results:
            n = len(results["omega"])
            labeled.append(labels[idx:idx+n])
            idx += n
        
        self.n_clusters = n_clusters
        self.labels     = labels
        self.kmeans     = kmeans
        self.labeled    = labeled
        self.flat_label = np.array([x for arr in labeled for x in arr.tolist()])

    
    def recon_cluster(self,Phi,omega,b,labels,cluster_id):
        window = self.window
        dt     = self.dt 
        t      = np.arange(window) * dt
        Xrec   = np.zeros((Phi.shape[0],window),dtype = complex)

        for j in range(len(omega)):
            if labels[j] == cluster_id:
                temp  =b[j]*np.exp(omega[j]*t)
                Xrec += np.outer(Phi[:,j],temp)
        
        return Xrec.real
    
    def reconstruct(self,cluster_id):
        
        """
        
        変動の時間スケールに応じた再構成

        args
        ================================
        cluster_id : クラスター番号

        returns
        ================================
        Xfull : 再構成したデータ

        """
        
        dt = self.dt
        X = self.X
        dmd_results = self.dmd_results
        labeled = self.labeled
        window = self.window
        n,m = X.shape
        Xfull = np.zeros((n,m))
        # count = np.zeros(m)
        weightsum = np.zeros(m)
        t = np.arange(window) * dt
        win_idx = 0

        for result in dmd_results:
            start = result["start_time"]
            Xw = X[:,start:start+window]
            Phi = result["Phi"]
            omega = result["omega"]
            b = result["b"]
            labels = labeled[win_idx]
            center = (result["start_time"] + result["end_time"])*dt/2
            sigma  = (result["end_time"]-result["start_time"])*dt/8
            weight = np.exp(-(t+result["start_time"]*dt-center)**2/sigma**2) 
            Xrec = self.recon_cluster(Phi,omega,b,labels,cluster_id)
            Xfull[:,start:start+window] += Xrec*weight[np.newaxis,:]
            # count[start:start+window] += 1
            weightsum[start:start+window] += weight
            win_idx += 1
        # count[count == 0] = 1
        # Xfull /= count
        weightsum[weightsum == 0] = 1
        Xfull /= weightsum[np.newaxis,:]

        return Xfull
    
    def run_pipeline(self,X,window,overlap,n_clusters):

        """
        
        メソッドを順に呼び出して時系列を再構成

        args
        ==============================
        X          : 解析するデータ
        window     : 窓長さ
        overlap    : 窓のオーバーラップ
        n_clusters : クラスター数

        returns
        ==============================
        self.reconstructed : クラスター番号順に再構成した時系列を並べたもの
        
        """
        self.set_data(X)
        self.perform_windowed_dmd(window,overlap)
        self.make_frequency_features()
        self.cluster_freq(n_clusters)
        self.reconstructed = {}
        
        for cid in range(n_clusters):
            self.reconstructed[cid] = self.reconstruct(cluster_id=cid)
        
        return self.reconstructed

        
        


        
