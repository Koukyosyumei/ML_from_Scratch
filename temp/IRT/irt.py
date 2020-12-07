import pystan
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

import utils


class IRT4Order:
    def __init__(self, df: pd.core.frame.DataFrame,
                 nc: int, stm=None, dat=None, D=1):
        """
        Attruute:
            df: 分析対象のデータフレーム
            nc: 設問のカテゴリ数 (段階数)
            stm: stanモデル - defaultではNone
                 Noneの場合は、set_modelで作成
            dat: stanモデルに食わせるデータ
                 Noneの場合は、set_modelで作成
            D: 定数
        """
        self.df = df
        self.nj = df.shape[1]
        self.ni = df.shape[0]
        self.nc = nc
        self.D = D

        self.fit = None
        self.result = None
        self.alpha = None
        self.beta = None
        self.theta = None

    def set_model(self, sample=None):
        """ Summary
        stanモデルとそれに投入するdatデータの作成

        Input
            sample: sample数 (Noneの場合はdf内の全てのデータを使う)
        """
        self.stm = pystan.StanModel(model_code=utils.model)
        if sample is not None:
            self.ni = sample

        self.dat = {"y": self.df.sample(
            self.ni).values, "nj": self.nj, "ni": self.ni, "nc": self.nc,
            "D": self.D}

    def model_fit(self, n_itr=2000, chains=4, n_warmup=1000, n_jpbs=-1,
                  algorithm="NUTS", verbose=False):
        """ Summary
        stanモデルの実行

        n_itr:      default = 2000
        chains:     default = 4
        n_warmup:   default = 1000
        n_jobs:     default = -1
        algorithm:  default = "NUTS
        verbose:    default = False
        """
        self.fit = self.stm.sampling(data=self.dat, iter=n_itr,
                                     chains=chains, n_jobs=-1,
                                     warmup=n_warmup, algorithm="NUTS",
                                     verbose=False)

    def extract(self):
        """ Summmary
        stanモデルの結果であるfitから、推定されたパラメータを取得
        """
        self.result = self.fit.extract()
        self.alpha = self.result["a"].mean(axis=0)
        self.beta = self.result["b"].mean(axis=0)
        self.theta = self.result["theta"].mean(axis=0)

    def plot(self, sort="Q", save_name="irt_Q"):
        """ Summary
        sort: プロットする時、設問ごとにするか段階ごとにするか default = "Q"
              Q: 設問ごと, L: 段階ごと
        save_name: 保存する画像名(拡張子は含めない) default = "irt_Q"
        """
        theta_ = np.arange(np.min(self.theta), np.max(self.theta),
                           step=(np.max(self.theta)
                                 - np.min(self.theta))/self.ni)
        irt_plot = np.array([[1/(1+np.exp(-self.alpha[j]*(theta_
                                                          - self.beta[j, c])))
                              for c in range(self.nc)]
                             for j in range(self.nj)])
        if sort == "Q":
            for j in range(self.nj):
                for c in range(self.nc):
                    plt.title("irt_plot_Q{q_num}".format(q_num=j))
                    plt.plot(irt_plot[j][c], label=str(c))
                    plt.xticks(np.arange(0, self.ni, step=10),
                               np.round(np.arange(np.min(self.theta),
                                                  np.max(self.theta),
                                                  step=(np.max(self.theta)
                                                        - np.min(self.theta))
                                                  / 10),
                                        decimals=2))
                    plt.legend()
                    plt.savefig(save_name+str(j))
                plt.show()

        elif sort == "L":
            for c in range(self.nc):
                for j in range(self.nj):
                    plt.title("irt_plot_Level{level}".format(level=c))
                    plt.plot(irt_plot[j][c], label="Q"+str(j))
                    plt.xticks(np.arange(0, self.ni, step=10),
                               np.round(np.arange(np.min(self.theta),
                                                  np.max(self.theta),
                                                  step=(np.max(self.theta)
                                                        - np.min(self.theta))
                                                  / 10),
                                        decimals=2))
                    plt.legend()
                    plt.savefig(save_name+str(c))
                plt.show()

        else:
            print("sort must be Q or L")
