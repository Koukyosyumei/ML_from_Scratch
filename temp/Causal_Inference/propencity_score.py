import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt


class Propencity_Score():

    """ Summary line

    傾向スコアに関する計算・分析等を行うクラス

    Attribute
        df       (pd.DataFrame) : 分析対象のデータフレーム、目的変数・介入変数・共変数を含む
        target    (str)         : 目的変数
        treatment (str)         : 介入変数
        method    (str) : 傾向スコアをもとに効果を測る方法 mathing (マッチング) or weight (逆さ重み)
        model           : 傾向スコアを計算するためのモデル
        bins            : 傾向スコアをどの程度の粒度でカテゴリー化するか mathingの時に使用

    """

    def __init__(self, df, target, treatment, method="matching", model=None, bins=0.01):
        self.biased_df = df.copy()
        self.target = target   # 目的変数
        self.treatment = treatment   # 介入変数
        self.covariates = list(
            set(df.columns.tolist()) - set([target, treatment]))  # 共変数
        self.method = method
        self.model = model

        self.biased_covariates = self.biased_df.drop(
            [self.treatment, self.target], axis=1)
        self.biased_treatment = self.biased_df[self.treatment]

        # -------------------------------------------------
        # 居変数・それをグループ化したものつくる
        self.propencity_score = self.cululate_ps()
        self.biased_df["propencity_score"] = self.propencity_score
        self.biased_df["propencity_group"] = self.get_groups(bins)

        # --------------------------------------------------
        # 調整済みのデータフレームを作成
        self.biased_df_positive = self.biased_df[self.biased_df[self.treatment] == 1]
        self.biased_df_negative = self.biased_df[self.biased_df[self.treatment] == 0]

        self.adjusted_df = self.get_matched_df()

        # -------------------------------------------------
        # 各共変数のASAMを計算
        self.asam_unadjusted, self.asam_adjusted = self.get_asam_of_covariate()

    def cululate_ps(self):
        """
        傾向スコアを計算
        """

        if self.model == None:
            glm = LogisticRegression()
            glm.fit(self.biased_covariates, self.biased_treatment)
            propencity_score = glm.predict_proba(self.biased_covariates)[:, 1]

        # 他のモデルの処理も後で頑張る

        return propencity_score

    def get_groups(self, bins):
        """
        傾向スコアを指定の粒度でカテゴリー化
        """
        return list(map(lambda x: int(x/bins), self.propencity_score))

    def get_matched_df(self):
        """
        傾向スコアマッチングを行う
        """
        group_numbers = self.biased_df["propencity_group"].unique()

        matched_group_list = []
        for group_number in group_numbers:
            specified_group_positive = self.biased_df_positive[
                self.biased_df_positive["propencity_group"] == group_number]
            specified_group_negative = self.biased_df_negative[
                self.biased_df_negative["propencity_group"] == group_number]

            positive_counts = specified_group_positive.shape[0]
            negative_counts = specified_group_negative.shape[0]

            if positive_counts > negative_counts:
                matched_pair = (specified_group_positive.sample(
                    negative_counts), specified_group_negative)
            elif negative_counts > positive_counts:
                matched_pair = (specified_group_positive,
                                specified_group_negative.sample(positive_counts))
            else:
                matched_pair = (specified_group_positive,
                                specified_group_negative)

            matched_group_list.append(pd.concat(matched_pair, axis=0))

        matched_df = pd.concat(matched_group_list, axis=0)
        return matched_df

    def get_coefficients(self):
        """
        調整前・後の、介入変数の目的変数に対する効果を計算
        """
        X_matched = np.array(self.adjusted_df[self.treatment]).reshape(-1, 1)
        Y_matched = np.array(self.adjusted_df[self.target])

        lr_matched = LinearRegression()
        lr_matched.fit(X_matched, Y_matched)

        print("coefficient_adjusted = ", lr_matched.coef_)

        X_biased = np.array(self.biased_df[self.treatment]).reshape(-1, 1)
        Y_biased = np.array(self.biased_df[self.target])

        lr_biased = LinearRegression()
        lr_biased.fit(X_biased, Y_biased)

        print("coefficient_unadjusted = ", lr_biased.coef_)

        return (lr_matched.coef_, lr_biased.coef_)

    def _asam_treatment(self, df):
        """
        介入変数をセグメントとして、ASAMを計算
        """
        feature_positive = df[df[self.treatment] == 1].drop([self.treatment], axis=1)
        feature_negative = df[df[self.treatment] == 0].drop([self.treatment], axis=1)

        mean_positive = np.mean(feature_positive)
        mean_negative = np.mean(feature_negative)

        std_positive = np.std(feature_positive)
        std_negative = np.std(feature_negative)

        n_positive = len(feature_positive)
        n_negative = len(feature_negative)

        sc = (n_positive * (std_positive**2) + n_negative
              * (std_negative**2)) / (n_positive + n_negative)

        asam = np.abs((mean_positive - mean_negative) / np.sqrt(sc))

        return asam

    def get_asam_of_covariate(self):
        """
        各共変数のASAMを計算
        """
        biased_covatiate = self.biased_df.loc[:, self.covariates + [self.treatment]]
        adjusted_covariate = self.adjusted_df.loc[:, self.covariates + [self.treatment]]

        asam_unadjusted = self._asam_treatment(biased_covatiate)
        asam_adjusted = self._asam_treatment(adjusted_covariate)

        return (asam_unadjusted, asam_adjusted)

    def plot_asam(self, save=True, title="covariated_asam"):
        """
        共変数のASAMを描画
        """
        plt.scatter(
            self.asam_unadjusted.values, self.asam_unadjusted.index.values, c="b", label="unadjusted")
        plt.scatter(
            self.asam_adjusted.values, self.asam_adjusted.index.values, c="r", label="adjusted")
        plt.grid()
        plt.legend()
        plt.xlabel("ASAM")
        plt.title(title)

        plt.savefig(title+"png")
