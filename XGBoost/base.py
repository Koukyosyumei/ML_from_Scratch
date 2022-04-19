import numpy as np
import pandas as pd


class Node:
    def __init__(
        self,
        x,
        y,
        gradient,
        hessian,
        idxs,
        subsample_cols=0.8,
        min_leaf=5,
        min_child_weight=1,
        depth=10,
        lam=1,
        gamma=1,
        eps=0.1,
        use_ispure=True,
    ):
        self.x, self.gradient, self.hessian = x, gradient, hessian
        self.idxs = idxs
        self.y = y
        self.depth = depth
        self.min_leaf = min_leaf
        self.lam = lam
        self.gamma = gamma
        self.min_child_weight = min_child_weight
        self.subsample_cols = subsample_cols
        self.eps = eps
        self.use_ispure = use_ispure

        self.row_count = len(idxs)
        self.col_count = x.shape[1]
        self.column_subsample = np.random.permutation(self.col_count)[
            : round(self.subsample_cols * self.col_count)
        ]
        self.val = self.compute_weight(
            self.gradient[self.idxs], self.hessian[self.idxs]
        )
        self.score = float("-inf")

        self.find_varsplit()

    def compute_weight(self, gradient, hessian):
        return -(np.sum(gradient) / (np.sum(hessian) + self.lam))

    def compute_gain(self, lhs, rhs):
        lhs_gradient = self.gradient[self.idxs][lhs].sum()
        rhs_gradient = self.gradient[self.idxs][rhs].sum()
        lhs_hessian = self.hessian[self.idxs][lhs].sum()
        rhs_hessian = self.hessian[self.idxs][rhs].sum()

        gain = (
            0.5
            * (
                (lhs_gradient**2 / (lhs_hessian + self.lam))
                + (rhs_gradient**2 / (rhs_hessian + self.lam))
                - (
                    (lhs_gradient + rhs_gradient) ** 2
                    / (lhs_hessian + rhs_hessian + self.lam)
                )
            )
            - self.gamma
        )
        return gain

    def find_varsplit(self):
        for c in self.column_subsample:
            self.find_greedy_split(c)
        if self.is_leaf():
            return
        x = self.split_col()

        lhs = np.nonzero(x <= self.split)[0]
        rhs = np.nonzero(x > self.split)[0]
        self.lhs = Node(
            x=self.x,
            y=self.y,
            gradient=self.gradient,
            hessian=self.hessian,
            idxs=self.idxs[lhs],
            min_leaf=self.min_leaf,
            depth=self.depth - 1,
            lam=self.lam,
            gamma=self.gamma,
            min_child_weight=self.min_child_weight,
            eps=self.eps,
            subsample_cols=self.subsample_cols,
        )
        self.rhs = Node(
            x=self.x,
            y=self.y,
            gradient=self.gradient,
            hessian=self.hessian,
            idxs=self.idxs[rhs],
            min_leaf=self.min_leaf,
            depth=self.depth - 1,
            lam=self.lam,
            gamma=self.gamma,
            min_child_weight=self.min_child_weight,
            eps=self.eps,
            subsample_cols=self.subsample_cols,
        )

    def find_greedy_split(self, var_idx):
        x = self.x.values[self.idxs, var_idx]
        for r in range(self.row_count):
            lhs = x <= x[r]
            rhs = x > x[r]

            lhs_indicies = np.nonzero(x <= x[r])[0]
            rhs_indicies = np.nonzero(x > x[r])[0]
            if (
                rhs.sum() < self.min_leaf
                or lhs.sum() < self.min_leaf
                or self.hessian[lhs_indicies].sum() < self.min_child_weight
                or self.hessian[rhs_indicies].sum() < self.min_child_weight
            ):
                continue

            curr_score = self.compute_gain(lhs, rhs)
            if curr_score > self.score:
                self.var_idx = var_idx
                self.score = curr_score
                self.split = x[r]

    def weighted_quantile_sketch(self, var_idx):
        x = self.x.values[self.idxs, var_idx]
        hessian = self.hessian[self.idxs]
        df = pd.DataFrame({"feature": x, "hess": hessian})

        df.sort_values(by=["feature"], ascending=True, inplace=True)
        hess_sum = df["hess"].sum()
        df["rank"] = df.apply(
            lambda x: (1 / hess_sum) * sum(df[df["feature"] < x["feature"]]["hess"]),
            axis=1,
        )

        for row in range(df.shape[0] - 1):
            # look at the current rank and the next ran
            rk_sk_j, rk_sk_j_1 = df["rank"].iloc[row : row + 2]
            diff = abs(rk_sk_j - rk_sk_j_1)
            if diff >= self.eps:
                continue

            split_value = (df["rank"].iloc[row + 1] + df["rank"].iloc[row]) / 2
            lhs = x <= split_value
            rhs = x > split_value

            lhs_indices = np.nonzero(x <= split_value)[0]
            rhs_indices = np.nonzero(x > split_value)[0]
            if (
                rhs.sum() < self.min_leaf
                or lhs.sum() < self.min_leaf
                or self.hessian[lhs_indices].sum() < self.min_child_weight
                or self.hessian[rhs_indices].sum() < self.min_child_weight
            ):
                continue

            curr_score = self.compute_gain(lhs, rhs)
            if curr_score > self.score:
                self.var_idx = var_idx
                self.score = curr_score
                self.split = split_value

    def split_col(self):
        return self.x.values[self.idxs, self.var_idx]

    def is_leaf(self):
        return self.is_pure() or self.score == float("-inf") or self.depth <= 0

    def is_pure(self):
        return self.use_ispure and len(np.unique(self.y[self.idxs])) == 1

    def predict(self, x):
        return np.array([self.predict_row(xi) for xi in x])

    def predict_row(self, xi):
        if self.is_leaf():
            return self.val
        else:
            node = self.lhs if xi[self.var_idx] <= self.split else self.rhs
            return node.predict_row(xi)


class XGBoostTree:
    def fit(
        self,
        x,
        y,
        gradient,
        hessian,
        subsample_cols=0.8,
        min_leaf=5,
        min_child_weight=1,
        depth=10,
        lam=1,
        gamma=1,
        eps=0.1,
        use_ispure=True,
    ):
        self.dtree = Node(
            x=x,
            y=y,
            gradient=gradient,
            hessian=hessian,
            idxs=np.array(np.arange(len(x))),
            subsample_cols=subsample_cols,
            min_leaf=min_leaf,
            min_child_weight=min_child_weight,
            depth=depth,
            lam=lam,
            gamma=gamma,
            eps=eps,
            use_ispure=use_ispure,
        )
        return self

    def predict(self, X):
        return self.dtree.predict(X.values)
