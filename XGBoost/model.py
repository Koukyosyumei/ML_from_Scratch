import copy

import numpy as np

from base import XGBoostTree
from utils import sigmoid


class XGBoostBase:
    def __init__(
        self,
        subsample_cols=0.8,
        min_child_weight=1,
        depth=5,
        min_leaf=5,
        learning_rate=0.4,
        boosting_rounds=5,
        lam=1.5,
        gamma=1,
        eps=0.1,
        use_ispure=True,
    ):
        self.depth = depth
        self.subsample_cols = subsample_cols
        self.eps = eps
        self.min_child_weight = min_child_weight
        self.min_leaf = min_leaf
        self.learning_rate = learning_rate
        self.boosting_rounds = boosting_rounds
        self.lam = lam
        self.gamma = gamma
        self.use_ispure = use_ispure

        self.estimators = []

    def grad(self, preds, labels):
        pass

    def hess(self, preds, labels):
        pass

    def get_init_pred(self, X, _):
        pass

    def fit(self, X, y):
        self.init_pred = self.get_init_pred(X, y)
        self.base_pred = copy.deepcopy(self.init_pred)
        for _ in range(self.boosting_rounds):
            grad = self.grad(self.base_pred, y)
            hess = self.hess(self.base_pred, y)
            boosting_tree = XGBoostTree().fit(
                X,
                y,
                grad,
                hess,
                depth=self.depth,
                min_leaf=self.min_leaf,
                lam=self.lam,
                gamma=self.gamma,
                eps=self.eps,
                min_child_weight=self.min_child_weight,
                subsample_cols=self.subsample_cols,
                use_ispure=self.use_ispure,
            )
            self.base_pred += self.learning_rate * boosting_tree.predict(X)
            self.estimators.append(boosting_tree)

    def predict_row(self, X):
        pred = np.zeros(X.shape[0])
        for estimator in self.estimators:
            pred += self.learning_rate * estimator.predict(X)

        return self.init_pred + pred


class XGBoostClassifier(XGBoostBase):
    def __init__(self, *args, **kwargs):
        super(XGBoostClassifier, self).__init__(*args, **kwargs)

    def grad(self, preds_row, labels):
        preds = sigmoid(preds_row)
        return preds - labels

    def hess(self, preds_row, _):
        preds = sigmoid(preds_row)
        return preds * (1 - preds)

    def get_init_pred(self, X, _):
        return np.full((X.shape[0], 1), 1).flatten().astype("float64")

    def predict_proba(self, X):
        return sigmoid(self.predict_row(X))

    def predict(self, X, threshold=0.5):
        predicted_probas = self.predict_proba(X)
        preds = np.where(predicted_probas > threshold, 1, 0)
        return preds


class XGBoostRegressor:
    def __init__(self, *args, **kwargs):
        super(XGBoostRegressor, self).__init__(*args, **kwargs)

    def grad(self, preds, labels):
        return 2 * (preds - labels)

    def hess(self, preds, _):
        return np.full((preds.shape[0], 1), 2).flatten().astype("float64")

    def get_init_pred(self, X, y):
        return np.full((X.shape[0], 1), np.mean(y)).flatten().astype("float64")

    def predict(self, X):
        return self.predict_row(X)
