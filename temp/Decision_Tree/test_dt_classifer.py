from sklearn import datasets
import numpy as np
from decision_tree import dt, predict


def test_classifer():
    data = datasets.load_iris()

    x = data.data
    y = data.target
    n = x.shape[0]
    s = int(n/10)

    for f in ["mis.math", "gini", "entropy"]:
        cv_scores = []
        for h in range(1, 11):
            test = list(range(h*s-s+1, h*s))
            train = list(set(list(range(n))) - set(test))
            vertexs = dt(x[train, ], y[train], f=f, n_min=4)
            SS = 0
            for t in test:
                if y[t] == predict(x[t, ], vertexs):
                    SS += 1
            cv_scores.append(SS/len(test))
        print(f)
        print("accuracy is ", np.mean(cv_scores))
        print("----")


if __name__ == "__main__":
    test_classifer()
