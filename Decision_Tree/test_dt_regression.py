from sklearn.datasets import load_boston
from decision_tree import dt, predict


def test_boston():
    boston = load_boston()
    # 説明変数
    X_array = boston.data
    # 目的変数
    y_array = boston.target

    vertexs = dt(X_array, y_array, n_min=50)

    x = X_array[:100, ]
    y = y_array[:100]
    n = x.shape[0]
    s = int(n/10)
    SS = 0

    # k=10のk-fold
    for h in range(1, 11):
        test = list(range(h*s-s+1, h*s))
        train = list(set(list(range(n))) - set(test))
        vertexs = dt(x[train, ], y[train])
        for t in test:
            SS += (y[t]-predict(x[t, ], vertexs))**2
    print(SS)


if __name__ == "__main__":
    test_boston()
