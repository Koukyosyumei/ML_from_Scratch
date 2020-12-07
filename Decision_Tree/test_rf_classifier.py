from tqdm import tqdm
from sklearn import datasets
from numpy.random import choice
from random_forest import random_forest, rf_predict


def test():

    data = datasets.load_iris()
    x = data.data
    y = data.target
    n = x.shape[0]
    shuffle_index = choice(list(range(n)), n)
    x = x[shuffle_index, ]
    y = y[shuffle_index]
    x_train, y_train = x[:100, ], y[:100, ]
    x_valid, y_valid = x[100:, ], y[100:, ]

    rf = random_forest(x_train, y_train, m=3, f="mis_match", n_min=2)
    y_predict = []
    for i in tqdm(range(x_valid.shape[0])):
        y_predict.append(rf_predict(x_valid[i, ], rf))

    print("accuracy is ", sum(y_predict == y_valid) / len(y_valid))


if __name__ == "__main__":
    test()
