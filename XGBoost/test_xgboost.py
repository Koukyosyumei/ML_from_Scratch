def test_ngboostclassifier():
    import copy

    import numpy as np
    import pandas as pd

    from model import XGBoostClassifier

    df = pd.DataFrame(columns=["x1", "x2", "y"])
    df["x1"] = [12, 32, 15, 24, 20, 25, 17, 16]
    df["x2"] = [1, 1, 0, 0, 1, 1, 0, 1]
    df["y"] = [1, 0, 1, 0, 1, 1, 0, 1]
    X = df[["x1", "x2"]]
    y = df["y"]

    clf = XGBoostClassifier(
        subsample_cols=1.0,
        min_child_weight=-float("inf"),
        depth=3,
        min_leaf=1,
        boosting_rounds=2,
        lam=1.0,
        gamma=0.0,
        learning_rate=0.4,
    )

    init_pred = clf.get_init_pred(X, y)
    assert np.allclose(
        clf.get_init_pred(X, y), np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    )

    base_pred = copy.deepcopy(init_pred)
    assert np.allclose(
        clf.grad(base_pred, y).values,
        np.array(
            [
                -0.26894,
                0.73106,
                -0.26894,
                0.73106,
                -0.26894,
                -0.26894,
                0.73106,
                -0.26894,
            ]
        ),
    )
    assert np.allclose(
        clf.hess(base_pred, y),
        np.array(
            [0.19661, 0.19661, 0.19661, 0.19661, 0.19661, 0.19661, 0.19661, 0.19661]
        ),
    )

    clf.fit(df[["x1", "x2"]], df["y"])

    assert np.allclose(clf.estimators[0].dtree.idxs, np.array([0, 1, 2, 3, 4, 5, 6, 7]))
    assert np.allclose(clf.estimators[0].dtree.lhs.idxs, np.array([0, 2, 7]))
    assert np.allclose(clf.estimators[0].dtree.rhs.idxs, np.array([1, 3, 4, 5, 6]))

    assert (
        clf.estimators[0].dtree.compute_gain(
            clf.estimators[0].dtree.lhs.idxs, clf.estimators[0].dtree.rhs.idxs
        )
        == 0.7556769418984197
    )

    assert clf.estimators[0].dtree.lhs.is_pure()
    assert not clf.estimators[0].dtree.rhs.is_pure()

    assert clf.estimators[0].dtree.split == 16
    assert clf.estimators[0].dtree.var_idx == 0

    assert clf.estimators[0].dtree.lhs.val == 0.5074890528001861
    assert clf.estimators[0].dtree.rhs.var_idx == 1

    assert clf.estimators[0].dtree.rhs.lhs.is_leaf()
    assert clf.estimators[0].dtree.rhs.rhs.split == 25
    assert clf.estimators[0].dtree.rhs.rhs.var_idx == 0

    assert clf.estimators[0].dtree.rhs.rhs.lhs.is_leaf()
    assert clf.estimators[0].dtree.rhs.rhs.rhs.is_leaf()
    assert clf.estimators[0].dtree.rhs.rhs.lhs.val == 0.3860706492904221
    assert clf.estimators[0].dtree.rhs.rhs.rhs.val == -0.6109404045885225

    first_pred = clf.init_pred + 0.4 * clf.estimators[0].predict(X)
    assert np.allclose(
        first_pred,
        np.array(
            [
                1.20299562,
                0.75562384,
                1.20299562,
                0.58022047,
                1.15442826,
                1.15442826,
                0.58022047,
                1.20299562,
            ]
        ),
    )
    assert np.allclose(
        clf.grad(first_pred, y).values,
        np.array(
            [
                -0.23094274,
                0.68040287,
                -0.23094274,
                0.64111813,
                -0.23968117,
                -0.23968117,
                0.64111813,
                -0.23094274,
            ]
        ),
    )
    assert np.allclose(
        clf.hess(first_pred, y),
        np.array(
            [
                0.17760819,
                0.21745481,
                0.17760819,
                0.23008567,
                0.18223411,
                0.18223411,
                0.23008567,
                0.17760819,
            ]
        ),
    )

    assert np.allclose(
        clf.predict_row(X),
        np.array(
            [
                1.38379341,
                0.53207456,
                1.38379341,
                0.22896408,
                1.29495549,
                1.29495549,
                0.22896408,
                1.38379341,
            ]
        ),
    )
    assert np.allclose(
        clf.predict_proba(X),
        np.array(
            [
                0.79959955,
                0.62996684,
                0.79959955,
                0.55699226,
                0.78498478,
                0.78498478,
                0.55699226,
                0.79959955,
            ]
        ),
    )
