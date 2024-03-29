#include <iostream>
#include <limits>
#include <vector>
#include <cassert>
#include "model.h"
using namespace std;

int main()
{
    // --- Load Data --- //
    int num_row, num_col;
    cin >> num_row >> num_col;
    vector<vector<double>> X(num_row, vector<double>(num_col));
    vector<double> y(num_row);

    for (int i = 0; i < num_col; i++)
        for (int j = 0; j < num_row; j++)
            cin >> X[j][i];
    for (int j = 0; j < num_row; j++)
        cin >> y[j];

    // --- Check Initialization --- //
    XGBoostClassifier clf = XGBoostClassifier(1.0,
                                              -1 * numeric_limits<double>::infinity(),
                                              3, 1, 0.4, 2, 1.0,
                                              0.0, 0.1, true);

    vector<double> test_init_pred = {1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0};
    vector<double> init_pred = clf.get_init_pred(X, y);
    for (int i = 0; i < init_pred.size(); i++)
        assert(init_pred[i] == test_init_pred[i]);

    vector<double> base_pred;
    copy(init_pred.begin(), init_pred.end(), back_inserter(base_pred));
    vector<double> test_base_pred = {
        -0.26894,
        0.73106,
        -0.26894,
        0.73106,
        -0.26894,
        -0.26894,
        0.73106,
        -0.26894,
    };
    vector<double> grad = clf.get_grad(base_pred, y);
    for (int i = 0; i < grad.size(); i++)
        assert(abs(grad[i] - test_base_pred[i]) <= 1e-5);

    vector<double> hess = clf.get_hess(base_pred, y);
    vector<double> test_hess = {0.19661, 0.19661, 0.19661, 0.19661, 0.19661, 0.19661, 0.19661, 0.19661};
    for (int i = 0; i < hess.size(); i++)
        assert(abs(hess[i] - test_hess[i]) <= 1e-5);

    // --- Check Training --- //
    clf.fit(X, y);

    vector<int> test_idxs_root = {0, 1, 2, 3, 4, 5, 6, 7};
    vector<int> idxs_root = clf.estimators[0].dtree.idxs;
    for (int i = 0; i < idxs_root.size(); i++)
        assert(idxs_root[i] == test_idxs_root[i]);
    assert(clf.estimators[0].dtree.depth == 3);
    assert(clf.estimators[0].dtree.split == 16);
    assert(clf.estimators[0].dtree.var_idx == 0);
    assert(clf.estimators[0].dtree.is_leaf() == 0);

    vector<int> test_idxs_left = {0, 2, 7};
    vector<int> idxs_left = clf.estimators[0].dtree.left->idxs;
    for (int i = 0; i < idxs_left.size(); i++)
        assert(idxs_left[i] == test_idxs_left[i]);
    assert(clf.estimators[0].dtree.left->is_pure());
    assert(clf.estimators[0].dtree.left->is_leaf());
    assert(clf.estimators[0].dtree.left->val == 0.5074890528001861);

    vector<int> test_idxs_right = {1, 3, 4, 5, 6};
    vector<int> idxs_right = clf.estimators[0].dtree.right->idxs;
    for (int i = 0; i < idxs_right.size(); i++)
        assert(idxs_right[i] == test_idxs_right[i]);
    assert(!clf.estimators[0].dtree.right->is_pure());
    assert(!clf.estimators[0].dtree.right->is_leaf());
    assert(clf.estimators[0].dtree.right->val == -0.8347166357912786);
    assert(clf.estimators[0].dtree.right->var_idx == 1);

    assert(clf.estimators[0].dtree.compute_gain(clf.estimators[0].dtree.left->idxs,
                                                clf.estimators[0].dtree.right->idxs) ==
           0.7556769418984197);

    assert(clf.estimators[0].dtree.right->right->split == 25);
    assert(clf.estimators[0].dtree.right->right->var_idx == 0);

    assert(clf.estimators[0].dtree.right->right->left->is_leaf());
    assert(clf.estimators[0].dtree.right->right->right->is_leaf());
    assert(clf.estimators[0].dtree.right->right->left->val == 0.3860706492904221);
    assert(clf.estimators[0].dtree.right->right->right->val == -0.6109404045885225);

    vector<double> predict_raw = clf.predict_raw(X);
    vector<double> test_predcit_raw = {1.38379341,
                                       0.53207456,
                                       1.38379341,
                                       0.22896408,
                                       1.29495549,
                                       1.29495549,
                                       0.22896408,
                                       1.38379341};
    for (int i = 0; i < test_predcit_raw.size(); i++)
        assert((predict_raw[i] - test_predcit_raw[i]) < 1e-6);

    vector<double> predict_proba = clf.predict_proba(X);
    vector<double> test_predcit_proba = {0.79959955,
                                         0.62996684,
                                         0.79959955,
                                         0.55699226,
                                         0.78498478,
                                         0.78498478,
                                         0.55699226,
                                         0.79959955};
    for (int i = 0; i < test_predcit_proba.size(); i++)
        assert(abs(predict_proba[i] - test_predcit_proba[i]) < 1e-6);
}