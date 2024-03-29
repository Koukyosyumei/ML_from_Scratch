#include <vector>
#include <iterator>
#include "base.h"
#include "utils.h"
#include <limits>
#include <iostream>
using namespace std;

struct XGBoostBase
{
    double subsample_cols;
    double min_child_weight;
    int depth;
    int min_leaf;
    double learning_rate;
    int boosting_rounds;
    double lam;
    double gamma;
    double eps;
    bool use_ispure;

    vector<double> init_pred;
    vector<XGBoostTree> estimators;

    XGBoostBase(double subsample_cols_ = 0.8,
                double min_child_weight_ = -1 * numeric_limits<double>::infinity(),
                int depth_ = 5, int min_leaf_ = 5,
                double learning_rate_ = 0.4, int boosting_rounds_ = 5,
                double lam_ = 1.5, double gamma_ = 1, double eps_ = 0.1,
                bool use_ispure_ = true)
    {
        subsample_cols = subsample_cols_;
        min_child_weight = min_child_weight_;
        depth = depth_;
        min_leaf = min_leaf_;
        learning_rate = learning_rate_;
        boosting_rounds = boosting_rounds_;
        lam = lam_;
        gamma = gamma_;
        eps = eps_;
        use_ispure = use_ispure_;

        estimators.resize(boosting_rounds);
    }

    virtual vector<double> get_grad(vector<double> y_pred, vector<double> y) = 0;
    virtual vector<double> get_hess(vector<double> y_pred, vector<double> y) = 0;
    virtual vector<double> get_init_pred(vector<vector<double>> x, vector<double> y) = 0;

    void fit(vector<vector<double>> x, vector<double> y)
    {
        int row_count = y.size();
        init_pred = get_init_pred(x, y);
        vector<double> base_pred;
        copy(init_pred.begin(), init_pred.end(), back_inserter(base_pred));

        for (int i = 0; i < boosting_rounds; i++)
        {
            vector<double> grad = get_grad(base_pred, y);
            vector<double> hess = get_hess(base_pred, y);

            XGBoostTree boosting_tree = XGBoostTree();
            boosting_tree.fit(x, y, grad, hess, subsample_cols,
                              min_child_weight, lam, gamma,
                              eps, min_leaf, depth, use_ispure);
            vector<double> pred_temp = boosting_tree.predict(x);
            for (int j = 0; j < row_count; j++)
                base_pred[j] += learning_rate * pred_temp[j];

            estimators[i] = boosting_tree;
        }
    }

    vector<double> predict_raw(vector<vector<double>> X)
    {
        int row_count = X.size();
        vector<double> y_pred;
        copy(init_pred.begin(), init_pred.end(), back_inserter(y_pred));
        int estimators_num = estimators.size();
        for (int i = 0; i < estimators_num; i++)
        {
            vector<double> y_pred_temp = estimators[i].predict(X);
            for (int j = 0; j < row_count; j++)
                y_pred[j] += learning_rate * y_pred_temp[j];
        }

        return y_pred;
    }
};

struct XGBoostClassifier : public XGBoostBase
{
    using XGBoostBase::XGBoostBase;

    vector<double> get_grad(vector<double> y_pred, vector<double> y)
    {
        int element_num = y_pred.size();
        vector<double> grad(element_num);
        for (int i = 0; i < element_num; i++)
            grad[i] = sigmoid(y_pred[i]) - y[i];
        return grad;
    }

    vector<double> get_hess(vector<double> y_pred, vector<double> y)
    {
        int element_num = y_pred.size();
        vector<double> hess(element_num);
        for (int i = 0; i < element_num; i++)
        {
            double temp_proba = sigmoid(y_pred[i]);
            hess[i] = temp_proba * (1 - temp_proba);
        }
        return hess;
    }

    vector<double> get_init_pred(vector<vector<double>> x, vector<double> y)
    {
        vector<double> init_pred(x.size(), 1);
        return init_pred;
    }

    vector<double> predict_proba(vector<vector<double>> x)
    {
        vector<double> raw_score = predict_raw(x);
        int row_count = x.size();
        vector<double> predicted_probas(row_count);
        for (int i = 0; i < row_count; i++)
            predicted_probas[i] = sigmoid(raw_score[i]);
        return predicted_probas;
    }
};