#include <vector>
#include <numeric>
#include <limits>
#include <set>
#include <cmath>
using namespace std;

struct Node
{
    vector<vector<double>> x;
    vector<double> y, gradient, hessian, my_gradiet, my_hessian;
    vector<int> idxs;
    double subsample_cols, min_child_weight;
    double lam, gamma, eps;
    int min_leaf, depth;
    bool use_ispure;

    int row_count, col_count;
    vector<int> column_subsample;

    int var_idx;
    double val, score, split;
    Node *left;
    Node *right;

    Node() {}
    Node(vector<vector<double>> x_, vector<double> y_,
         vector<double> gradient_, vector<double> hessian_,
         vector<int> idxs_, double subsample_cols_, double min_child_weight_,
         double lam_, double gamma_, double eps_, int min_leaf_, int depth_, bool use_ispure_)
    {
        x = x_;
        y = y_;
        gradient = gradient_;
        hessian = hessian_;
        idxs = idxs_;
        subsample_cols = subsample_cols_;
        min_child_weight = min_child_weight_;
        lam = lam_;
        gamma = gamma_;
        eps = eps_;
        min_leaf = min_leaf_;
        depth = depth_;
        use_ispure = use_ispure_;

        row_count = idxs.size();
        col_count = x.at(0).size();

        column_subsample.resize(col_count);
        iota(column_subsample.begin(), column_subsample.end(), 0);

        my_gradiet.resize(row_count);
        my_hessian.resize(row_count);
        for (int i = 0; i < row_count; i++)
        {
            my_gradiet[i] = gradient[idxs[i]];
            my_hessian[i] = hessian[idxs[i]];
        }

        val = compute_weight();
        score = -1 * numeric_limits<double>::infinity();
        find_varsplit();
    }

    double compute_weight()
    {
        double sum_grad = 0;
        double sum_hess = 0;
        for (int i = 0; i < row_count; i++)
        {
            sum_grad += my_gradiet[i];
            sum_hess += my_hessian[i];
        }
        return -1 * (sum_grad / (sum_hess + lam));
    }

    double compute_gain(vector<int> left_idxs, vector<int> right_idxs)
    {
        double left_grad_sum = 0;
        double right_grad_sum = 0;
        double left_hess_sum = 0;
        double right_hess_sum = 0;

        for (int j = 0; j < left_idxs.size(); j++)
        {
            left_grad_sum += my_gradiet[left_idxs[j]];
            left_hess_sum += my_hessian[left_idxs[j]];
        }
        for (int j = 0; j < right_idxs.size(); j++)
        {
            right_grad_sum += my_gradiet[right_idxs[j]];
            right_hess_sum += my_hessian[right_idxs[j]];
        }

        double gain = 0.5 * ((left_grad_sum * left_grad_sum) / (left_hess_sum + lam) +
                             (right_grad_sum * right_grad_sum) / (right_hess_sum + lam) -
                             ((left_grad_sum + right_grad_sum) *
                              (left_grad_sum + right_grad_sum) / (left_hess_sum + right_hess_sum + lam))) -
                      gamma;
        return gain;
    }

    void find_varsplit()
    {
        find_greedy_split();

        if (is_leaf())
            return;
        else
        {
            vector<double> x_column = split_col(var_idx);
            vector<int> left_idxs, right_idxs;
            for (int i = 0; i < x_column.size(); i++)
            {
                if (x_column[i] <= split)
                    left_idxs.push_back(idxs[i]);
                else
                    right_idxs.push_back(idxs[i]);
            }

            left = new Node(x, y, gradient, hessian, left_idxs, subsample_cols, min_child_weight,
                            lam, gamma, eps, min_leaf, depth - 1, use_ispure);
            right = new Node(x, y, gradient, hessian, right_idxs, subsample_cols, min_child_weight,
                             lam, gamma, eps, min_leaf, depth - 1, use_ispure);
        }
    }

    void find_greedy_split()
    {
        for (int i = 0; i < column_subsample.size(); i++)
        {
            int var_idx_temp = column_subsample[i];
            vector<double> x_temp = split_col(var_idx_temp);

            for (int r = 0; r < row_count; r++)
            {
                vector<int> left_idxs, right_idxs;
                for (int j = 0; j < x_temp.size(); j++)
                {
                    if (x_temp[j] <= x_temp[r])
                        left_idxs.push_back(j);
                    else
                        right_idxs.push_back(j);
                }

                int left_size = left_idxs.size();
                int right_size = right_idxs.size();
                double hessian_left_sum = 0;
                double hessian_right_sum = 0;
                for (int j = 0; j < left_size; j++)
                    hessian_left_sum += my_hessian[left_idxs[j]];
                for (int j = 0; j < right_size; j++)
                    hessian_right_sum += my_hessian[right_idxs[j]];

                if (left_size < min_leaf ||
                    right_size < min_leaf ||
                    hessian_left_sum < min_child_weight ||
                    hessian_right_sum < min_child_weight)
                    continue;

                double curr_score = compute_gain(left_idxs, right_idxs);
                if (curr_score > score)
                {
                    var_idx = var_idx_temp;
                    score = curr_score;
                    split = x_temp[r];
                }
            }
        }
    }

    bool is_leaf()
    {
        return is_pure() || std::isinf(score) || depth <= 0;
    }

    bool is_pure()
    {
        vector<int> y_temp(row_count);
        for (int i = 0; i < row_count; i++)
            y_temp[i] = y[idxs[i]];
        set<int> y_set_temp(y_temp.begin(), y_temp.end());
        return use_ispure && y_set_temp.size() == 1;
    }

    vector<double> split_col(int column_idx)
    {
        vector<double> x_column(row_count);
        for (int i = 0; i < row_count; i++)
            x_column[i] = x[idxs[i]][column_idx];
        return x_column;
    }

    vector<double> predict(vector<vector<double>> x_new)
    {
        int x_new_size = x_new.size();
        vector<double> y_pred(x_new_size);
        for (int i = 0; i < x_new_size; i++)
        {
            auto temp_pred = predict_row(x[i]);
            y_pred[i] = temp_pred;
        }
        return y_pred;
    }

    double predict_row(vector<double> xi)
    {
        if (is_leaf())
            return val;
        else
        {
            Node *node;
            if (xi[var_idx] <= split)
                node = left;
            else
                node = right;
            return node->predict_row(xi);
        }
    }
};

struct XGBoostTree
{
    Node dtree;

    XGBoostTree() {}

    void fit(vector<vector<double>> x, vector<double> y,
             vector<double> gradient, vector<double> hessian, double subsample_cols,
             double min_child_weight, double lam, double gamma, double eps,
             int min_leaf, int depth, bool use_ispure)
    {
        vector<int> idxs(y.size());
        iota(idxs.begin(), idxs.end(), 0);
        dtree = Node(x, y, gradient, hessian, idxs, subsample_cols,
                     min_child_weight, lam, gamma, eps, min_leaf,
                     depth, use_ispure);
    }

    vector<double> predict(vector<vector<double>> X)
    {
        return dtree.predict(X);
    }
};