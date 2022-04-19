#include <iostream>
#include <model.h>
using namespace std;

int main()
{
    int num_row, num_col;
    cin >> num_row >> num_col;
    vector<vector<double>> X(num_row, vector<double>(num_col));
    vector<double> y(num_row);

    for (int i = 0; i < num_col; i++)
    {
        for (int j = 0; j < num_row; j++)
        {
            double temp;
            cin >> temp;
            X[j][i] = temp;
        }
    }

    for (int j = 0; j < num_row; j++)
    {
        double temp;
        cin >> temp;
        y[j] = temp;
    }

    // XGBoostClassifier clf = XGBoostClassifier()
}