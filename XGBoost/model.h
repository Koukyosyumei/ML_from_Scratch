#include <vector>
#include <base.h>
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
};