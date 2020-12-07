import numpy as np
from numpy.random import choice
import stats
from decision_tree import branch, Vertex, predict
from utils import sq_loss, mis_math, gini, entropy


def rf_branch(x, y, f, S, m):
    """Summary:
        左右の枝内のyの分散が最小になるように、
        閾値を設定する

    args:
        x: 説明変数のnp.array
        y: 目的変数のnp.array
        f: 損失関数
        S: xに含まれる各データのindexのlist
        m: 使用する特徴量の数、x.shape[1]以下

    return:
        info:  {"i": i,
                "j": j, x[i, j]がこのノードの閾値となる
                "left": 左側の枝が含むデータのindexのリスト,
                "right": 右側の枝が含むデータのindexのリスト,
                "score": left_score + right_score,
                "left_score": 左側の枝のスコア,
                "right_score": 右側の枝のスコア}

    usage:

    """
    n = len(S)
    p = x.shape[1]
    best_score = float("inf")
    if n == 0:
        return None
    if m < p:
        T = choice(list(range(p)), m)
    else:
        T = list(range(p))

    for j in T:
        for i in S:
            left, right = [], []
            for k in S:
                if x[k, j] < x[i, j]:
                    left.append(k)
                else:
                    right.append(k)

            L, R = f(y[left]), f(y[right])
            score = L+R
            if score < best_score:
                best_score = score
                info = {"i": i,
                        "j": j,
                        "left": left,
                        "right": right,
                        "score": best_score,
                        "left_score": L,
                        "right_score": R}
    return info


def rf_tree(x, y, m=None, f="sq_loss", alpha=0, n_min=10, branch=branch):
    """ Summary
    決定木を一本作って、返す関数

    args:
        x: 説明変数のnp.array
        y: 目的変数のnp.array
        m: 説明変数の個数, x.shape[1]以下
        f: 損失関数
        alpha:正則可項
        n_min:各木の所属するデータの最低個数
        branch: 枝を発生させる関数, branchなら普通の決定木、rf_branch
                ならランダムフォレスト用の決定木
    returns: Vertex

    usage:
    """
    if m is None:
        m = x.shape[1]

    if f == "sq_loss":
        g = sq_loss
    elif f == "mis_math":
        g = mis_math
    elif f == "gini":
        g = gini
    else:
        g = entropy

    n = len(y)
    stack = []
    # 木の初期化 rootのparentは-1にしておく
    stack.append(Vertex(parent=-1, sets=list(range(n)), score=g(y)))
    vertexs = []
    k = 0  # pushされたvertexのID

    while(len(stack) > 0):
        node = stack.pop()
        # どこで分けるのが最良なのかを計算
        res = branch(x, y, g, node.sets, m)
        # 分割して際の改善が規定値以下の場合 or nodeに所属しているデータの個数
        # が既定の個数以下の場合 or 片方の枝に全てのデータが所属している場合
        # jは端点の場合-1 そうでない場合はその説明変数のcolumn_index
        if(((node.score-res["score"]) < alpha) or len(node.sets) < n_min or
           len(res["left"]) == 0 or len(res["right"]) == 0):
            # 端点である
            vertexs.append(Vertex(parent=node.parent, j=-1, sets=node.sets))
        else:
            vertexs.append(Vertex(parent=node.parent, sets=node.sets,
                                  th=x[res["i"], res["j"]], j=res["j"]))
            stack.append(
                Vertex(parent=k, sets=res["right"], score=res["right_score"]))
            stack.append(
                Vertex(parent=k, sets=res["left"], score=res["left_score"]))
        k = k+1

    r = len(vertexs)
    assert r != 0, "there is no vertexs"

    for h in range(r):
        temp = vertexs[h]
        temp.left, temp.right = None, None
        vertexs[h] = temp
    for h in range(r-1, 0, -1):  # rから2まで
        pa = vertexs[h].parent
        temp = vertexs[pa]
        if temp.right is None:
            temp.right = h
        else:
            temp.left = h
        vertexs[pa] = temp

    g = np.mean if f == "sq_loss" else lambda x: stats.mode(x)[0][0]

    # 端点に、その値(center)を設定
    for h in range(r):
        temp = vertexs[h]
        if temp.j == -1:
            temp.center = g(y[temp.sets])
            vertexs[h] = temp

    return(vertexs)


def random_forest(x, y, m=None, f="mis_math", alpha=0,
                  n_min=10, branch=branch, B=100):
    """ Summary

    Args:
            x: 説明変数のnp.array
            y: 目的変数のnp.array
            m: 説明変数の個数, x.shape[1]以下
            f: 損失関数
            alpha:正則可項
            n_min:各木の所属するデータの最低個数
            branch: 枝を発生させる関数, branchなら普通の決定木、rf_branch
                    ならランダムフォレスト用の決定木
            B: 発生させる決定木の本数
    Returns:
            rf: [Vertex], 各決定木のリスト
    Usage:
    """
    rf = []
    N = x.shape[0]
    for b in range(B):
        index = choice(list(range(N)), N)
        vertexs = rf_tree(x[index, :], y[index], m=m,
                          f=f, alpha=alpha, n_min=n_min, branch=branch)
        rf.append(vertexs)
    return rf


def rf_predict(u, random_forest):
    predicts = []
    for forest in random_forest:
        predicts.append(predict(u, forest))
    return stats.mode(predicts)[0][0]
