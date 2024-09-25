import math

import pandas as pd
from collections import Counter

def Entropy(DataList):
    counts = len(DataList)  # 总数量
    counter = Counter(DataList)  # 每个变量出现的次数
    prob = {i[0]: i[1] / counts for i in counter.items()}  # 每个变量出现的比例 p
    H = - sum([i[1] * math.log2(i[1]) for i in prob.items()])  # 计算熵
    return H

if __name__ == "__main__":
    data = {
        'X': [0.46875, 0.28125, 0.25,0],
        'Y': [0.5, 0.25, 0.125, 0.125]
    }
    df = pd.DataFrame(data)

    X = df['X']
    Y = df['Y']

    XY = list(zip(X,Y))
    HX = Entropy(X)
    HY = Entropy(Y)
    HXY = Entropy(XY)
    HY_X = HXY - HX
    HX_Y = HXY - HY
    IXY = HX - HX_Y
    print("HX=",HX)
    print("HY=",HY)
    print("H(X,Y)=",HXY)
    print("H(X|Y)=",HX_Y)
    print("H(Y|X)=",HY_X)
    print("I(X,Y)=",IXY)
