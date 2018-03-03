import sys
import pandas as pd
import numpy as np

x_train = pd.read_csv("c:/ws/train.csv", header=None)
y_train = pd.read_csv("c:/ws/test.csv")

def get_feature_name():
    row = []
    lens = 0
    for i in range(14):
        if (x_train[i][1].isdigit()):
            row.append(x_train[i][0])
        else:
            for  j in x_train[i]:
                if j != x_train[i][0]: 
                    row.append(j)
    return list(set(row))

fn = get_feature_name()
print(fn)