import sys
from math import floor, log

import numpy as np
import pandas as pd

'''
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
'''
'''
a = [[1,2,3],[6,5,4]]
print(a)
print()
a = np.array(a)
print (a)
print()
a = sum(a)
print(a)
print()
a = np.tile(a ,(3,2))
print(a)
print()
y = [[[1.4  ],[2.55252],[3.1321]]]
print (np.around(y))
'''