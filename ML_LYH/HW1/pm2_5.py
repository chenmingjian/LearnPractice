#-*- coding:utf-8 –*-
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas import DataFrame
import math
'''
with open('train.csv', encoding='big5') as csv_training_data:
	reader = csv.reader(csv_training_data)
	dict_reader = csv.DictReader(csv_training_data)
	row = [row for row in dict_reader]
	print(row[1])
'''
'''
#数据处理
train = pd.read_csv('train.csv')
pm2_5 = train[train['test']=='PM2.5']
pm2_5.drop(['date','station','test'],axis=1,inplace=True)

x=[]
y=[]
for i in range(15):
	tmp_x = pm2_5.iloc[:,i:i+9]
	tmp_x.columns = np.array(range(9))
	tmp_y = pm2_5.iloc[:,i + 9]
	tmp_y.columns = np.array(range(1))
	x.append(tmp_x)
	y.append(tmp_y)

x = pd.concat(x) #合并数组，x变成dataframe
y = pd.concat(y)
x = np.array(x, float)
y = np.array(y, float)
np.save("x.npy", x)
np.save("y.npy", y)
'''
#比较简单可以使用closed form solution(封闭解)来验证model

x = np.load('x.npy')
y = np.load('y.npy')

x = np.concatenate((np.ones((x.shape[0],1)),x), axis=1)
w = np.zeros(len(x[0]))
lr = 1
iteration = 100000
s_grad = np.zeros(len(x[0]))
x_t = x.T
#train
for i in range(iteration):
	tmp = np.dot(x, w)
	loss = tmp - y
	grad = np.dot(x_t, loss)#这算的是啥？
	s_grad += grad**2
	ada = np.sqrt(s_grad)
	w = w - lr * grad/ada
np.save ('model.npy', w)

#有点不懂
model=np.load("model.npy")
test=pd.read_csv("test.csv")
t=test[test["test"]=="PM2.5"]
t.drop(["date","test"],axis=1,inplace=True)
t=np.array(t,float)
t=np.concatenate((np.ones((t.shape[0],1)),t), axis=1)
res=[]
res=np.dot(t,w)
print(res)