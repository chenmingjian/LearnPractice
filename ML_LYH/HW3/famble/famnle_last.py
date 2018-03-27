import pandas as pd 
import numpy as np
import time
import pickle


def read_pickle():
    x_pkl_file = open(r'C:\ws\push\LearnPractice\ML_LYH\HW3\train_data\x.pk', 'rb')
    y_pkl_file = open(r'C:\ws\push\LearnPractice\ML_LYH\HW3\train_data\y.pk', 'rb')
    x = pickle.load(x_pkl_file)
    y = pickle.load(y_pkl_file)
    x_pkl_file.close()
    y_pkl_file.close()
    x = np.array(x)
    y = np.array(y)
    print(x.shape)
    return x,y

def read_np():
    x_np_file = open(r'C:\ws\push\LearnPractice\ML_LYH\HW3\train_data\x', 'rb')
    y_np_file = open(r'C:\ws\push\LearnPractice\ML_LYH\HW3\train_data\y', 'rb')
    x = np.load(x_np_file)
    y = np.load(y_np_file)
    x_np_file.close()
    y_np_file.close()
    print(x.shape)
    return x,y

def data_load(path=r'C:\ws\push\LearnPractice\ML_LYH\HW3\train_data\train.csv'):
    fx = open(r"C:\ws\push\LearnPractice\ML_LYH\HW3\train_data\x.npy", 'wb')
    fy =open(r"C:\ws\push\LearnPractice\ML_LYH\HW3\train_data\y.npy","wb")
    train_csv = pd.read_csv(path)
    x_train = []
    for i in train_csv.feature:
        x_train.append([num  for num in i.split(" ") ])
    x_train = np.array(x_train,dtype='float')
    x_train = x_train.reshape(x_train.shape[0], 48,48,1)
    y_train = np.array(train_csv.label, dtype='float')
    
    np.save(fx, x_train)
    np.save(fy, y_train)
    return x_train, y_train
    
def data_load_2(path=r'C:\ws\push\LearnPractice\ML_LYH\HW3\train_data\train.csv'):
    fx = open(r"C:\ws\push\LearnPractice\ML_LYH\HW3\train_data\x.npy", 'wb')
    fy =open(r"C:\ws\push\LearnPractice\ML_LYH\HW3\train_data\y.npy","wb")
    train_csv = pd.read_csv(path)
    x_train = np.array(train_csv.feature)
    y_train = np.array(train_csv.label)
    print(x_train.shape,y_train.shape)
    np.save(fx, x_train)
    np.save(fy, y_train)

start = time.time()
data_load(r'C:\ws\push\LearnPractice\ML_LYH\HW3\train_data\train.csv')
print(time.time() - start)


