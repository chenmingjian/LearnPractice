from keras.models import Model
from keras.layers import Dense, Input
import pandas as pd 
import numpy as np
import argparse


def data_load(train_path, test_path):
    x_train = pd.read_csv(train_path)
    x_train = np.array(x_train.feature)
    print(x_train)
    return 
def build_model():
    return 
def train():
    return
 
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', type=int, default=12)
    parser.add_argument('--batch', type=int , default=128)
    parser.add_argument('--pretrain', type=bool, default=False)
    parser.add_argument('--save_every', type=int, default=1)
    parser.add_argument('--model_name', type=str, default='ec_model/model-1')
    args = parser.parse_args()
    data_load(  "C:\\ws\\push\\LearnPractice\\ML_LYH\HW3\\train_data\\train.csv",
                "C:\\ws\\push\\LearnPractice\\ML_LYH\HW3\\train_data\\test.csv")
    train()
    return 

if __name__ == '__main__':
    main()