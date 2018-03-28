from keras.models import Model, load_model
from keras.layers import Input, Dense, Dropout, Flatten, Activation, Reshape
from keras.layers.convolutional import Conv2D, ZeroPadding2D
from keras.layers.pooling import MaxPooling2D, AveragePooling2D
from keras.optimizers import SGD, Adam, Adadelta
from keras import utils
import pandas as pd
import numpy as np
import argparse
import win_unicode_console
import time
win_unicode_console.enable()
DATA_PATH = r"C:\ws\push\LearnPractice\ML_LYH\HW3\train_data"
PATIENCE = 15


def data_load(train_path):
    x_np_file = open(train_path+r'\x.npy', 'rb')
    y_np_file = open(train_path+r'\y.npy', 'rb')
    x = np.load(x_np_file)
    y = np.load(y_np_file)
    x_np_file.close()
    y_np_file.close()

    x_train = x[:int(x.shape[0]*0.7)]
    x_vaild = x[int(x.shape[0]*0.7)+1:]
    y_train = y[:int(y.shape[0]*0.7)]
    y_vaild = y[int(y.shape[0]*0.7)+1:]

    #y_train = utils.to_categorical(y_train, 7)
    #y_vaild = utils.to_categorical(y_vaild, 7)
    print((x_train.shape, y_train.shape), (x_vaild.shape, y_vaild.shape))
    return (x_train, y_train), (x_vaild, y_vaild)


def build_model():
    input_img = Input(shape=(48, 48, 1))

    block1 = Conv2D(64, (5, 5), padding='valid', activation='relu')(input_img)
    block1 = ZeroPadding2D(padding=(2, 2), data_format='channels_last')(block1)
    block1 = MaxPooling2D(pool_size=(5, 5), strides=(2, 2))(block1)
    block1 = ZeroPadding2D(padding=(1, 1), data_format='channels_last')(block1)

    block2 = Conv2D(64, (3, 3), activation='relu')(block1)
    block2 = ZeroPadding2D(padding=(1, 1), data_format='channels_last')(block2)

    block3 = Conv2D(64, (3, 3), activation='relu')(block2)
    block3 = AveragePooling2D(pool_size=(3, 3), strides=(2, 2))(block3)
    block3 = ZeroPadding2D(padding=(1, 1), data_format='channels_last')(block3)

    block4 = Conv2D(128, (3, 3), activation='relu')(block3)
    block4 = ZeroPadding2D(padding=(1, 1), data_format='channels_last')(block4)

    block5 = Conv2D(128, (3, 3), activation='relu')(block4)
    block5 = ZeroPadding2D(padding=(1, 1), data_format='channels_last')(block5)
    block5 = AveragePooling2D(pool_size=(3, 3), strides=(2, 2))(block5)
    block5 = Flatten()(block5)

    fc1 = Dense(1024, activation='relu')(block5)
    fc1 = Dropout(0.5)(fc1)

    fc2 = Dense(1024, activation='relu')(fc1)
    fc2 = Dropout(0.5)(fc2)

    predict = Dense(7)(fc2)
    predict = Activation('softmax')(predict)
    model = Model(inputs=input_img, outputs=predict)

    # opt = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
    # opt = Adam(lr=1e-3)
    opt = Adadelta(lr=0.1, rho=0.95, epsilon=1e-08)
    model.compile(loss='categorical_crossentropy',
                  optimizer=opt, metrics=['accuracy'])
    model.summary()
    return model


def train(batch_size, num_epoch, pretrain,
          save_every, train_pixels, train_labels,
          val_pixels, val_labels, model_name=None):

    if pretrain == False:
        model = build_model()
    else:
        model = load_model(model_name)

    '''
    "1 Epoch" means you have been looked all of the training data once already.
    Batch size B means you look B instances at once when updating your parameter.
    Thus, given 320 instances, batch size 32, you need 10 iterations in 1 epoch.
    '''

    num_instances = len(train_labels)

    iter_per_epoch = int(num_instances / batch_size) + 1
    batch_cutoff = [0]
    for i in range(iter_per_epoch - 1):
        batch_cutoff.append(batch_size * (i+1))
    batch_cutoff.append(num_instances)

    total_start_t = time.time()
    best_metrics = 0.0
    early_stop_counter = 0
    for e in range(num_epoch):
        # shuffle data in every epoch
        rand_idxs = np.random.permutation(num_instances)
        print('#######')
        print('Epoch ' + str(e+1))
        print('#######')
        start_t = time.time()

        for i in range(iter_per_epoch):
            if i % 100 == 0:
                print('Iteration ' + str(i+1))
            X_batch = []
            Y_batch = []
            ''' fill data into each batch '''
            for n in range(batch_cutoff[i], batch_cutoff[i+1]):
                X_batch.append(train_pixels[rand_idxs[n]])
                Y_batch.append(np.zeros((7, ), dtype=np.float))
                Y_batch[-1][int(train_labels[rand_idxs[n]])] = 1.
            ''' use these batch data to train your model '''
            model.train_on_batch(np.asarray(X_batch), np.asarray(Y_batch))

        '''
        The above process is one epoch, and then we can check the performance now.
        '''
        loss_and_metrics = model.evaluate(val_pixels, val_labels, batch_size)
        print('\nloss & metrics:')
        print(loss_and_metrics)

        '''
        early stop is a mechanism to prevent your model from overfitting
        '''
        if loss_and_metrics[1] >= best_metrics:
            best_metrics = loss_and_metrics[1]
            print("save best score!! "+str(loss_and_metrics[1]))
            early_stop_counter = 0
        else:
            early_stop_counter += 1

        '''
        Sample code to write result :

        if e == e:
            val_proba = model.predict(val_pixels)
            val_classes = val_proba.argmax(axis=-1)


            with open('result/simple%s.csv' % str(e), 'w') as f:
                f.write('acc = %s\n' % str(lossandmetrics[1]))
                f.write('id,label')
                for i in range(len(valclasses)):
                    f.write('\n' + str(i) + ',' + str(valclasses[i]))
        '''

        print('Elapsed time in epoch ' + str(e+1) +
              ': ' + str(time.time() - start_t))

        if (e+1) % save_every == 0:
            model.save('ec_model/model-%d.h5' % (e+1))
            print('Saved model %s!' % str(e+1))

        if early_stop_counter >= PATIENCE:
            print('Stop by early stopping')
            print('Best score: '+str(best_metrics))
            break

    print('Elapsed time in total: ' + str(time.time() - total_start_t))

    return


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', type=int, default=100)
    parser.add_argument('--batch', type=int, default=63)
    parser.add_argument('--pretrain', type=bool, default=True)
    parser.add_argument('--save_every', type=int, default=1)
    parser.add_argument('--model_name', type=str,
                        default='ec_model/model-9.h5')
    args = parser.parse_args()

    (x_train, y_train), (x_vaild, y_vaild) = data_load(DATA_PATH)

    y_vaild_tmp = []
    for i in range(len(y_vaild)):
        onehot = np.zeros((7, ), dtype=np.float)
        onehot[int(y_vaild[i])] = 1.
        y_vaild_tmp.append(onehot)
    y_vaild = np.array(y_vaild_tmp)
    train(args.batch, args.epoch, args.pretrain, args.save_every,
          x_train, y_train, x_vaild, y_vaild,
          args.model_name)
    return


if __name__ == '__main__':
    main()
