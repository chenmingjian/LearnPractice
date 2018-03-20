import keras
from keras.datasets import mnist
from keras.models import Model
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dropout, Input, Dense
from keras.utils import plot_model
from keras import backend as K
from sklearn.metrics import confusion_matrix
from keras.models import load_model
from utils import *
import itertools
import numpy as np
import matplotlib.pyplot as plt
import os
import argparse
import time
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


def build_model(input_shape):
    inputs = Input(shape=input_shape)

    block1 = Conv2D(32, kernel_size=(3, 3), activation='relu')(inputs)
    block1 = MaxPooling2D(pool_size=(2, 2))(block1)

    block2 = Conv2D(64, (3, 3), activation='relu')(block1)
    block2 = MaxPooling2D(pool_size=(2, 2))(block2)

    Drop = Dropout(0.25)(block2)
    out = Flatten()(Drop)
    block3 = Dense(128, activation='relu')(out)
    Drop = Dropout(0.5)(block3)
    out = Dense(10, activation='softmax')(Drop)
    model = Model(inputs=inputs, outputs=out)
    return model


def train(input_shape,batch_size, num_epoch, pretrain,
          save_every, x_train, y_train, model_name=None):
    if pretrain == False:
        model = build_model(input_shape)
        '''
        model.compile(optimizer='adam',
                        loss='categorical_crossentropy',
                        metrics=['accuracy'])
        model.fit(x_train, y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=1,
                    validation_data=(x_test, y_test))
        '''
    else:
        model = load_model(model_name)
        '''
        model.load_weights('my_model_weights')
        model.compile(optimizer='adam',
                        loss='categorical_crossentropy',
                        metrics=['accuracy'])
        '''
    num_instances = len(y_train)
    iter_per_epoch = int(num_instances / batch_size) + 1

    batch_cutoff = [0]
    for i in range(iter_per_epoch - 1):
        batch_cutoff.append(batch_size * (i+1))
    batch_cutoff.append(num_instances)

    total_start = time.time()
    best_metrics = 0.0
    early_step_counter = 0
    for e in range(num_epoch):
        rand_dixs = np.random.permutation(num_instances)
        print('#######')
        print('Epoch' + str(e+1))
        print('#######')
        start_t = time.time()

        for i in range(iter_per_epoch):
            if i % 50 == 0:
                print("Iteration" + str(i+1))
            X_batch = []
            Y_batch = []
            # fill data into each batch
            for n in range(batch_cutoff[i], batch_cutoff[i+1]):
                X_batch.append(x_train[rand_dixs[n]])
                Y_batch.append(np.zeros((7, ), dtype=np.float))
                X_batch[-1] = np.fromstring(X_batch[-1],
                                            dtype=float, sep=' ').reshape((28, 28, 1))
                Y_batch[-1][int(train_labels[rand_idxs[n]])] = 1.
            model.train_on_batch(np.asarray(X_batch), np.asarray(Y_batch))
        loss_and_metrics = model.evaluate(val_pixels, val_labels, batch_size)
        print('\nloss & metrics:')
        print(loss_and_metrics)
        if loss_and_metrics[1] >= best_metrics:
            best_metrics = loss_and_metrics[1]
            print("save best score!! "+str(loss_and_metrics[1]))
            early_stop_counter = 0
        else:
            early_stop_counter += 1
        print('Elapsed time in epoch ' + str(e+1) +
              ': ' + str(time.time() - startt))

        if (e+1) % saveevery == 0:
            model.save('model/model-%d.h5' % (e+1))
            print('Saved model %s!' % str(e+1))

        if earlystopcounter >= PATIENCE:
            print('Stop by early stopping')
            print('Best score: '+str(best_metrics))
            break

    print('Elapsed time in total: ' + str(time.time() - total_start_t))


def main():
    parser = argparse.ArgumentParser(prog='train.py')
    parser.add_argument('--epoch', type=int, default=12)
    parser.add_argument('--batch', type=int, default=64)
    parser.add_argument('--pretrain', type=bool, default=False)
    parser.add_argument('--save_every', type=int, default=1)
    parser.add_argument('--model_name', type=str, default='model/model-1')
    args = parser.parse_args()

    # load training data
    img_rows, img_cols = 28, 28
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    if K.image_data_format() == 'channels_first':
        x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
        x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
        input_shape = (1, img_rows, img_cols)
    else:
        x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
        x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
        input_shape = (img_rows, img_cols, 1)
    
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255
    print('x_train shape:', x_train.shape)
    print(x_train.shape[0], 'train samples')
    print(x_test.shape[0], 'test samples')
    y_train = keras.utils.to_categorical(y_train, 10)
    y_test = keras.utils.to_categorical(y_test, 10)

    # train
    train(input_shape, args.batch, args.epoch, args.pretrain,
          args.save_every, x_train, y_train)


'''
score = model.evaluate(x_test, y_test, batch_size=32,
                       verbose=1, sample_weight=None)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

model.save_weights('my_model_weights')
plot_model(model, to_file='model.png')
print(model.summary())
'''
if __name__ == '__main__':
    main()
