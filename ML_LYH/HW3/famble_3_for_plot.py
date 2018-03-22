import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from keras.models import load_model
from utils import *
from keras.datasets import mnist
import keras.backend as K
import itertools


def read_dataset():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    if K.image_data_format() == 'channels_first':
        x_train = x_train.reshape(x_train.shape[0], 1, 28, 28)
        x_test = x_test.reshape(x_test.shape[0], 1, 28, 28)
    else:
        x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
        x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)

    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    return x_test, y_test

model_path = 'model/model-1.h5'
model = load_model(model_path)
np.set_printoptions(precision=2)
dev_feats, te_labels = read_dataset()
predictions = model.predict(dev_feats)
predictions = predictions.argmax(axis=-1)
conf_mat = confusion_matrix(te_labels, predictions)

conf_mat = conf_mat.astype('float') / conf_mat.sum(axis=1)[:, np.newaxis]

plt.figure()
plt.imshow(conf_mat,interpolation='nearest', cmap=plt.cm.jet)
plt.colorbar()
tick_marks = np.arange(10)
plt.xticks(tick_marks, ['0','1','2','3','4','5','6','7','8','9'], rotation=45)
plt.yticks(tick_marks, ['0','1','2','3','4','5','6','7','8','9'])

thresh = conf_mat.max() / 2.
for i, j in itertools.product(range(conf_mat.shape[0]), range(conf_mat.shape[1])):
    plt.text(j, i, '{:.2f}'.format(conf_mat[i, j]), horizontalalignment="center",
                color="white" if conf_mat[i, j] > thresh else "black")
plt.tight_layout()
plt.ylabel('True label')
plt.xlabel('Predicted label')


plt.show()
