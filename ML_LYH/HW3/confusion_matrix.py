from keras.models import load_model
from sklearn.metrics import confusion_matrix
from utils import *
from keras.datasets import mnist
import itertools
import numpy as np
import matplotlib.pyplot as plt
import keras.backend as K


def plot_confusion_matrix(cm, classes, title='Confusion matrix', cmap=plt.cm.jet):
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, '{:.2f}'.format(cm[i, j]), horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


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


def main():
    model_path = 'model/model-1.h5'
    model = load_model(model_path)
    np.set_printoptions(precision=2)
    dev_feats, te_labels = read_dataset()
    predictions = model.predict(dev_feats)
    predictions = predictions.argmax(axis=-1)
    conf_mat = confusion_matrix(te_labels, predictions)

    plt.figure()
    plot_confusion_matrix(conf_mat, classes=["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"])
    plt.show()


if __name__ == '__main__':
    main()
