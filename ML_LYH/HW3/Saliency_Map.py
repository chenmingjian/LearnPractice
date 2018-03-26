import os
import argparse
from keras.models import load_model
from termcolor import colored, cprint
import keras.backend as K
from utils import *
import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import mnist


def deprocess_image(x):
    return x


base_dir = os.getcwd()
img_dir = os.path.join(base_dir, 'image')
if not os.path.exists(img_dir):
    os.makedirs(img_dir)
cmap_dir = os.path.join(img_dir, 'cmap')
if not os.path.exists(cmap_dir):
    os.makedirs(cmap_dir)
partial_see_dir = os.path.join(img_dir, 'partial_see')
if not os.path.exists(partial_see_dir):
    os.makedirs(partial_see_dir)
model_dir = os.path.join(base_dir, 'model')


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
    parser = argparse.ArgumentParser(prog='Saliency_Map.py',
                                     description='ML-Assignment3 visualize attention heat map.')
    parser.add_argument('--epoch', type=int, metavar='<#epoch>', default=12)
    args = parser.parse_args()
    model_name = "model-%s.h5" % str(args.epoch)
    model_path = os.path.join(model_dir, model_name)
    model = load_model(model_path)
    print("Loaded model from {}".format(model_name))

    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    private_pixels = [i.reshape(1, 28, 28, 1) for i in x_test]

    input_img = model.input
    img_ids = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    heatmap = private_pixels[0].reshape(28, 28)

    for idx in img_ids:
        val_proba = model.predict(private_pixels[idx])
        pred = val_proba.argmax(axis=-1)
        target = K.mean(model.output[:, pred])
        grads = K.gradients(target, input_img)[0]
        fn = K.function([input_img, K.learning_phase()], [grads])
        heatmap = private_pixels[0].reshape(28, 28)
        """
        Implement your heatmap processing here!
        hint: Do some normalization or smoothening on grads
        """

        thres = 0.5
        heatmap = private_pixels[idx].reshape(28, 28)
        see = private_pixels[idx].reshape(28, 28)
        # for i in range(28):
        # for j in range(28):
        # print heatmap[i][j]

        see[np.where(heatmap <= thres)] = np.mean(see)

        plt.figure()
        plt.imshow(heatmap, cmap=plt.cm.jet)
        plt.colorbar()
        plt.tight_layout()
        fig = plt.gcf()
        plt.draw()
        test_dir = os.path.join(cmap_dir, 'test')
        if not os.path.exists(test_dir):
            os.makedirs(test_dir)
        fig.savefig(os.path.join(test_dir, '{}.png'.format(idx)), dpi=100)

        plt.figure()
        plt.imshow(see, cmap='gray')
        plt.colorbar()
        plt.tight_layout()
        fig = plt.gcf()
        plt.draw()
        test_dir = os.path.join(partial_see_dir, 'test')
        if not os.path.exists(test_dir):
            os.makedirs(test_dir)
        fig.savefig(os.path.join(test_dir, '{}.png'.format(idx)), dpi=100)


if __name__ == "__main__":
    main()
