import keras
from keras.datasets import mnist
from keras.models import Model
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dropout, Input, Dense
from keras.utils import plot_model
from keras import backend as K
import os

batch_size = 128
num_classes = 10
epochs = 12

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

y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

inputs = Input(shape=input_shape)

block1 = Conv2D(32, kernel_size=(3, 3), activation='relu')(inputs)
block1 = MaxPooling2D(pool_size=(2, 2))(block1)

block2 = Conv2D(64, (3, 3), activation='relu')(block1)
block2 = MaxPooling2D(pool_size=(2, 2))(block2)

Drop = Dropout(0.25)(block2)

out = Flatten()(Drop)

block3 = Dense(128, activation='relu')(out)

Drop = Dropout(0.5)(block3)

out = Dense(num_classes, activation='softmax')(Drop)

model = Model(inputs=inputs, outputs=out)

if not os.path.exists('my_model_weights'):
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    model.fit(x_train, y_train)
else:
    model.load_weights('my_model_weights')
    model.compile(optimizer='adam',
                loss='categorical_crossentropy',
                metrics=['accuracy'])

score = model.evaluate(x_test, y_test, batch_size=32,
                       verbose=1, sample_weight=None)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

model.save_weights('my_model_weights')

plot_model(model, to_file='model.png')

print(model.summary())