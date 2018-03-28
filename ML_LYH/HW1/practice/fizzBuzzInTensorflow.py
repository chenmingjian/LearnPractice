import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers import Conv2D, MaxPooling2D,Flatten
from keras.optimizers import SGD, Adam, RMSprop, Adagrad
from keras.utils import np_utils

NUM_DIGISTS = 10

def binary_encode (i, num_digists):
	return np.array([i >> d & 1 for d in range(num_digists)])

def fizz_buzz_encode(i):
	if i % 15 == 0 : return np.array([0,0,0,1])
	elif i % 5 == 0: return np.array([0,0,1,0])
	elif i % 3 == 0: return np.array([0,1,0,0])
	else :return np.array([1,0,0,0])

def fizzbuzz(begin, end):
	return np.array([binary_encode(i, NUM_DIGISTS) for i in range(begin, end)]),np.array([fizz_buzz_encode(i) for i in range(begin, end)])

x_train, y_train = fizzbuzz(101,1000)
x_test, y_test = fizzbuzz(1,100)

def train():
	model = Sequential();
	model.add(Dense(input_dim=10, output_dim = 1000))
	model.add(Activation('relu'))
#	for i in range (0):
#		model.add(Dense(units=666, activation='relu'))
	model.add(Dense(output_dim=4))
	model.add(Activation('softmax'))

	model.compile(loss="categorical_crossentropy", optimizer='adam',metrics=['accuracy'] )

	model.fit(x_train, y_train, batch_size=20, nb_epoch=100)
	result = model.evaluate(x_train, y_train , batch_size=1000)
	print('\nTrain acc:' + '%s\n'%result[1])
	result = model.evaluate(x_test, y_test, batch_size = 1000)
	print('\nTrain acc:' + '%s\n'%result[1])

train()