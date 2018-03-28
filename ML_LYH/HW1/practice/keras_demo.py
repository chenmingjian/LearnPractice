import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers import Conv2D, MaxPooling2D, Flatten
from keras.optimizers import SGD, Adam, RMSprop,Adagrad
from keras.utils import np_utils
from keras.datasets import mnist
def load_data():
	(x_train, y_train), (x_test, y_test) = mnist.load_data()
	number = 10000
	x_train = x_train[0:number]
	y_train = y_train[0:number]
	x_train = x_train.reshape(number, 28*28)
	x_test = x_test.reshape(x_test.shape[0], 28*28)
	x_train = x_train.astype('float32')
	x_test = x_test.astype('float32')

	y_train = np_utils.to_categorical(y_train, 10)
	y_test = np_utils.to_categorical(y_test, 10)
	x_train = x_train
	x_test = x_test

	x_train = x_train/255
	x_test = x_test/255
	return (x_train, y_train), (x_test, y_test)

(x_train, y_train), (x_test, y_test) = load_data()
model = Sequential()
model.add(Dense(input_dim = 28*28, units=666, activation='relu'))
for i in range(3):
	model.add(Dense(units=666,activation='relu'))

model.add(Dense(units=10, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=0.001), metrics=['accuracy'])#为什么这里用Adam、Adagrad表现不好？lr给
																		#  的太大了。。。那为什么SGD就能很好的适应大LR
					#为什么子mse（均方误差）在分类问题上表现的不好？？？？？？？？？？？？？？在讲逻辑回归时讲过这件事了？
					#逻辑回归用的是什么作为loss Function? cross entropy?
model.fit(x_train, y_train, batch_size=100, epochs=20)

result = model.evaluate(x_train,y_train,batch_size=10000)
print("\nTrain acc:" + "%s\n"%result[1])
result = model.evaluate(x_test,y_test,batch_size=10000)
print("\nTest acc:" + "%s\n"%result[1])


