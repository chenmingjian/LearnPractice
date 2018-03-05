from keras.models import Model
from keras.layers import Dense, Input

inputs = Input(shape=(784,))

x = Dense(64, activation='relu')(inputs)
x = Dense(64, activation='relu')(x)
predictions = Dense(10, activation='softmax')(x)

model = Model(inputs= inputs, outputs= predictions)
model.compile(optimizer = 'rmsprop',loss='categorical_crossentropy',metrics=['accuracy'])


