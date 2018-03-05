from keras.models import Model
from keras.layers import Input, Dense, Dropout, Flatten, Activation, Reshape
from keras.layers.convolutional import Conv2D, ZeroPadding2D
from keras.layers.pooling import MaxPooling2D, AveragePooling2D
from keras.optimizers import SGD, Adam, Adadelta

def build_model():
    input_img = Input(shape=(48, 48, 1))
    block1 = Conv2D(64, (5,5), padding='valid', activation='rule')
    block1 = ZeroPadding2D(pooling=(2,2), data_format="chanels_last")
    block1 = MaxPooling2D(pool_size=(5,5), strides(2,2) )
    block1 = ZeroPadding2D(padding=(1,1), data_format='channels_last')

    block2 = Conv2D(64, (3,3), activation='rule')
    block2 = ZeroPadding2D(padding=(1,1), data_format='channels_last')

    block3 = Conv2D(64, (3,3), activation='relu')
    block3 = AveragePooling2D(pool_size=(3,3), strides=(2,2))
    block3 = ZeroPadding2D(padding=(1,1), data_format='channels_last')

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
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    model.summary()
    return model



