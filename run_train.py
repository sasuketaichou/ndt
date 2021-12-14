import os
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

import numpy as np
import tensorflow.keras as keras
from tensorflow.keras import optimizers
from create_dataset import load_dataset
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense, Flatten, MaxPooling2D
from tensorflow.keras.callbacks import LearningRateScheduler, TensorBoard

batch_size    = 128
epochs        = 500
num_classes   = 2

def build_model():
    model = Sequential()
    model.add(Conv2D(6, (5, 5), padding='valid', activation = 'relu', kernel_initializer='he_normal', input_shape=(64,64,3)))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))
    model.add(Conv2D(16, (5, 5), padding='valid', activation = 'relu', kernel_initializer='he_normal'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))
    model.add(Flatten())
    model.add(Dense(120, activation = 'relu', kernel_initializer='he_normal'))
    model.add(Dense(84, activation = 'relu', kernel_initializer='he_normal'))
    model.add(Dense(num_classes, activation = 'softmax', kernel_initializer='he_normal'))
    sgd = optimizers.SGD(lr=.0001, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
    return model

def scheduler(epoch):
    if epoch < 100:
        return 0.01
    if epoch < 150:
        return 0.005
    return 0.001

if __name__ == '__main__':

    # load data
    x_train,y_train,x_test,y_test = load_dataset((64,64))
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    
    #x_train /= 255.0
    #x_test /= 255.0
    # mean  = [125.307, 122.95, 113.865]
    # std   = [62.9932, 62.0887, 66.7048]
    # for i in range(3):
    #     x_train[:,:,:,i] = (x_train[:,:,:,i] - mean[i]) / std[i]
    #     x_test[:,:,:,i] = (x_test[:,:,:,i] - mean[i]) / std[i]

    # build network
    model = build_model()
    print(model.summary())

    # set callback
    tb_cb = TensorBoard(log_dir='./lenet_dp', histogram_freq=0)
    change_lr = LearningRateScheduler(scheduler)
    cbks = [change_lr,tb_cb]

    # start train 
    model.fit(x_train, y_train,batch_size=batch_size,epochs=epochs,callbacks=cbks,
                  validation_data=(x_test, y_test), shuffle=True)
    # save model
    model.save('lenet_xray.h5')

