''' MNIST parameter optimization Using Keras and Tensorflow
How to select best parameter for model using MNITS datasets
this code is tested only for one epoc and trained on CPU
one can try multiple combination of parameters as per computing power he has'''

# Imports
from __future__ import print_function
import keras
import time
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D, Activation
from keras import backend as K
from keras.callbacks import TensorBoard

batch_size = 128
num_classes = 10
# input mnist image dimensions
img_rows, img_cols = 28, 28

# load mnist data
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
# normalize data
x_train /= 255
x_test /= 255

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

# Define parameters to run model on
# here one can re cofigure other parameters as well
dense_layers = [0, 1, 2]
layer_sizes = [32, 64, 128]
conv_layers = [1, 2, 3]

for dense_layer in dense_layers:
    for layer_size in layer_sizes:
        for conv_layer in conv_layers:
            NAME = "{}-conv-{}-nodes-{}-dense-{}".format(conv_layer,
                                                         layer_size, dense_layer, int(time.time()))
            print(NAME)
            # define models and layer in loop
            model = Sequential()
            model.add(Conv2D(layer_size, kernel_size=(3, 3),
                             activation='relu',
                             input_shape=input_shape))
            model.add(MaxPooling2D(pool_size=(2, 2)))

            for l in range(conv_layer-1):
                model.add(Conv2D(layer_size, (3, 3)))
                model.add(Activation('relu'))
                model.add(MaxPooling2D(pool_size=(2, 2)))
            model.add(Flatten())

            for _ in range(dense_layer):
                model.add(Dense(layer_size))
                model.add(Activation('relu'))
                model.add(Dense(num_classes, activation='softmax'))

                # add tensorbord callbacks to visulize comparisons
                tensorboard = TensorBoard(log_dir="logs/{}".format(NAME))

                model.compile(loss=keras.losses.categorical_crossentropy,
                              optimizer=keras.optimizers.Adadelta(),
                              metrics=['accuracy'])

                model.fit(x_train, y_train,
                          batch_size=batch_size,
                          epochs=1,
                          verbose=1,
                          validation_data=(x_test, y_test), callbacks=[tensorboard])

                score = model.evaluate(x_test, y_test, verbose=0)
                # print test accuracy an loss after every combination results
                print('Test loss:', score[0])
                print('Test accuracy:', score[1])
