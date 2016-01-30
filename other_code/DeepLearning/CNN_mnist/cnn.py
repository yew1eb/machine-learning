# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import print_function
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.advanced_activations import PReLU
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import SGD, Adadelta, Adagrad
from keras.utils import np_utils, generic_utils
from data import load_data

import matplotlib.pyplot as plt

'''
train the dataset of face 42000 image
learning rate : 0.0015
pathc size:100
dropout :0.25
l2 = 0.01
'''

def funcnn(LR,BS):
    data, label = load_data()
    label = np_utils.to_categorical(label, 10)
    
    model = Sequential()
    
    model.add(Convolution2D(4, 1, 5, 5, border_mode='valid')) 
    model.add(Activation('relu'))
    model.add(Dropout(0.25))
    
    model.add(Convolution2D(8,4, 3, 3, border_mode='valid'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(poolsize=(2, 2)))
    model.add(Dropout(0.25))
    
    model.add(Convolution2D(16, 8, 3, 3, border_mode='valid')) 
    model.add(Activation('relu'))
    model.add(MaxPooling2D(poolsize=(2, 2)))
    model.add(Dropout(0.25))
    
    model.add(Flatten())
    model.add(Dense(16*4*4, 256, init='normal'))
    model.add(Activation('tanh'))
    
    model.add(Dense(256, 10, init='normal'))
    model.add(Activation('softmax'))
    
    sgd = SGD(l2=0.001,lr=LR, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=sgd,class_mode="categorical")
    
    #checkpointer = ModelCheckpoint(filepath="weight.hdf5",verbose=1,save_best_only=True)
    #model.fit(data, label, batch_size=100,nb_epoch=10,shuffle=True,verbose=1,show_accuracy=True,validation_split=0.2,callbacks=[checkpointer])
    result = model.fit(data, label, batch_size=BS,nb_epoch=20,shuffle=True,verbose=1,show_accuracy=True,validation_split=0.2)
    #model.save_weights(weights,accuracy=False)
    
    # plot the result
    
    plt.figure
    plt.plot(result.epoch,result.history['acc'],label="acc")
    plt.plot(result.epoch,result.history['val_acc'],label="val_acc")
    plt.scatter(result.epoch,result.history['acc'],marker='*')
    plt.scatter(result.epoch,result.history['val_acc'])
    plt.legend(loc='under right')
    plt.show()
    
    plt.figure
    plt.plot(result.epoch,result.history['loss'],label="loss")
    plt.plot(result.epoch,result.history['val_loss'],label="val_loss")
    plt.scatter(result.epoch,result.history['loss'],marker='*')
    plt.scatter(result.epoch,result.history['val_loss'],marker='*')
    plt.legend(loc='upper right')
    plt.show()
    

