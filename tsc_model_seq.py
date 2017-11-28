# -*- coding: utf-8 -*-
import numpy as np
from keras.layers import Input, Add, Dense, Dropout, Activation, ZeroPadding2D
from keras.layers import BatchNormalization, Flatten, Conv2D, AveragePooling2D, MaxPooling2D
from keras.models import Model

from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot
from keras.utils import plot_model
from keras.initializers import glorot_uniform

import keras.backend as K
K.set_image_data_format('channels_last')
K.set_learning_phase(1)

import tensorflow as tf

import tsc_util as tscu

def cnn_model(input_shape = (32, 32, 3), classes = 43):
    X_input = Input(input_shape)
    X= Conv2D(32, kernel_size=[3,3], strides=[2,2], padding="same", 
              kernel_initializer= glorot_uniform(seed= 1), name="conv1")(X_input)
    X= BatchNormalization(axis= -1, name="bn1")(X)
    X= Activation("relu")(X)
    X= MaxPooling2D([3,3],[2,2])(X)
    
    X= Conv2D(64, kernel_size=[3,3], strides=[2,2],padding="same",
              kernel_initializer= glorot_uniform(seed =1), name= "conv2")(X)
    X= BatchNormalization(axis= -1, name="bn2")(X)
    X= Activation("relu")(X)
    X= MaxPooling2D([3,3],[2,2])(X)
    
    X= Flatten()(X)
    X= Dense(classes, activation="softmax", name='fc' + str(classes),
             kernel_initializer = glorot_uniform(seed=0))(X)
    
    model= Model(inputs= X_input, outputs= X, name= "cnn")
    
    return model
    


#%% main
if __name__=="__main__":
    X_train, Y_train, X_test, Y_test= tscu.load_data()
    nClass= np.unique(Y_train).shape[0]
    
    model = cnn_model(input_shape = (32, 32, 3), classes = nClass)
    
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        
    # Prep images
    X_train = tscu.prep_images(X_train)
    X_test = tscu.prep_images(X_test)
    
    # Convert training and test labels to one hot matrices
    #Y_train_oh =  tf.one_hot(np.squeeze(Y_train), nClass)
    #Y_test_oh = tf.one_hot(np.squeeze(Y_test), nClass)
    Y_train_oh= tscu.convert_to_one_hot(Y_train, nClass).T
    Y_test_oh= tscu.convert_to_one_hot(Y_test, nClass).T
    
    # train now
    model.fit(X_train, Y_train_oh, epochs = 20, batch_size = 32,)
    
    # test now
    preds = model.evaluate(X_test, Y_test_oh)
    print ("Loss = " + str(preds[0]))
    print ("Test Accuracy = " + str(preds[1]))
    
    # model summary
    model.summary()
    #plot_model(model, to_file='model.png')
    #SVG(model_to_dot(model).create(prog='dot', format='svg'))