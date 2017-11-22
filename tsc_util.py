# -*- coding: utf-8 -*-
import pickle
import numpy as np
import cv2

#%% Utillity functions

def load_data():
    """ load data util function
    
    input:  \n
    
    return:
        X_train: training set images [nb, nh,nw,nc] 32x32x3
        Y_train: training set label [nb, 1] 
        X_test: test set images [nb, nh,nw,nc] 32x32x3
        Y_test: test set label [nb, 1] 
    """
    train_file= "./data/train.p"
    test_file= "./data/test.p"
    
    with open(train_file, "rb") as f:
        train_dict= pickle.load(f)
    
    with open(test_file, "rb") as f:
        test_dict= pickle.load(f)
        
    X_train= train_dict["features"]
    Y_train= np.expand_dims(train_dict["labels"], axis= -1)
    X_test= test_dict["features"]
    Y_test= np.expand_dims(test_dict["labels"], axis= -1)
    return X_train, Y_train, X_test, Y_test

def prep_images(X):
    """ preprocess images
    - normalize 
    
    input:
        X: images [nb,nh,nw,nc] \n
    
    return:
        X_prep: prep'ed images [nb,nh,nw,nc]   
    """
    X_prep= X/ 255.
    return X_prep

#%% Unit test functions

def test_load_data():
    X_train,Y_train,X_test,Y_test= load_data()
    print(X_train.shape)
    print(Y_train.shape)
    print(X_test.shape)
    print(Y_test.shape)
    print("number of classes: ", np.unique(Y_train).shape[0]) 

def test_prep_image():
    X_train,Y_train,X_test,Y_test= load_data()
    
    X= X_train[:2,...]
    X_prep= prep_images(X)
    
    for i in range(2):
        img= np.squeeze(X[i,...])[...,[2,1,0]]
        img1=np.squeeze(X_prep[i,...])[...,[2,1,0]]
        
        cv2.imshow("original", img)
        cv2.imshow("prep", img1)
        cv2.waitKey(0)
    cv2.destroyAllWindows()
#%% main
if __name__=="__main__":
    test_load_data()
    test_prep_image()

