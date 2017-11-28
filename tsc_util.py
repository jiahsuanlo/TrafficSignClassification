# -*- coding: utf-8 -*-
import pickle
import numpy as np
import cv2
from PIL import Image
import tensorflow as tf


#%% Utillity functions

def convert_to_one_hot(Y, C):
    Y = np.eye(C)[Y.reshape(-1)].T
    return Y

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

def flip_images(X):
    """ image augmentation
    
    input:
        X: images [nb,nh,nw,nc]\n
        
    output:
        X_out: output images [nb,nh,nw,nc]    
    """
    X_out= X[:,:,::-1,:]
    return X_out

def crop_and_resize(X):
    """ image random crop and resize
    
    input:
        X: images [nb,nh,nw,nc]\n
        
    output:
        X_out: output images [nb,nh,nw,nc]    
    """
    nb= X.shape[0]
    nh= X.shape[1]
    nw= X.shape[2]
    h_margin= int(nh*0.1)
    w_margin= int(nw*0.1)
    
    x1= np.random.randint(0, w_margin)
    y1= np.random.randint(0, h_margin)
    x2= nw - np.random.randint(0, w_margin)
    y2= nh - np.random.randint(0, h_margin)
    # crop now
    X_crop= X[:,y1:y2,x1:x2,:]
    # resize to original size
    X_resize= []
    for i in range(nb):
        img= cv2.resize(np.squeeze(X_crop[i,...]),(nh,nw))
        X_resize.append(img)
    X_resize= np.array(X_resize)
    return X_resize    

def rotate_images(X):
    """ random rotates images
    
    input:
        X: images [nb,nh,nw,nc]\n
        
    output:
        X_out: output images [nb,nh,nw,nc]    
    """
    nb,nh,nw,nc= X.shape
    deg= np.random.randint(-5,5)
    M= cv2.getRotationMatrix2D((nw/2,nh/2),deg,1)
    X_rot= []
    for i in range(nb):
        img= cv2.warpAffine(np.squeeze(X[i,...]),M,(nw,nh))
        X_rot.append(img)
    X_rot= np.array(X_rot)
    return X_rot

def adjust_brightness(X):
    """ adjust brightness of the images
    
    input:
        X: images [nb,nh,nw,nc] \n
    
    return:
        X_prep: output images [nb,nh,nw,nc]   
    """
    nb,nh,nw,nc= X.shape
    sf_min= 0.5
    sf_max= 1
    
    sf= np.random.rand()*sf_min + (sf_max-sf_min)
    Xout= np.minimum(np.maximum(0, sf*X),255).astype(np.uint8)    
    return Xout
    
def augment_data(X,Y,size_ratio):
    """ augment images
    
    input:
        X: images [nb,nh,nw,nc] \n
        Y: labels [nb,1]\n
        size_ratio: augmented ratio= 
           num of augmented images/ num of images -1 
        
    return:
        X_aug: output augmented images [nb,nh,nw,nc]
        Y_aug: output augmented labels 
    """
    nb,nh,nw,nc= X.shape
    nb_aug= int(nb*(1+size_ratio))
    nb_aug4= nb_aug//4
    
    inds= np.random.permutation(nb)
    
    
    
    
    
    
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

def test_augment_image():
    X_train,Y_train,X_test,Y_test= load_data()
    X= X_train[0:3,...]
    X_flip= flip_images(X)
    X_crop= crop_and_resize(X)
    X_rot= rotate_images(X)
    X_bright= adjust_brightness(X)
    
    img= Image.fromarray(np.squeeze(X[0,...]))
    img_flip= Image.fromarray(np.squeeze(X_flip[0,...]))
    img_crop= Image.fromarray(np.squeeze(X_crop[0,...]))
    img_rot= Image.fromarray(np.squeeze(X_rot[0,...]))
    img_bright= Image.fromarray(np.squeeze(X_bright[0,...]))
    img.show()
    img_flip.show()
    img_crop.show()
    img_rot.show()
    img_bright.show()
    
        
    
    
#%% main
if __name__=="__main__":
    #test_load_data()
    #test_prep_image()
    test_augment_image()
