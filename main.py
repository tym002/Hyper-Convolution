#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 21 23:25:48 2020

@author: tianyu
"""

from unet_vanilla import *
import os
import numpy as np
from keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import pandas as pd
from kernel_visual import *

mode = 't'
folder_name = '...'
file_name = '...'
gpu = "0"
b_size = 8
save_folder = '...'+folder_name+'/'
weight_path = save_folder+file_name+'.hdf5'
save_path = save_folder+'Prediction_'+file_name+'.npy'


def load_train_data():
    '''
    load the training data and ground truth   
    '''
    x_train = np.load('...')
    y_train = np.load('...')  
    x_test = np.load('...')
    y_test = np.load('...') 

    return x_train,y_train,x_test,y_test    

def load_test_data():
    '''
    load the test data and ground truth   
    '''
    imgs_train = np.load('...')
    imgs_mask = np.load('...')

    return imgs_train, imgs_mask


def TrainandValidate(gpu):  

    print('----- Loading and preprocessing train data... -----')
    
    
    imgs_test, mask_test = load_test_data()
    x_train,y_train,x_test,y_test = load_train_data()    
    
    print('Number of train:',x_train.shape[0])
    print('Number of val:',x_test.shape[0])
    print('Number of test:',imgs_test.shape[0])

    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"]= gpu
      
    print('----- Creating and compiling model... -----')
    
    data_gen_args = dict(featurewise_center=False,
                     featurewise_std_normalization=False,
                     horizontal_flip=True,
                     vertical_flip=True,
                     width_shift_range=0.2,
                     height_shift_range=0.2,
                     zoom_range=0.2,
                     rotation_range=20
                     )
    image_datagen = ImageDataGenerator(**data_gen_args)
    mask_datagen = ImageDataGenerator(**data_gen_args)
    seed = 1
   
    image_generator = image_datagen.flow(x_train,batch_size=b_size,seed=seed)
    mask_generator = mask_datagen.flow(y_train,batch_size=b_size,seed=seed)

    train_generator = (pair for pair in zip(image_generator, mask_generator))


    model = unet(img_shape=(256,256,1),
                depth=3, # nums of maxpooling layers
                dropout=0.5,
                activation='relu',
                start_ch = 4, # initial channels
                residual=False, # Residual connection
                batchnorm=True, # Batch Normalization
                att=False, # Attention 
                nl = False, # Non-local block at the bottom
                pos= False, # Positional encoding
                hyper=5) # Hyper-conv kernel size. Use False for regular network
    print(model.summary())
        
    model_checkpoint = ModelCheckpoint(weight_path, 
                                       monitor='val_loss',verbose=1, save_best_only=True,save_weights_only = True)
    
    print('----- Fitting model... -----')
        
    mtrain = model.fit_generator(train_generator, steps_per_epoch=len(x_train)//b_size,
                                 epochs=400, verbose=1, shuffle=True,callbacks=[model_checkpoint],validation_data=(x_test, [y_test]))
        
    model_predict = model.predict([imgs_test], verbose =1 ,batch_size =8)
    np.save(save_path, model_predict)
    plt.plot(mtrain.history['loss'])
    plt.plot(mtrain.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Training', 'Validation'], loc = 'upper left')
    plt.show()
    plt.savefig(save_folder+file_name)
  
    pd.DataFrame.from_dict(mtrain.history).to_csv(save_folder+'history_'+file_name+'.csv',index=False)
    
def prediction(gpu):

    print('----- Loading and preprocessing test data... -----')

    imgs_test, mask_test = load_test_data()

    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"]= gpu

    print('----- Creating and compiling model... -----')
    
    ##############################
    ### this is the prediction!!##
    ##############################
    model = unet(img_shape=(256,256,1),
                depth=3, # nums of maxpooling layers
                dropout=0.5,
                activation='relu',
                start_ch = 4, # initial channels
                residual=False, # Residual connection
                batchnorm=True, # Batch Normalization
                att=False, # Attention 
                nl = False, # Non-local block at the bottom
                pos= False, # Positional encoding
                hyper=5) # Hyper-conv kernel size. Use False for regular network

    model.load_weights(weight_path)
    print(model.summary())

    print('----- Fitting model... -----')
    
    model_predict = model._predict([imgs_test], verbose =1 ,batch_size =1)
    np.save(save_path, model_predict)

if __name__ == '__main__':
    if not os.path.exists(save_folder):
       print('making folder...')
       os.makedirs(save_folder)
    if mode == 't':
       model = TrainandValidate(gpu)
    else:
       model = prediction(gpu)


