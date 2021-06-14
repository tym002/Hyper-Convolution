#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import numpy as np 
import os

import numpy as np
from keras.models import Model
from keras.layers import Softmax,Lambda,Input,Conv2D,UpSampling2D,Dropout,MaxPooling2D,Concatenate,BatchNormalization,Activation,Conv2DTranspose,LeakyReLU
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
import tensorflow as tf
from keras import backend as K
from keras.losses import binary_crossentropy
from non_local import non_local_block
from kernal import hyperNet
from BiasNet import BiasNet
img_rows = 512
img_cols = 512
in_c = 1

def dice_coef(y_true, y_pred):
    y_pred = tf.cast((y_pred>0.5),tf.float32)
    intersection = K.sum(y_true * y_pred)
    union = K.sum(y_true) + K.sum(y_pred)
    return (2. * intersection + 0.01) / (union + 0.01)

def soft_dice_loss(y_true, y_pred):
    numerator = 2. * K.sum(y_pred * y_true) + 1.0
    denominator = K.sum(K.square(y_pred)) + K.sum(K.square(y_true)) + 1.0
    loss = 1 - (numerator / denominator)
    return loss

def combine_loss(y_true, y_pred):
    crossentropy = binary_crossentropy(y_true, y_pred)
    return soft_dice_loss(y_true, y_pred) + crossentropy


def Tversky_loss(b):
    def loss(y_true,y_pred):
        beta = b
        TP = K.sum(y_pred * y_true)
        FN = beta * K.sum((1-y_pred)*y_true)
        FP = (1-beta) * K.sum(y_pred*(1-y_true))
        return 1 - (TP + 1)/(TP + FN + FP + 1)
    return loss

def coverage(y_true,y_pred):
    y_pred = tf.cast((y_pred>0.5),tf.float32)
    return tf.reduce_sum(y_true*y_pred)/(tf.reduce_sum(y_true)+K.epsilon())


def conv_block(m,dim,acti='relu',bn=False,res=False,do=0):
    n = Conv2D(dim, (3, 3), padding='same',dilation_rate=(1,1))(m)
    n = BatchNormalization()(n) if bn else n
    n = Activation(acti)(n)
    
    n = Dropout(do)(n) if do else n
    
    n = Conv2D(dim, (3, 3), padding='same',dilation_rate=(1,1))(n)
    n = BatchNormalization()(n) if bn else n
    n = Activation(acti)(n)
    return Concatenate()([m, n]) if res else n

def p_conv(ip,kernal):
    pos = tf.squeeze(kernal,axis=0)
    out = tf.nn.convolution(ip,pos,padding='SAME') 
    #out = Activation('relu')(out)
    return out

def p_kernal(ip,x_dim,y_dim,ch_in,ch_out):
    num_c = int(ch_in * ch_out)
    pos = Conv2D(16, (1, 1), padding='same', activation=None, use_bias=True, kernel_initializer='he_normal')(ip)
    pos = LeakyReLU(alpha=0.1)(pos)
    
    pos = Conv2D(16, (1, 1), padding='same', activation=None, use_bias=True, kernel_initializer='he_normal')(pos)
    pos = LeakyReLU(alpha=0.1)(pos)


    pos = Conv2D(4, (1, 1), padding='same', activation=None, use_bias=True, kernel_initializer='he_normal')(pos)
    pos = LeakyReLU(alpha=0.1)(pos)
    
    pos = Conv2D(num_c, (1, 1), padding='same', activation=None, use_bias=True, kernel_initializer='he_normal')(pos)
    pos = Reshape((x_dim, y_dim,ch_in,ch_out))(pos)
    return pos

def hyper_block(ip,x_dim,y_dim,ch_in,ch_out,acti='relu',bn=False,do=0,mode='xy',multi=True,res=False):
    input_channel = ch_in
    kernal1 = Lambda(lambda x: hyperNet(x_dim,y_dim,input_channel,ch_out,mode))(ip)
    kernal1 = p_kernal(kernal1,x_dim,y_dim,input_channel,ch_out)
    n = Lambda(lambda x: p_conv(x[0],x[1]))([ip,kernal1])
    n = BiasNet()(n)
    n = BatchNormalization()(n) if bn else n
    if acti:
       n = Activation(acti)(n)
    n = Dropout(do)(n) if do else n
    if multi:
       kernal2 = Lambda(lambda x: hyperNet(x_dim,y_dim,ch_out,ch_out))(ip)
       kernal2 = p_kernal(kernal2,x_dim,y_dim,ch_out,ch_out)
       n = Lambda(lambda x: p_conv(x[0],x[1]))([n,kernal2])
       n = BiasNet()(n)
       n = BatchNormalization()(n) if bn else n
       if acti:
          n = Activation(acti)(n)

    return Concatenate()([ip, n]) if res else n

def combine_block(ip,x_dim,y_dim,ch_in,ch_out,acti='relu',bn=False,do=0,mode='xy',multi=True,res=False):
    n = Conv2D(ch_out, (3, 3), padding='same')(ip)
    n = BatchNormalization()(n) if bn else n
    n = Activation(acti)(n)

    n = Dropout(do)(n) if do else n

    kernal1 = Lambda(lambda x: hyperNet(x_dim,y_dim,ch_out,ch_out,mode))(n)
    kernal1 = p_kernal(kernal1,x_dim,y_dim,ch_out,ch_out)
    n = Lambda(lambda x: p_conv(x[0],x[1]))([n,kernal1])
    n = BiasNet()(n)
    n = BatchNormalization()(n) if bn else n
    if acti:
       n = Activation(acti)(n)    
    return Concatenate()([ip, n]) if res else n

def level_block(m, dim, depth, inc, acti, do, bn, mp, up, res,att,nl,pos,hyper):
    if depth > 0:
        if hyper:
           ft = hyper
           in_c = 1 if res else 1/2
           n = hyper_block(m,ft,ft,int(in_c*dim),dim,acti=acti,bn=bn,do=do,
                           mode='xy',
                           multi=True,
                           res = res)

        else:
           n = conv_block(m, dim, acti, bn, res)
        if att:
           n1 = Conv2D(1, 1,padding='same',activation='linear')(n)
           n1 = Softmax(axis=(1,2))(n1)
           n = Lambda(lambda x: tf.math.multiply(x[0],x[1]))([n,n1])
        m = MaxPooling2D()(n) if mp else Conv2D(dim, 3, strides=2, padding='same')(n)
        m = level_block(m, int(inc*dim), depth-1, inc, acti, do, bn, mp, up, res,att,nl,pos,hyper)
        if up:
            m = UpSampling2D()(m)
            if hyper:
               in_c = 4 if res else 2
               m = hyper_block(m,hyper,hyper,int(in_c*dim),dim,acti=acti,bn=bn,do=do,
                               mode='xy',
                               multi=False,
                               res = False)
            else:
               m = Conv2D(dim, 3, activation=acti, padding='same')(m)
        else:
            m = Conv2DTranspose(dim, 3, strides=2, activation=acti, padding='same')(m)
        n = Concatenate()([n, m])
        if hyper:
           in_c = 3 if res else 2
           m = hyper_block(n,hyper,hyper,int(in_c*dim),dim,acti=acti,bn=bn,do=do,mode='xy',multi=True,res=res)
        else:
           m = conv_block(n, dim, acti, bn, res)
    else:
        if hyper:
            in_c = 1 if res else 1/2
            m = hyper_block(m,hyper,hyper,int(in_c*dim),dim,acti=acti,bn=bn,do=do,mode='xy',multi=True,res=res)
        else:
            m = conv_block(m, int(dim), acti, bn, res, do)
            if nl or pos:
               m = non_local_block(m,compression=1, mode='dot',nl = nl,posnet=pos)
    return m

def unet(img_shape=(img_rows,img_cols,in_c), out_ch=1, start_ch=16, depth=4, inc_rate=2., activation='relu',dropout=0, batchnorm=False, maxpool=True, upconv=True, residual=False,att=False,nl = False, pos=False, hyper=False):
    i = Input(shape=img_shape)
    if hyper:
       i1 = hyper_block(i,hyper,hyper,1,int(1/2*start_ch),acti=activation,bn=batchnorm,do=dropout,mode='xy',multi=False)
    else:
       i1 = i
    o1 = level_block(i1, start_ch, depth, inc_rate, activation, dropout, batchnorm, maxpool, upconv, residual,att,nl,pos,hyper)
    o1 = Conv2D(out_ch, 1, activation='sigmoid')(o1)
    model = Model(inputs=i, outputs=[o1]) 
    model.compile(optimizer = Adam(lr = 1e-4), loss = soft_dice_loss, metrics = [dice_coef,coverage])
    return model  


