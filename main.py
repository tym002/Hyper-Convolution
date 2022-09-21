#!/usr/bin/env python3
# -*- coding: utf-8 -*-


from unet_vanilla import *
import os
import numpy as np
from keras.callbacks import ModelCheckpoint
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import pandas as pd
import argparse
import json


def load_train_data(x_train_path, y_train_path, x_val_path, y_val_path):
    """
    load the training data and ground truth
    """
    x_train = np.load(x_train_path)
    y_train = np.load(y_train_path)
    x_val = np.load(x_val_path)
    y_val = np.load(y_val_path)

    return x_train, y_train, x_val, y_val


def load_test_data(x_test_path, y_test_path):
    """
    load the test data and ground truth
    """
    x_test = np.load(x_test_path)
    y_test = np.load(y_test_path)

    return x_test, y_test


def train_validate(gpu, b_size, weight_path, save_path, history_path, config_arg):
    print('----- Loading and preprocessing train data... -----')

    imgs_test, mask_test = load_test_data(config_arg["x_test_path"], config_arg["y_test_path"])
    x_train, y_train, x_test, y_test = load_train_data(config_arg["x_train_path"], config_arg["y_train_path"],
                                                       config_arg["x_val_path"], config_arg["y_val_path"])

    print('Number of train:', x_train.shape[0])
    print('Number of val:', x_test.shape[0])
    print('Number of test:', imgs_test.shape[0])

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu

    print('----- Creating and compiling model... -----')

    data_gen_args = dict(featurewise_center=False,
                         featurewise_std_normalization=False,
                         horizontal_flip=False,
                         vertical_flip=False,
                         width_shift_range=0,
                         height_shift_range=0,
                         zoom_range=0,
                         rotation_range=0
                         )
    image_datagen = ImageDataGenerator(**data_gen_args)
    mask_datagen = ImageDataGenerator(**data_gen_args)

    seed = 42

    image_generator = image_datagen.flow(x_train, batch_size=b_size, seed=seed)
    mask_generator = mask_datagen.flow(y_train, batch_size=b_size, seed=seed)

    train_generator = (pair for pair in zip(image_generator, mask_generator))

    model = unet(img_shape=(config_arg["img_shape_x"], config_arg["img_shape_y"], config_arg["img_shape_z"]),
                 # size of the input image
                 depth=config_arg["depth"],  # nums of max-pooling layers
                 dropout=config_arg["dropout"],  # dropout rate
                 activation=config_arg["activation"],  # non-linear activation type
                 start_ch=config_arg["start_ch"],  # initial channels
                 residual=config_arg["residual"],  # Residual connection
                 batchnorm=config_arg["batchnorm"],  # Batch Normalization
                 att=config_arg["attention"],  # Attention module
                 nl=config_arg["non_local"],  # Non-local block at the bottom
                 pos=config_arg["pos"],  # Positional encoding
                 hyper=config_arg["hyper"])  # Hyper-conv kernel size. Use False for regular network

    model_checkpoint = ModelCheckpoint(weight_path,
                                       monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=True)

    print('----- Fitting model... -----')

    mtrain = model.fit_generator(train_generator, steps_per_epoch=len(x_train) // b_size,
                                 epochs=1000, verbose=1, shuffle=True, callbacks=[model_checkpoint],
                                 validation_data=(x_test, [y_test]))

    model_predict = model.predict([imgs_test], verbose=1, batch_size=16)
    np.save(save_path, model_predict)
    pd.DataFrame.from_dict(mtrain.history).to_csv(history_path, index=False)


def prediction(gpu, weight_path, save_path, config_arg):
    print('----- Loading and preprocessing test data... -----')

    imgs_test, mask_test = load_test_data(config_arg["x_test_path"], config_arg["y_test_path"])

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu

    print('----- Creating and compiling model... -----')

    ##############################
    # this is the prediction!!   #
    ##############################
    model = unet(img_shape=(config_arg["img_shape_x"], config_arg["img_shape_y"], config_arg["img_shape_z"]),
                 # size of the input image
                 depth=config_arg["depth"],  # nums of max-pooling layers
                 dropout=config_arg["dropout"],  # dropout rate
                 activation=config_arg["activation"],  # non-linear activation type
                 start_ch=config_arg["start_ch"],  # initial channels
                 residual=config_arg["residual"],  # Residual connection
                 batchnorm=config_arg["batchnorm"],  # Batch Normalization
                 att=config_arg["attention"],  # Attention module
                 nl=config_arg["non_local"],  # Non-local block at the bottom
                 pos=config_arg["pos"],  # Positional encoding
                 hyper=config_arg["hyper"])  # Hyper-conv kernel size. Use False for regular network

    model.load_weights(weight_path)
    print(model.summary())

    print('----- Fitting model... -----')

    model_predict = model.predict([imgs_test], verbose=1, batch_size=1)
    np.save(save_path, model_predict)


def main(arg, config_arg):
    mode = arg.mode
    batch_size = arg.b_size
    gpu = arg.gpu
    file_name = arg.file_name
    save_folder = '/result/' + arg.folder_name + '/'
    weight_path = save_folder + file_name + '.hdf5'
    save_path = save_folder + 'Prediction_' + file_name + '.npy'
    history_path = save_folder + 'history_' + file_name + '.csv'

    if not os.path.exists(save_folder):
        print(f'making save folder {save_folder}')
        os.makedirs(save_folder)
    if mode == "train":
        train_validate(gpu, batch_size, weight_path, save_path, history_path, config_arg)
    elif mode == "test":
        prediction(gpu, weight_path, save_path, config_arg)
    else:
        print("mode should be either train or test")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="training and testing script")
    parser.add_argument("--mode", default="train", help="train or test")
    parser.add_argument("--config_path", default="config.json", help="path to config file")
    parser.add_argument("--folder_name", default="training", help="name of the folder to save results")
    parser.add_argument("--file_name", default="training", help="name of the trained model file")
    parser.add_argument("--gpu", default=0, help="which gpu to use")
    parser.add_argument("--b_size", default=8, help="batch size")

    args = parser.parse_args()
    config_args = json.load(open(args.config_path))
    main(args, config_args)
