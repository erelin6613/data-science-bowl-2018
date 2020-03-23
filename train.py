#!/usr/bin/env python3

import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.image as image
from sklearn.model_selection import train_test_split
from skimage.transform import rescale, resize
from skimage.io import imread
from keras.models import Model, load_model, save_model
from keras.layers import Input
from keras.layers.core import Dropout, Lambda
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.pooling import MaxPooling2D
from keras.layers.merge import concatenate
import keras.backend as K

"""
Specify the root directory where data and scripts are placed
"""

#ROOT_PATH = '/content/drive/My Drive/data-science-bowl-2018'
ROOT_PATH = ''

IMG_HEIGHT = 128
IMG_WIDTH = 128
IMG_CHANNELS = 3
STAGE_1_LABELS = os.path.join(ROOT_PATH, 'stage1_train_labels.csv')
STAGE_1_SOLUTION = os.path.join(ROOT_PATH, 'stage1_solution.csv')
STAGE_1_TEST = os.path.join(ROOT_PATH, 'stage1_test')
STAGE_1_TRAIN = os.path.join(ROOT_PATH, 'stage1_train')
STAGE_2_TEST = os.path.join(ROOT_PATH, 'stage2_test_final')
EPOCHS = 50


"""
Function get_images
:param ids: identifiers of images to locate in the
	data sctructure
:param height: image height
:param width: image width
:param channels: image channels (3 for RGB, 1 for Grayscale)
:param return_masks: load with or without corresponding mask
return: if return_masks is set to true tuple of ndarrays
	correcsponding to image and its mask
	if return_masks is set to false ndarrays for images only
"""
def get_images(ids, height=IMG_HEIGHT, width=IMG_WIDTH,
                channels=IMG_CHANNELS, return_masks=False):

    X = np.zeros((len(ids), height, width, channels))
    y = np.zeros((len(ids), height, width, 1), dtype=np.bool)
    for n, i in tqdm(enumerate(ids), total=len(ids)):
        path = os.path.join(STAGE_1_TRAIN, i)
        img = imread(os.path.join(path, 'images', i+'.png'))[:,:,:channels]
        img = resize(img, (IMG_HEIGHT, IMG_WIDTH),
                        mode='constant', preserve_range=True)
        X[n] = img
        if return_masks:
            mask = np.zeros((height, width, 1), dtype=np.bool)
            for mask_file in next(os.walk(path + '/masks/'))[2]:
                mask_ = imread(path + '/masks/' + mask_file)
                mask_ = np.expand_dims(resize(mask_, (height, width),
                                            mode='constant',
                                            preserve_range=True), axis=-1)
                mask = np.maximum(mask, mask_)
            y[n] = mask

    if return_masks:
        return (X, y)
    return X


"""
Loss function dice_loss
:param y_true: true masks tensor
:param y_pred: predicted masks
:return: loss to propogate backward
"""
def dice_loss(y_true, y_pred, smooth=1):

    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return -((2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth))


"""
Function build_unet_model
Building the basic UNet model
:param height: heigth of training images
:param width: heigth of training width
:param channels: channels of images 

:return: Keras Model object
"""
def build_unet_model(height=IMG_HEIGHT,
                    width=IMG_WIDTH,
                    channels=IMG_CHANNELS):

    inputs = Input((height, width, channels))
    inp = Lambda(lambda x: x / 255) (inputs)

    conv1 = Conv2D(16, (3, 3), activation='relu', padding='same')(inp)
    conv1 = Dropout(0.2)(conv1)
    conv1 = Conv2D(16, (3, 3), activation='relu', padding='same')(conv1)
    pool1 = MaxPooling2D((2, 2))(conv1)

    conv2 = Conv2D(32, (3, 3), activation='relu', padding='same')(pool1)
    conv2 = Dropout(0.2)(conv2)
    conv2 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv2)
    pool2 = MaxPooling2D((2, 2))(conv2)

    conv3 = Conv2D(64, (3, 3), activation='relu', padding='same')(pool2)
    conv3 = Dropout(0.2)(conv3)
    conv3 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv3)
    pool3 = MaxPooling2D((2, 2)) (conv3)

    conv4 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool3)
    conv4 = Dropout(0.2)(conv4)
    conv4 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = Conv2D(256, (3, 3), activation='relu', padding='same')(pool4)
    conv5 = Dropout(0.2)(conv5)
    c5 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv5)

    upconv1 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(conv5)
    upconv1 = concatenate([upconv1, conv4])
    conv6 = Conv2D(128, (3, 3), activation='relu', padding='same')(upconv1)
    conv6 = Dropout(0.1)(conv6)
    conv6 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv6)

    upconv2 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(conv6)
    upconv2 = concatenate([upconv2, conv3])
    conv7 = Conv2D(64, (3, 3), activation='relu', padding='same')(upconv2)
    conv7 = Dropout(0.1)(conv7)
    conv7 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv7)

    upconv8 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(conv7)
    upconv8 = concatenate([upconv8, conv2])
    conv8 = Conv2D(32, (3, 3), activation='relu', padding='same')(upconv8)
    conv8 = Dropout(0.1)(conv8)
    conv8 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv8)

    upconv9 = Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same')(conv8)
    upconv9 = concatenate([upconv9, conv1], axis=3)
    conv9 = Conv2D(16, (3, 3), activation='relu', padding='same')(upconv9)
    conv9 = Dropout(0.1)(conv9)
    conv9 = Conv2D(16, (3, 3), activation='relu', padding='same')(conv9)

    outputs = Conv2D(1, (1, 1), activation='sigmoid')(conv9)

    model = Model(inputs=[inputs], outputs=[outputs])
    model.compile(optimizer='adam',
                    loss=iou_loss,
                    metrics=['accuracy'])
    model.summary()

    return model

def main():
    stage_1_train_ids = next(os.walk(STAGE_1_TRAIN))[1]
    stage_1_test_ids = next(os.walk(STAGE_1_TRAIN))[1]
    #stage_2_test_ids = next(os.walk(STAGE_1_TRAIN))[1]
    print('Loading images...')
    X_train, y_train = get_images(ids = stage_1_train_ids, return_masks=True)
    X_train, X_test, y_train, y_test = train_test_split(X_train, 
                                                        y_train, 
                                                        test_size=0.2, 
                                                        random_state=0)

    model = build_unet_model()
    print('Training model...')
    model.fit(X_train, y_train, batch_size=16, epochs=EPOCHS, verbose=2)
    print('How good is our model performing?')
    print(model.evaluate(X_test, y_test, batch_size=16))
    model.save('unet_model.h5')

if __name__ == '__main__':
    main()
