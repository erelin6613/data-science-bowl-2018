import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.image as image
from skimage.transform import rescale, resize
from skimage.io import imread
"""
from keras.models import Model, load_model, save_model
from keras.layers import Input
from keras.layers.core import Dropout, Lambda
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.pooling import MaxPooling2D
from keras.layers.merge import concatenate
"""

IMG_HEIGHT = 64
IMG_WIDTH = 64
IMG_CHANNELS = 3
STAGE_1_LABELS = 'stage1_train_labels.csv'
STAGE_1_SOLUTION = 'stage1_solution.csv'
STAGE_1_TEST = 'stage1_test'
STAGE_1_TRAIN = 'stage1_train'
STAGE_2_TEST = 'stage2_test_final'

def get_images(ids, height=IMG_HEIGHT, width=IMG_WIDTH,
                channels=IMG_CHANNELS, return_masks=False):

    X = np.zeros((len(ids), height, width, channels)) #, dtype=np.uint8)
    y = np.zeros((len(ids), height, width, 1), dtype=np.bool)
    #sys.stdout.flush()
    for n, i in tqdm(enumerate(ids), total=len(ids)):
        #print(i)
        path = os.path.join(STAGE_1_TRAIN, i)
        #print(path)
        img = imread(os.path.join(path, 'images', i+'.png'))[:,:,:channels]
        #plt.imshow(img)#print(img)
        #plt.show()
        #return
        #print(img.shape)
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


def resize_image(image, output_size=IMG_WIDTH):

    image = resize(image, (output_size, output_size, IMG_CHANNELS),
                    anti_aliasing=True)
    return image

def build_unet_model(height=IMG_HEIGHT,
                    width=IMG_WIDTH,
                    channels=IMG_CHANNELS):

    inputs = Input((height, width, channels))
    inp = Lambda(lambda x: x / 255) (inputs)

    conv1 = Conv2D(16, (3, 3), activation='relu', padding='same')(inp)
    conv1 = Conv2D(16, (3, 3), activation='relu', padding='same')(conv1)
    pool1 = MaxPooling2D((2, 2))(conv1)

    conv2 = Conv2D(32, (3, 3), activation='relu', padding='same')(pool1)
    conv2 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv2)
    pool2 = MaxPooling2D((2, 2))(conv2)

    conv3 = Conv2D(64, (3, 3), activation='relu', padding='same')(pool2)
    conv3 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv3)
    pool3 = MaxPooling2D((2, 2)) (conv3)

    conv4 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool3)
    conv4 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2)) (conv4)

    conv5 = Conv2D(256, (3, 3), activation='relu', padding='same')(pool4)
    c5 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv5)

    upconv1 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(conv5)
    upconv1 = concatenate([upconv1, conv4])
    conv6 = Conv2D(128, (3, 3), activation='relu', padding='same')(upconv1)
    conv6 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv6)

    upconv2 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(conv6)
    upconv2 = concatenate([upconv2, conv3])
    conv7 = Conv2D(64, (3, 3), activation='relu', padding='same')(upconv2)
    conv7 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv7)

    upconv8 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(conv7)
    upconv8 = concatenate([upconv8, conv2])
    conv8 = Conv2D(32, (3, 3), activation='relu', padding='same')(upconv8)
    conv8 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv8)

    upconv9 = Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same')(conv8)
    upconv9 = concatenate([u9, c1], axis=3)
    conv9 = Conv2D(16, (3, 3), activation='relu', padding='same')(upconv9)
    conv9 = Conv2D(16, (3, 3), activation='relu', padding='same')(conv9)

    outputs = Conv2D(1, (1, 1), activation='sigmoid')(conv9)

    model = Model(inputs=[inputs], outputs=[outputs])
    model.compile(optimizer='adam',
                    loss='binary_crossentropy',
                    metrics=['accuracy'])
    model.summary()

    return model

def main():
    stage_1_train_ids = next(os.walk(STAGE_1_TRAIN))[1][:50]
    stage_1_test_ids = next(os.walk(STAGE_1_TEST))[1][:10]
    stage_2_test_ids = next(os.walk(STAGE_2_TEST))[1][:10]
    #print(stage_1_train_ids)
    X_train, y_train = get_images(ids = stage_1_train_ids, return_masks=True)
    #get_images9ids = stage_1_test
    model = build_unet_model()
    model.fit(X_train, y_train)

if __name__ == '__main__':
    main()
