#!/usr/bin/env python3

from train import get_images

import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from skimage.transform import rescale, resize
from skimage.io import imread
from keras.models import load_model

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

def main():

	model = load_model('unet_model.h5')

	stage_1_test_ids = next(os.walk(STAGE_1_TEST))[1]
	X_test = get_images(ids = stage_1_train_ids, return_masks=False)
	results = model.predict(X_test)
	rles = []
	for n, img_id in enumerate(test_ids):
	    rle = list(prob_to_rles(preds_test_upsampled[n]))
	    rles.extend(rle)
	    new_test_ids.extend([img_id] * len(rle))
	results = pd.DataFrame()
	results['ImageId'] = new_test_ids
	results['EncodedPixels'] = pd.Series(rles).apply(lambda x: ' '.join(str(y) for y in x))
	results.to_csv('results.csv')

if __name__ == '__main__':
    main()
