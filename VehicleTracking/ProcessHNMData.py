import os
import pickle
from glob import glob

import numpy as np
from skimage.io import imread

neg_mining_paths = []
neg_mining_dir = '../data/neg_mining/'
pattern = "*.png"
img_shape = (64, 64, 3)
nb_samples = 10000

if __name__ == '__main__':
    """
    This script loads al images from the hard negative mining, creates labels
    and combines them into a pickle files for easier processing.
    """

    # Neg mined images
    for dir, _, _ in os.walk(neg_mining_dir):
        neg_mining_paths.extend(glob(os.path.join(dir, pattern)))
    print('Neg mined images found: ', len(neg_mining_paths))

    if nb_samples == -1:
        nb_samples = len(neg_mining_paths)

    # Select a sample with a given size
    selection = np.random.choice(len(neg_mining_paths), nb_samples, replace=False)
    images = np.zeros((len(selection), *img_shape), dtype=np.uint8)
    for i, path_ix in enumerate(selection):
        images[i] = imread(neg_mining_paths[path_ix])[:, :, :3]
    labels = np.zeros(nb_samples, dtype=np.uint8)

    print('Neg Mined sample size: ', len(images))

    with open('../data/data_hnm.p', 'wb') as f:
        pickle.dump((images, labels), f)
