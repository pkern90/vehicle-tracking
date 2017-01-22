import os
import pickle
from glob import glob

import numpy as np
from skimage.io import imread
from sklearn.utils import shuffle

neg_mining_paths = []
neg_mining_dir = '../data/neg_mining/'
pattern = "*.png"
img_shape = (64, 64, 3)
nb_samples = 5000

# Neg mined images
for dir, _, _ in os.walk(neg_mining_dir):
    neg_mining_paths.extend(glob(os.path.join(dir, pattern)))
print('Neg mined images found: ', len(neg_mining_paths))

selection = np.random.choice(len(neg_mining_paths), nb_samples, replace=False)
images = np.zeros((len(selection), *img_shape), dtype=np.uint8)
for i, path_ix in enumerate(selection):
    images[i] = imread(neg_mining_paths[path_ix])[:, :, :3]
labels = np.zeros(nb_samples, dtype=np.uint8)

# Base dataset
with open('../data/data.p', 'rb') as f:
    data = pickle.load(f)

X_train, y_train = data['train']

X = np.concatenate([X_train, images])
y = np.concatenate([y_train, labels])

X, y = shuffle(X, y, random_state=7)
print('Base train size: ', len(X_train))
print('Neg Mined sample size: ', len(images))
print('Total train size: ', len(X))

with open('../data/data_hnm.p', 'wb') as f:
    pickle.dump((X, y), f)
