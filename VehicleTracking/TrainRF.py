import pickle
import time

import numpy as np
from ImageUtils import extract_features
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import log_loss
from sklearn.model_selection import GridSearchCV
from sklearn.svm import LinearSVC
from sklearn.utils import shuffle
import scipy as sp

with open('../data/data.p', 'rb') as f:
    data = pickle.load(f)

with open('../data/data_hnm.p', 'rb') as f:
    data_hnm = pickle.load(f)

X_train, y_train = data_hnm
X_val, y_val = data['val']

X_train = np.concatenate([X_train, X_val])
y_train = np.concatenate([y_train, y_val])
X_train, y_train = shuffle(X_train, y_train, random_state=7)

X_test, y_test = data['test']

orient = 9
pix_per_cell = 8
cells_per_block = 2
spatial_size = (32, 32)
hist_bins = 32
hist_range = (0, 256)
cspace = 'LAB'
hog_channel = 0
pca = 64

X_train = extract_features(X_train, cspace=cspace, orient=orient, pix_per_cell=pix_per_cell,
                           cells_per_block=cells_per_block, normalize=True)

X_test = extract_features(X_test, cspace=cspace, orient=orient, pix_per_cell=pix_per_cell,
                          cells_per_block=cells_per_block, normalize=True)


def log_score(estimator, X, y):
    pred = estimator.predict_proba(X)
    return -log_loss(y, pred)


def des_func_loss(estimator, X, y):
    des_func = estimator.decision_function(X)
    des_func += np.min(des_func)
    des_func /= np.max(des_func)
    epsilon = 1e-15
    des_func = sp.maximum(epsilon, des_func)
    des_func = sp.minimum(1 - epsilon, des_func)
    return -log_loss(y, des_func)


parameters = [{'loss': ['squared_hinge'],
               'C': [1e-3, 1e-2, 1e-1, 1, 1e1, 1e2, 1e3],
               'dual': [False],
               'penalty': ['l1', 'l2'],
               'tol': [1e-4, 1e-3, 1e-2, 1e-1]},
              {'loss': ['hinge'],
               'C': [1e-3, 1e-2, 1e-1, 1, 1e1, 1e2, 1e3],
               'dual': [True],
               'penalty': ['l2'],
               'tol': [1e-4, 1e-3, 1e-2, 1e-1]}]


parameters = {"max_depth": [3, None],
              "max_features": [1, 3, 10],
              "min_samples_split": [1, 3, 10],
              "min_samples_leaf": [1, 3, 10],
              "bootstrap": [True, False],
              "criterion": ["gini", "entropy"]}

#parameters = {'penalty': ['l1'], 'C': [0.01], 'dual': [False], 'tol': [0.1], 'loss': ['squared_hinge']}

cls = RandomForestClassifier()
cls = GridSearchCV(cls, parameters, n_jobs=4, verbose=1, scoring=log_score)

print('Begin training')
t = time.time()
cls.fit(X_train, y_train)
t2 = time.time()
print('Finished training after ', t2 - t, ' seconds')

print('Best params: ', cls.best_params_)
print('Train Accuracy of SVC = ', cls.best_estimator_.score(X_train, y_train))
print('Validation Accuracy of SVC = ', cls.best_estimator_.score(X_test, y_test))

predictions = cls.predict(X_test)
print(classification_report(y_test, predictions, target_names=['no car', 'car']))

# perf_runs = 10
# t = time.time()
# _ = [cls.predict(X_val[i].reshape(1, -1)) for i in range(perf_runs)]
# t2 = time.time()
# print((t2 - t) / perf_runs, ' seconds avg for one prediction')
#
# clf = CalibratedClassifierCV(cls.best_estimator_)
# clf.fit(X_train, y_train)

# t = time.time()
# _ = [clf.predict_proba(X_val[i].reshape(1, -1)) for i in range(perf_runs)]
# t2 = time.time()
# print((t2 - t) / perf_runs, ' seconds avg for one propability prediction')

with open('../models/rf_hnm.p', 'wb') as f:
    pickle.dump(cls.best_estimator_, f)
