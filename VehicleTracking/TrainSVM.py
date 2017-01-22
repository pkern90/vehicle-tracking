import pickle
import time

import numpy as np
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.utils import shuffle

from VehicleTracking.ImageUtils import FeatureExtractor

if __name__ == '__main__':
    with open('../data/data.p', 'rb') as f:
        data = pickle.load(f)

    # with open('../data/data_hnm.p', 'rb') as f:
    #     data_hnm = pickle.load(f)

    # X_train, y_train = data_hnm
    X_val, y_val = data['val']
    X_train, y_train = data['train']

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

    clf_params = {'clf__C': [1e-2, 1e-1, 1, 1e1, 1e2]}

    feature_params = {'feature_extractor__cells_per_block': [2, 3, 4],
                      'feature_extractor__pix_per_cell': [4, 8, 16],
                      'feature_extractor__cspace': ['HLS', 'YUV', 'LAB'],
                      'feature_extractor__orient': [8, 9, 10, 12, 15, 20]}

    # clf_params = {'clf__C': [1e-2]}
    #
    # feature_params = {'feature_extractor__cells_per_block': [1],
    #                   'feature_extractor__pix_per_cell': [4],
    #                   'feature_extractor__cspace': ['LAB'],
    #                   'feature_extractor__orient': [20],
    #                   'feature_extractor__normalize': [False]}

    parameters = {**clf_params, **feature_params}

    feature_chooser = FeatureExtractor()
    clf = LinearSVC()
    pipe = Pipeline([('feature_extractor', feature_chooser),
                     ('clf', clf)])

    cls = GridSearchCV(pipe, parameters, n_jobs=-1, verbose=3, scoring='f1')

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

    with open('../models/svm_hnm.p', 'wb') as f:
        pickle.dump(cls.best_estimator_, f)
