import json
import pickle
import time

import numpy as np
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.svm import LinearSVC
from sklearn.utils import shuffle

from FeatureExtractionPipeline import SpatialBining, ColorHistogram, ColorSpaceConverter, \
    OptionalBranch, OptionalPCA, AcceptEmptyMinMaxScaler, HogExtractorOpenCV

from VehicleTracking.FeatureExtractionPipeline import HogExtractor

N_JOBS = 1
SAMPLE_SIZE = 10

if __name__ == '__main__':
    """
    Training script for a linear SVM. Uses GridSearch to find optimal hyper parameters.
    The configurations are loaded from files inside GridSearchConfig.
    """

    np.random.seed(0)

    with open('../data/data_adj.p', 'rb') as f:
        data = pickle.load(f)

    with open('../data/data_hnm.p', 'rb') as f:
        data_hnm = pickle.load(f)

    X_train, y_train = data['train']
    X_hnm, y_hnm = data_hnm
    X_val, y_val = data['val']

    # sample = np.random.choice(len(y_train), SAMPLE_SIZE, replace=False)
    # X_train = X_train[sample]
    # y_train = y_train[sample]

    # Concat train and validation set since we will use K-fold CV
    # X_train = np.concatenate([X_train, X_hnm])
    # y_train = np.concatenate([y_train, y_hnm])

    X_train, y_train = shuffle(X_train, y_train, random_state=7)

    print('Train size:', len(y_train))

    # Pipeline setup
    sb_optional = OptionalBranch()
    sb_csc = ColorSpaceConverter()
    spatial_bining = SpatialBining()
    sb_minmax = AcceptEmptyMinMaxScaler(feature_range=(0, 1), copy=False)
    sb_pipeline = Pipeline([("sb_optional", sb_optional),
                            ("sb_csc", sb_csc),
                            ("spatial_bining", spatial_bining),
                            ("sb_minmax", sb_minmax)])

    chist_optional = OptionalBranch()
    chist_csc = ColorSpaceConverter()
    color_histogram = ColorHistogram()
    chist_minmax = AcceptEmptyMinMaxScaler(feature_range=(0, 1), copy=False)
    chist_pca = OptionalPCA()
    chist_pipeline = Pipeline([("chist_optional", chist_optional),
                               ("chist_csc", chist_csc),
                               ("color_histogram", color_histogram),
                               ("chist_minmax", chist_minmax),
                               ("chist_pca", chist_pca)])

    hoh_csc = ColorSpaceConverter()
    hog_extractor = HogExtractor()
    hog_minmax = AcceptEmptyMinMaxScaler(feature_range=(0, 1), copy=False)
    hog_pca = OptionalPCA()
    hog_pipeline = Pipeline([("hog_csc", hoh_csc),
                             ("hog_extractor", hog_extractor),
                             ("hog_minmax", hog_minmax),
                             ("hog_pca", hog_pca)])

    features = FeatureUnion([("hog", hog_pipeline), ('chist', chist_pipeline), ('sb', sb_pipeline)], n_jobs=1)

    clf = LinearSVC()
    pipeline = Pipeline([('features', features),
                         ('clf', clf)])

    # Specifiy the config file containing the parameters
    with open('GridSearchConfig/FinalConfig.json') as data_file:
        params = json.load(data_file)

    cls = GridSearchCV(pipeline, params, cv=2, n_jobs=N_JOBS, verbose=3, scoring='roc_auc', )

    print('Begin training')
    t = time.time()
    cls.fit(X_train, y_train)
    t2 = time.time()
    print('Finished training after ', t2 - t, ' seconds')

    # Save the best estimator
    with open('../models/svm_adj.p', 'wb') as f:
        pickle.dump(cls.best_estimator_, f)

    # Also save the grid search object for analysis
    with open('../models/gridsearch_adj.p', 'wb') as f:
        pickle.dump(cls, f)

    print('Best params: ', cls.best_params_)
    print('Best auc roc score: ', cls.best_score_)
    print('Train Accuracy of SVC = ', cls.best_estimator_.score(X_train, y_train))
    print('Validation Accuracy of SVC = ', cls.best_estimator_.score(X_val, y_val))

    predictions = cls.predict(X_val)
    print(classification_report(y_val, predictions, target_names=['no car', 'car']))
