import pickle
import time

import numpy as np
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.svm import LinearSVC

from FeatureExtractionPipeline import HogExtractor, SpatialBining, ColorHistogram, ColorSpaceConverter, \
    OptionalBranch, OptionalPCA, AcceptEmptyMinMaxScaler

N_JOBS = 3
SAMPLE_SIZE = 50

if __name__ == '__main__':
    np.random.seed(0)

    with open('../data/data.p', 'rb') as f:
        data = pickle.load(f)

    # with open('../data/data_hnm.p', 'rb') as f:
    #     data_hnm = pickle.load(f)

    # X_train, y_train = data_hnm
    X_val, y_val = data['val']

    X_train, y_train = data['train']
    sample = np.random.choice(len(y_train), SAMPLE_SIZE, replace=False)
    X_train = X_train[sample]
    y_train = y_train[sample]
    print('Train size:', len(y_train))

    # Concat train and validation set since we will use K-fold CV
    # X_train = np.concatenate([X_train, X_val])
    # y_train = np.concatenate([y_train, y_val])
    #
    # X_train, y_train = shuffle(X_train, y_train, random_state=7)

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

    # Search parameter
    params = [{'clf__C': [1],

               'features__sb__sb_optional__use': [True],
               'features__sb__sb_csc__cspace': ['RGB', 'HLS', 'YCrCb', 'LAB'],
               'features__sb__spatial_bining__bins': [32],

               'features__chist__chist_optional__use': [True],
               'features__chist__chist_csc__cspace': ['RGB', 'HLS', 'YCrCb', 'LAB'],
               'features__chist__color_histogram__bins': [32],
               'features__chist__chist_pca__n_components': [None, 96],

               'features__hog__hog_csc__cspace': ['RGB', 'HLS', 'YCrCb', 'LAB'],
               'features__hog__hog_extractor__orient': [9, 12, 18],
               'features__hog__hog_extractor__pix_per_cell': [8],
               'features__hog__hog_extractor__cells_per_block': [2, 3],
               'features__hog__hog_pca__n_components': [None, 96]},

              {'clf__C': [1],

               'features__sb__sb_optional__use': [False],

               'features__chist__chist_optional__use': [True],
               'features__chist__chist_csc__cspace': ['RGB', 'HLS', 'YCrCb', 'LAB'],
               'features__chist__color_histogram__bins': [32],
               'features__chist__chist_pca__n_components': [None, 96],

               'features__hog__hog_csc__cspace': ['RGB', 'HLS', 'YCrCb', 'LAB'],
               'features__hog__hog_extractor__orient': [9, 12, 18],
               'features__hog__hog_extractor__pix_per_cell': [8],
               'features__hog__hog_extractor__cells_per_block': [2, 3],
               'features__hog__hog_pca__n_components': [None, 96]},

              {'clf__C': [1],

               'features__sb__sb_optional__use': [True],
               'features__sb__sb_csc__cspace': ['RGB', 'HLS', 'YCrCb', 'LAB'],
               'features__sb__spatial_bining__bins': [32],

               'features__chist__chist_optional__use': [False],

               'features__hog__hog_csc__cspace': ['RGB', 'HLS', 'YCrCb', 'LAB'],
               'features__hog__hog_extractor__orient': [9, 12, 18],
               'features__hog__hog_extractor__pix_per_cell': [8],
               'features__hog__hog_extractor__cells_per_block': [2, 3],
               'features__hog__hog_pca__n_components': [None, 96]},

              {'clf__C': [1],

               'features__sb__sb_optional__use': [False],

               'features__chist__chist_optional__use': [False],

               'features__hog__hog_csc__cspace': ['RGB', 'HLS', 'YCrCb', 'LAB'],
               'features__hog__hog_extractor__orient': [9, 12, 18],
               'features__hog__hog_extractor__pix_per_cell': [8],
               'features__hog__hog_extractor__cells_per_block': [2, 3],
               'features__hog__hog_pca__n_components': [None, 96]}
              ]

    cls = GridSearchCV(pipeline, params, n_jobs=3, verbose=3, scoring='roc_auc')

    print('Begin training')
    t = time.time()
    cls.fit(X_train, y_train)
    t2 = time.time()
    print('Finished training after ', t2 - t, ' seconds')

    with open('svm_best.p', 'wb') as f:
        pickle.dump(cls.best_estimator_, f)

    with open('gridsearch.p', 'wb') as f:
        pickle.dump(cls, f)

    print('Best params: ', cls.best_params_)
    print('Train Accuracy of SVC = ', cls.best_estimator_.score(X_train, y_train))
    print('Validation Accuracy of SVC = ', cls.best_estimator_.score(X_val, y_val))

    predictions = cls.predict(X_val)
    print(classification_report(y_val, predictions, target_names=['no car', 'car']))
