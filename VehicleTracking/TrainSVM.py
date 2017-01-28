import pickle
import time

import numpy as np
from FeatureExtractionPipeline import HogExtractor, SpatialBining, ColorHistogram, ColorSpaceConverter, \
    OptionalBranch, OptionalPCA, AcceptEmptyMinMaxScaler, HogExtractorOpenCV
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.svm import LinearSVC
from sklearn.utils import shuffle

N_JOBS = 4
SAMPLE_SIZE = 10

if __name__ == '__main__':
    np.random.seed(0)

    with open('../data/data_adj.p', 'rb') as f:
        data = pickle.load(f)

    with open('../data/data_hnm.p', 'rb') as f:
        data_hnm = pickle.load(f)

    X_train, y_train = data['train']
    X_hnm, y_hnm = data_hnm
    X_val, y_val = data['val']

    sample = np.random.choice(len(y_train), SAMPLE_SIZE, replace=False)
    X_train = X_train[sample]
    y_train = y_train[sample]

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
    hog_extractor = HogExtractorOpenCV()
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
    # params = [{'clf__C': [1],
    #
    #            'features__sb__sb_optional__use': [True],
    #            'features__sb__sb_csc__cspace': ['RGB', 'HLS', 'YCrCb', 'LAB'],
    #            'features__sb__spatial_bining__bins': [32],
    #
    #            'features__chist__chist_optional__use': [True],
    #            'features__chist__chist_csc__cspace': ['RGB', 'HLS', 'YCrCb', 'LAB'],
    #            'features__chist__color_histogram__bins': [32],
    #            'features__chist__chist_pca__n_components': [None, 96],
    #
    #            'features__hog__hog_csc__cspace': ['RGB', 'HLS', 'YCrCb', 'LAB'],
    #            'features__hog__hog_extractor__orient': [9, 12, 18],
    #            'features__hog__hog_extractor__pix_per_cell': [8],
    #            'features__hog__hog_extractor__cells_per_block': [2, 3],
    #            'features__hog__hog_pca__n_components': [None, 96]},
    #
    #           {'clf__C': [1],
    #
    #            'features__sb__sb_optional__use': [False],
    #
    #            'features__chist__chist_optional__use': [True],
    #            'features__chist__chist_csc__cspace': ['RGB', 'HLS', 'YCrCb', 'LAB'],
    #            'features__chist__color_histogram__bins': [32],
    #            'features__chist__chist_pca__n_components': [None, 96],
    #
    #            'features__hog__hog_csc__cspace': ['RGB', 'HLS', 'YCrCb', 'LAB'],
    #            'features__hog__hog_extractor__orient': [9, 12, 18],
    #            'features__hog__hog_extractor__pix_per_cell': [8],
    #            'features__hog__hog_extractor__cells_per_block': [2, 3],
    #            'features__hog__hog_pca__n_components': [None, 96]},
    #
    #           {'clf__C': [1],
    #
    #            'features__sb__sb_optional__use': [True],
    #            'features__sb__sb_csc__cspace': ['RGB', 'HLS', 'YCrCb', 'LAB'],
    #            'features__sb__spatial_bining__bins': [32],
    #
    #            'features__chist__chist_optional__use': [False],
    #
    #            'features__hog__hog_csc__cspace': ['RGB', 'HLS', 'YCrCb', 'LAB'],
    #            'features__hog__hog_extractor__orient': [9, 12, 18],
    #            'features__hog__hog_extractor__pix_per_cell': [8],
    #            'features__hog__hog_extractor__cells_per_block': [2, 3],
    #            'features__hog__hog_pca__n_components': [None, 96]},
    #
    #           {'clf__C': [1],
    #
    #            'features__sb__sb_optional__use': [False],
    #
    #            'features__chist__chist_optional__use': [False],
    #
    #            'features__hog__hog_csc__cspace': ['RGB', 'HLS', 'YCrCb', 'LAB'],
    #            'features__hog__hog_extractor__orient': [9, 12, 18],
    #            'features__hog__hog_extractor__pix_per_cell': [8],
    #            'features__hog__hog_extractor__cells_per_block': [2, 3],
    #            'features__hog__hog_pca__n_components': [None, 96]}
    #           ]

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
               'features__hog__hog_extractor__layout': [(16, 8, 8), (24, 10, 8), (24, 8, 8)],
               'features__hog__hog_extractor__win_sigma': [1, 2, 4, 7],
               'features__hog__hog_extractor__gamma_correction': [True, False],
               'features__hog__hog_pca__n_components': [None, 96]},

              {'clf__C': [1],

               'features__sb__sb_optional__use': [False],

               'features__chist__chist_optional__use': [True],
               'features__chist__chist_csc__cspace': ['RGB', 'HLS', 'YCrCb', 'LAB'],
               'features__chist__color_histogram__bins': [32],
               'features__chist__chist_pca__n_components': [None, 96],

               'features__hog__hog_csc__cspace': ['RGB', 'HLS', 'YCrCb', 'LAB'],
               'features__hog__hog_extractor__orient': [9, 12, 18],
               'features__hog__hog_extractor__layout': [(16, 8, 8), (24, 10, 8), (24, 8, 8)],
               'features__hog__hog_extractor__win_sigma': [1, 2, 4],
               'features__hog__hog_extractor__gamma_correction': [True, False],
               'features__hog__hog_pca__n_components': [None, 96]},

              {'clf__C': [1],

               'features__sb__sb_optional__use': [True],
               'features__sb__sb_csc__cspace': ['RGB', 'HLS', 'YCrCb', 'LAB'],
               'features__sb__spatial_bining__bins': [32],

               'features__chist__chist_optional__use': [False],

               'features__hog__hog_csc__cspace': ['RGB', 'HLS', 'YCrCb', 'LAB'],
               'features__hog__hog_extractor__orient': [9, 12, 18],
               'features__hog__hog_extractor__layout': [(16, 8, 8), (24, 10, 8), (24, 8, 8)],
               'features__hog__hog_extractor__win_sigma': [1, 2,  4, 7],
               'features__hog__hog_extractor__gamma_correction': [True, False],
               'features__hog__hog_pca__n_components': [None, 96]},

              {'clf__C': [1],

               'features__sb__sb_optional__use': [False],

               'features__chist__chist_optional__use': [False],

               'features__hog__hog_csc__cspace': ['RGB', 'HLS', 'YCrCb', 'LAB'],
               'features__hog__hog_extractor__orient': [9, 12, 18],
               'features__hog__hog_extractor__layout': [(16, 8, 8), (24, 10, 8), (24, 8, 8)],
               'features__hog__hog_extractor__win_sigma': [1, 2, 4, 7],
               'features__hog__hog_extractor__gamma_correction': [True, False],
               'features__hog__hog_pca__n_components': [None, 96]},
              ]

    # Keep best params from feature extraction grid search and vary C
    # params = {'clf__C': [1e-3, 1e-2, 1e-1, 1, 1e1, 1e2, 1e3, 1e4, 1e5, 1e6, 1e7, 1e8, 1e9, 1e10],
    #           'features__chist__chist_csc__cspace': ['HLS'],
    #           'features__chist__chist_optional__use': [True],
    #           'features__chist__chist_pca__n_components': [None],
    #           'features__chist__color_histogram__bins': [32],
    #           'features__hog__hog_csc__cspace': ['LAB'],
    #           'features__hog__hog_extractor__cells_per_block': [2],
    #           'features__hog__hog_extractor__orient': [18],
    #           'features__hog__hog_extractor__pix_per_cell': [8],
    #           'features__hog__hog_pca__n_components': [None],
    #           'features__sb__sb_csc__cspace': ['LAB'],
    #           'features__sb__sb_optional__use': [True],
    #           'features__sb__spatial_bining__bins': [32]}

    # params = {'clf__C': [1],
    #           'features__chist__chist_csc__cspace': ['HLS'],
    #           'features__chist__chist_optional__use': [True],
    #           'features__chist__chist_pca__n_components': [None],
    #           'features__chist__color_histogram__bins': [32],
    #           'features__hog__hog_csc__cspace': ['LAB'],
    #           'features__hog__hog_extractor__cells_per_block': [2],
    #           'features__hog__hog_extractor__orient': [18],
    #           'features__hog__hog_extractor__pix_per_cell': [8],
    #           'features__hog__hog_pca__n_components': [None],
    #           'features__sb__sb_csc__cspace': ['LAB'],
    #           'features__sb__sb_optional__use': [True],
    #           'features__sb__spatial_bining__bins': [32]}

    # params = {'features__chist__color_histogram__bins': [32],
    #           'features__chist__chist_optional__use': [True],
    #           'features__sb__spatial_bining__bins': [32],
    #           'features__hog__hog_extractor__pix_per_cell': [8],
    #           'features__hog__hog_extractor__cells_per_block': [2],
    #           'features__sb__sb_csc__cspace': ['LAB'],
    #           'features__chist__chist_pca__n_components': [96],
    #           'features__chist__chist_csc__cspace': ['YCrCb'],
    #           'features__hog__hog_csc__cspace': ['LAB'],
    #           'features__hog__hog_pca__n_components': [96],
    #           'clf__C': [1],
    #           'features__hog__hog_extractor__orient': [18],
    #           'features__sb__sb_optional__use': [True]}

    cls = GridSearchCV(pipeline, params, n_jobs=N_JOBS, verbose=3, scoring='roc_auc')

    print('Begin training')
    t = time.time()
    cls.fit(X_train, y_train)
    t2 = time.time()
    print('Finished training after ', t2 - t, ' seconds')

    with open('../models/svm_test.p', 'wb') as f:
        pickle.dump(cls.best_estimator_, f)

    # with open('../models/gridsearch_hnm.p', 'wb') as f:
    #     pickle.dump(cls, f)

    print('Best params: ', cls.best_params_)
    print('Best auc roc score: ', cls.best_score_)
    print('Train Accuracy of SVC = ', cls.best_estimator_.score(X_train, y_train))
    print('Validation Accuracy of SVC = ', cls.best_estimator_.score(X_val, y_val))

    predictions = cls.predict(X_val)
    print(classification_report(y_val, predictions, target_names=['no car', 'car']))
