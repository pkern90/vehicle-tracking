import pickle
import time

from sklearn.model_selection import GridSearchCV
from sklearn.svm import LinearSVC

from ImageUtils import extract_features

with open('../data/data.p', 'rb') as f:
    data = pickle.load(f)

X_train, y_train = data['train']
X_val, y_val = data['val']

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

X_val = extract_features(X_val, cspace=cspace, orient=orient, pix_per_cell=pix_per_cell,
                         cells_per_block=cells_per_block, normalize=True)

parameters = [{'loss': ['squared_hinge'],
               'C': [1e-3, 1e-2, 1e-1, 1, 1e1, 1e2],
               'dual': [False],
               'penalty': ['l1', 'l2'],
               'tol': [1e-4, 1e-3, 1e-2, 1e-1]},
              {'loss': ['hinge'],
               'C': [1e-3, 1e-2, 1e-1, 1, 1e1, 1e2],
               'dual': [True],
               'penalty': ['l2'],
               'tol': [1e-4, 1e-3, 1e-2, 1e-1]}]

svc = LinearSVC(random_state=7)
cls = GridSearchCV(svc, parameters, n_jobs=4)

print('Begin training')
t = time.time()
cls.fit(X_train, y_train)
t2 = time.time()
print('Finished training after ', t2 - t, ' seconds')

print('Best params: ', cls.best_params_)
print('Train Accuracy of SVC = ', cls.score(X_train, y_train))
print('Validation Accuracy of SVC = ', cls.score(X_val, y_val))

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

with open('../models/svm_hog.p', 'wb') as f:
    pickle.dump(cls.best_estimator_, f)
