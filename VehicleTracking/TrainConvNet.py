import pickle

from keras.layers import Conv2D, Flatten, Dense, Dropout, MaxPooling2D
from keras.models import Sequential
import numpy as np
import matplotlib.pyplot as plt

from ImageUtils import convert_cspace

with open('../data/data.p', 'rb') as f:
    data = pickle.load(f)

X_train, y_train = data['train']
X_val, y_val = data['val']

for i in range(X_val.shape[0]):
    X_train[i] = convert_cspace(X_train[i], 'YUV')

for i in range(X_val.shape[0]):
    X_val[i] = convert_cspace(X_val[i], 'YUV')

X_train = X_train.astype(np.float32) / 255
X_val = X_val.astype(np.float32) / 255

# plt.imshow(X_train[0])
# plt.show()

model = Sequential()
model.add(Conv2D(64, 5, 5, subsample=(3, 3), activation='relu', input_shape=(64, 64, 3)))
model.add(Conv2D(128, 5, 5, subsample=(3, 3), activation='relu'))
model.add(Flatten())
model.add(Dropout(0.2))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))

model.summary()

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

history = model.fit(X_train, y_train,
                    batch_size=128,
                    nb_epoch=30,
                    validation_data=(X_val, y_val),
                    shuffle=True)

train_eval = model.evaluate(X_train, y_train, batch_size=32)
val_eval = model.evaluate(X_val, y_val, batch_size=32)

print()
print('Train Accuracy = ', train_eval[1])
print('Validation Accuracy = ', val_eval[1])
