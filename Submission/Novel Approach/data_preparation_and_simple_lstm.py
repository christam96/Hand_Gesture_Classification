# For moving through folders and paths
import os
from pathlib import Path

# Useful libraries
import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import LSTM
from sklearn.metrics import accuracy_score


# Assign path
rt = Path(os.getcwd())  # Get current working directory
rt = rt.parents[0]  # Go one level up in the current working directory

X_train = []
X_cv = []
X_tst = []

y_train = []
y_cv = []
y_tst = []

# Load data
for gest in np.arange(0, 4):
    file2read = 'Data/Reformatted/gesture_' + str(gest) + '.csv'
    df = np.array(pd.read_csv(rt/file2read))

    # Get total number of windows
    win_size = 40  # Myo sampling rate is 200 Hz thus, 200 ms of data equals to 40 samples.
    numwin = df.shape[0] // win_size
    nsignals = 8  # No. of channels. The Myo armband has 8

    dat_X = df[:, :nsignals]
    y = df[:numwin, -1]
    dat_X = np.reshape(dat_X, (numwin, win_size, nsignals))

    #################################################
    # Create txt files with the index order for splitting the data
    # Not to be used again
    #################################################
    # a = np.arange(numwin)
    # np.random.shuffle(a)
    # a = a.reshape(1, -1)
    #
    # a_file = open('index_gest_' + str(gest) + '.txt', "w")
    # for row in a:
    #     np.savetxt(a_file, row)
    #
    # a_file.close()
    #################################################
    original_array = np.loadtxt('index_gest_' + str(gest) + '.txt').reshape(1, numwin).astype(int)
    a = tuple(original_array)

    dat_X[:] = dat_X[a]

    # Train, CV, test splits (60/20/20)
    tst_size = math.floor(numwin*0.2)
    tr_X = dat_X[:numwin-tst_size*2]
    cv_X = dat_X[numwin-(tst_size*2):numwin-(tst_size*2)+tst_size]
    tst_X = dat_X[numwin-(tst_size*2)+tst_size:]

    X_train = np.append(X_train, tr_X)
    X_cv = np.append(X_cv, cv_X)
    X_tst = np.append(X_tst, tst_X)
    y_train = np.append(y_train, y[:tr_X.shape[0]])
    y_cv = np.append(y_cv, y[:cv_X.shape[0]])
    y_tst = np.append(y_tst, y[:tst_X.shape[0]])


# Reshape data and reshuffle it
# Training data
aux = np.arange(y_train.shape[0])
np.random.shuffle(aux)
X_train = X_train.reshape(-1, win_size, nsignals)
X_train[:] = X_train[aux]
y_train[:] = y_train[aux]

# Cross_validation data
aux = np.arange(y_cv.shape[0])
np.random.shuffle(aux)
X_cv = X_cv.reshape(-1, win_size, nsignals)
X_cv[:] = X_cv[aux]
y_cv[:] = y_cv[aux]

# Testing data
aux = np.arange(y_tst.shape[0])
np.random.shuffle(aux)
X_tst = X_tst.reshape(-1, win_size, nsignals)
X_tst[:] = X_tst[aux]
y_tst[:] = y_tst[aux]

# one hot encode the outputs for the softmax layer
y_train = to_categorical(y_train)
y_cv = to_categorical(y_cv)
y_tst = to_categorical(y_tst)

# LSTM Model
model = Sequential()

# Set the value 'return_sequences' to true if want to add another LSTM layer
# When adding another LSTM layer, we don't have to specify the input shape
model.add(LSTM(units=50, return_sequences=False, input_shape=(X_train.shape[1], X_train.shape[2])))

# The dropout layer is optional
model.add(Dropout(0.2))

# Non LSTM layer (normal hidden layer)
model.add(Dense(100, activation='relu'))
model.add(Dense(4, activation='softmax'))

# Compile the model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Fit the model
batch_size = X_train.shape[0]  # Max batch_size = number of training samples
history = model.fit(X_train, y_train, epochs=500, batch_size=batch_size, verbose=2,
                    validation_data=(X_cv, y_cv), shuffle=False)

# Plot training & validation accuracy values
plt.figure(figsize=(12, 5), dpi=100)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

# Plot training & validation loss values
plt.figure(figsize=(12, 5), dpi=100)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

# Predict the test data
y_pred = model.predict(X_tst)

# Reverse one-hot encoding
y_pred = [np.argmax(y, axis=None, out=None) for y in y_pred]
y_true = [np.argmax(y, axis=None, out=None) for y in y_tst]

acc = accuracy_score(y_true, y_pred)
print(acc)
