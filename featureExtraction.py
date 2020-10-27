# For moving through folders and paths
import os
from pathlib import Path

# Useful libraries
import pandas as pd
import numpy as np
import timeDomainFeatures as tdf

# Assign path
rt = Path(os.getcwd())  # Get current working directory
rt = rt.parents[0]  # Go one level up in the current working directory

# Read data
df = pd.read_csv(rt/'Data/Reformatted/gesture_0.csv')
df_gest = df['Gesture']

df = np.array(df.drop(['Gesture'], axis=1))

# Get total number of windows
win_size = 40  # Myo sampling rate is 200 Hz thus, 200 ms of data equals to 40 samples.
numwin = df.shape[0]//win_size

idx_dum = np.arange(0, df.size).reshape(df.shape[0], df.shape[1])  # Column wise reshaping
out = idx_dum[np.arange(0, df.shape[0], win_size), :].reshape(1, numwin*df.shape[1])  # row wise reshaping
temp = np.arange(0, win_size*8, 8).reshape(-1, 1)
idxmat = out+temp

# Data reformatted. Every 8 columns represent a window for each channel. Each row represent the first sample of
# the window
vectdata = np.take(df, idxmat)

"""
Start computing features
"""
# Number of features extracted from the signal
feats = ['MAV', 'WL', 'ZC', 'AR']
nFeats = len(feats)  # MAV, WL, ZC, AR
arOrder = 4

temp = nFeats + arOrder-1

# Preallocate memory
featMat = np.empty([numwin, 8*temp])
featMat[:] = np.NaN
idx = 0
row = np.arange(0,numwin)

for ii in feats:
    if ii == 'MAV':
        aux = tdf.MAV(vectdata, 8, numwin)
        col = np.arange(idx, idx + aux.shape[1])
        featMat[np.ix_(row, col)] = aux
        idx = idx + aux.shape[1]
    elif ii == 'WL':
        aux = tdf.WL(vectdata, 8, numwin)
        col = np.arange(idx, idx + aux.shape[1])
        featMat[np.ix_(row, col)] = aux
        idx = idx + aux.shape[1]
    elif ii == 'ZC':
        aux = tdf.ZC(vectdata, 0.02, win_size, 8, numwin)
        col = np.arange(idx, idx + aux.shape[1])
        featMat[np.ix_(row, col)] = aux
        idx = idx + aux.shape[1]
    else:
        print("no function yet for AR")

df_feat = pd.concat([pd.DataFrame(featMat), df_gest], axis=1)
