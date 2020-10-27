import numpy as np
from scipy import signal


def MAV(x, nsignals, nwin):
    MAV_feat = np.sum(abs(x), axis=0)/len(x)
    MAV_feat = MAV_feat.reshape(nwin, nsignals)
    return MAV_feat


def ZC(x, threshold, winsize, nsignals, nwin):
    y = (x > threshold).astype(int) - (x < -threshold).astype(int)
    b = np.exp(-np.arange(1, winsize//2+1)).reshape(-1, 1)
    z = signal.convolve2d(y, b, mode='same')

    z = (z > 0).astype(int) - (z < -0).astype(int)
    ZC_feat = np.sum((abs(np.diff(z, axis=0)) == 2), axis=0)
    ZC_feat = ZC_feat.reshape(nwin, nsignals)
    return ZC_feat


def WL(x, nsignals, nwin):
    WL_feat = np.sum(abs(np.diff(x, axis=0)), axis=0)
    WL_feat = WL_feat.reshape(nwin, nsignals)
    return WL_feat




