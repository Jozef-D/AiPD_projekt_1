import numpy as np


def detect_silence(params, threshold_ratio=0.05):
    vol = params['volume']
    max_vol = max(np.max(vol), 1e-10)
    threshold = threshold_ratio * max_vol
    mask = vol < threshold
    return mask, threshold
