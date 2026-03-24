import numpy as np


def classify_voiced_unvoiced(params, f0_array, energy_ratio=0.1, zcr_threshold=0.3):
    vol = params['volume']
    zcr = params['zcr']
    n = params['num_frames']

    max_vol = max(np.max(vol), 1e-10)
    silence_thresh = energy_ratio * max_vol

    labels = []
    for i in range(n):
        if vol[i] < silence_thresh:
            labels.append('S')
        elif f0_array[i] > 0 and zcr[i] < zcr_threshold:
            labels.append('V')
        else:
            labels.append('U')

    return labels
