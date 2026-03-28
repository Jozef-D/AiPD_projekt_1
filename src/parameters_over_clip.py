import numpy as np
import pandas as pd

from src import compute_all_params


def parameters_over_clip(samples, sample_rate, frame_ms=20, overlap=0.5):
    params = compute_all_params(samples, sample_rate, frame_ms, overlap)
    if params is None:
        return None

    ste_arr = params['ste']
    zcr_arr = params['zcr']
    n = params['num_frames']

    ste_mean = 0.0
    for v in ste_arr:
        ste_mean += v
    ste_mean /= n

    lster_count = 0
    for v in ste_arr:
        if v < 0.5 * ste_mean:
            lster_count += 1
    lster = lster_count / n

    zcr_mean = 0.0
    for v in zcr_arr:
        zcr_mean += v
    zcr_mean /= n

    hzcrr_count = 0
    for v in zcr_arr:
        if v > 1.5 * zcr_mean:
            hzcrr_count += 1
    hzcrr = hzcrr_count / n

    zstd_sum = 0.0
    for v in zcr_arr:
        zstd_sum += (v - zcr_mean) ** 2
    zstd = (zstd_sum / n) ** 0.5

    total_energy = 0.0
    for v in ste_arr:
        total_energy += v

    clip_entropy = 0.0
    if total_energy > 0:
        for v in ste_arr:
            p = v / total_energy
            if p > 0:
                clip_entropy -= p * np.log2(p)

    return {
        'lster':          lster,
        'hzcrr':          hzcrr,
        'zstd':           zstd,
        'energy_entropy': clip_entropy,
    }