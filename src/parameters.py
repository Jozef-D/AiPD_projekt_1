import numpy as np
import pandas as pd

def split_into_frames(samples, sample_rate, frame_ms=20, overlap=0.5):
    frame_len = max(1, int(sample_rate * frame_ms / 1000))
    hop = max(1, int(frame_len * (1 - overlap)))
    num_frames = max(0, (len(samples) - frame_len) // hop + 1)

    frames = []
    times = np.zeros(num_frames)

    for i in range(num_frames):
        start = i * hop
        end = start + frame_len
        frames.append(samples[start:end])
        times[i] = (start + end) / 2.0 / sample_rate

    return frames, times, frame_len, hop


def volume(frame):
    total = 0.0
    for x in frame:
        total += x * x
    return (total / len(frame)) ** 0.5


def short_time_energy(frame):
    total = 0.0
    for x in frame:
        total += x * x
    return total / len(frame)


def zero_crossing_rate(frame):
    crossings = 0
    for i in range(1, len(frame)):
        if (frame[i] >= 0 and frame[i - 1] < 0) or \
           (frame[i] < 0 and frame[i - 1] >= 0):
            crossings += 1
    return crossings / len(frame)

def silent_ratio(frame, volume_threshold, zcr_threshold):
    vol = volume(frame)
    zcr = zero_crossing_rate(frame)

    if vol < volume_threshold and zcr < zcr_threshold:
        return 1.0
    return 0.0

def energy_entropy(frame, n_subframes=10):
    frame_len = len(frame)
    subframe_len = frame_len // n_subframes
    if subframe_len == 0:
        return 0.0

    total_energy = 0.0
    for x in frame:
        total_energy += x * x

    if total_energy == 0.0:
        return 0.0

    entropy = 0.0
    for j in range(n_subframes):
        start = j * subframe_len
        end = start + subframe_len
        sub_energy = 0.0
        for x in frame[start:end]:
            sub_energy += x * x
        p = sub_energy / total_energy
        if p > 0:
            entropy -= p * np.log2(p)

    return entropy


def compute_all_params(samples, sample_rate, frame_ms=20, overlap=0.5,
                       vol_threshold=None, zcr_threshold=None):
    frames, times, frame_len, hop = split_into_frames(
        samples, sample_rate, frame_ms, overlap
    )
    n = len(frames)
    if n == 0:
        return None

    vol = np.zeros(n)
    ste = np.zeros(n)
    zcr = np.zeros(n)
    ee  = np.zeros(n)
    sr  = np.zeros(n)

    for i, frame in enumerate(frames):
        vol[i] = volume(frame)
        ste[i] = short_time_energy(frame)
        zcr[i] = zero_crossing_rate(frame)
        ee[i]  = energy_entropy(frame)
        if vol_threshold is not None and zcr_threshold is not None:
            sr[i] = silent_ratio(frame, vol_threshold, zcr_threshold)

    return {
        'frame_times':    times,
        'volume':         vol,
        'ste':            ste,
        'zcr':            zcr,
        'energy_entropy': ee,
        'silent_ratio':   sr,
        'frame_len':      frame_len,
        'hop':            hop,
        'num_frames':     n,
    }

def parameters_to_csv(samples: np.ndarray, sample_rate, frame_ms=20, overlap=0.50,
                       f0: np.ndarray = None,
                       vol_threshold: float = None,
                       zcr_threshold: float = None):
    params_dict = compute_all_params(
        samples, sample_rate, frame_ms, overlap,
        vol_threshold, zcr_threshold
    )
    df = pd.DataFrame({
        'time':           params_dict['frame_times'],
        'volume':         params_dict['volume'],#
        'ste':            params_dict['ste'],#
        'zcr':            params_dict['zcr'], #
        'energy_entropy': params_dict['energy_entropy'],
        'silent_ratio':   params_dict['silent_ratio'],
    })
    if f0 is not None:
        df['f0'] = f0
    return df