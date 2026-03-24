import numpy as np


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


def compute_all_params(samples, sample_rate, frame_ms=20, overlap=0.5):
    frames, times, frame_len, hop = split_into_frames(
        samples, sample_rate, frame_ms, overlap
    )
    n = len(frames)
    if n == 0:
        return None

    vol = np.zeros(n)
    ste = np.zeros(n)
    zcr = np.zeros(n)

    for i, frame in enumerate(frames):
        vol[i] = volume(frame)
        ste[i] = short_time_energy(frame)
        zcr[i] = zero_crossing_rate(frame)

    return {
        'frame_times': times,
        'volume': vol,
        'ste': ste,
        'zcr': zcr,
        'frame_len': frame_len,
        'hop': hop,
        'num_frames': n,
    }
