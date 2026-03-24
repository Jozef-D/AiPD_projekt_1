import numpy as np


def autocorrelation_f0(frame, sample_rate, f_min=50, f_max=500):
    n = len(frame)
    tau_min = max(1, int(sample_rate / f_max))
    tau_max = min(n - 1, int(sample_rate / f_min))

    if tau_min >= tau_max:
        return 0.0

    best_tau = tau_min
    best_r = -1e30

    for tau in range(tau_min, tau_max + 1):
        r = 0.0
        for i in range(n - tau):
            r += frame[i] * frame[i + tau]
        if r > best_r:
            best_r = r
            best_tau = tau

    r0 = 0.0
    for i in range(n):
        r0 += frame[i] * frame[i]

    if r0 < 1e-10 or best_r / r0 < 0.2:
        return 0.0

    return sample_rate / best_tau


def amdf_f0(frame, sample_rate, f_min=50, f_max=500):
    n = len(frame)
    tau_min = max(1, int(sample_rate / f_max))
    tau_max = min(n - 1, int(sample_rate / f_min))

    if tau_min >= tau_max:
        return 0.0

    best_tau = tau_min
    best_amdf = 1e30

    for tau in range(tau_min, tau_max + 1):
        d = 0.0
        count = n - tau
        for i in range(count):
            diff = frame[i] - frame[i + tau]
            if diff < 0:
                diff = -diff
            d += diff
        d /= count

        if d < best_amdf:
            best_amdf = d
            best_tau = tau

    avg_energy = 0.0
    for i in range(n):
        v = frame[i]
        if v < 0:
            v = -v
        avg_energy += v
    avg_energy /= n

    if avg_energy < 1e-10 or best_amdf / avg_energy > 0.5:
        return 0.0

    return sample_rate / best_tau


def compute_pitch(frames, sample_rate, method='autocorrelation', f_min=50, f_max=500):
    n = len(frames)
    f0 = np.zeros(n)

    func = autocorrelation_f0 if method == 'autocorrelation' else amdf_f0

    for i, frame in enumerate(frames):
        f0[i] = func(frame, sample_rate, f_min, f_max)

    return f0
