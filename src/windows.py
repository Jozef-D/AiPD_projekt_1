import numpy as np


def rectangular(n):
    w = np.zeros(n)
    for i in range(n):
        w[i] = 1.0
    return w


def triangular(n):
    w = np.zeros(n)
    for i in range(n):
        w[i] = 1.0 - abs((i - (n - 1) / 2.0) / ((n - 1) / 2.0))
    return w


def hamming(n):
    w = np.zeros(n)
    for i in range(n):
        w[i] = 0.54 - 0.46 * np.cos(2 * np.pi * i / (n - 1))
    return w


def hann(n):
    w = np.zeros(n)
    for i in range(n):
        w[i] = 0.5 * (1 - np.cos(2 * np.pi * i / (n - 1)))
    return w


def blackman(n):
    w = np.zeros(n)
    for i in range(n):
        w[i] = (0.42
                - 0.5 * np.cos(2 * np.pi * i / (n - 1))
                + 0.08 * np.cos(4 * np.pi * i / (n - 1)))
    return w


WINDOWS = {
    'Prostokątne': rectangular,
    'Trójkątne': triangular,
    'Hamminga': hamming,
    'von Hanna': hann,
    'Blackmana': blackman,
}

def apply_window(signal, window_name):
    w = WINDOWS[window_name](len(signal))
    return signal * w, w
