import numpy as np
from .windows import apply_window, WINDOWS


def compute_fft(signal, sample_rate, window_name='Prostokątne'):
    windowed, window = apply_window(signal, window_name)
    n = len(windowed)
    fft_result = np.fft.fft(windowed)
    magnitude = np.abs(fft_result[:n // 2])
    magnitude_db = 20 * np.log10(magnitude + 1e-10)
    freqs = np.arange(n // 2) * sample_rate / n
    return freqs, magnitude, magnitude_db, windowed, window


def compute_fft_for_frame(samples, sample_rate, frame_index, frame_len, hop,
                          window_name='Prostokątne'):
    start = frame_index * hop
    end = start + frame_len
    if end > len(samples):
        end = len(samples)
    frame = samples[start:end]
    return compute_fft(frame, sample_rate, window_name)


def compute_spectrogram(samples, sample_rate, frame_ms=20, overlap=0.5,
                        window_name='Hamminga'):
    frame_len = max(1, int(sample_rate * frame_ms / 1000))
    hop = max(1, int(frame_len * (1 - overlap)))
    num_frames = max(0, (len(samples) - frame_len) // hop + 1)

    if num_frames == 0:
        return None, None, None

    n_fft = frame_len
    n_bins = n_fft // 2

    spectrogram = np.zeros((n_bins, num_frames))
    window_func = WINDOWS[window_name]
    w = window_func(frame_len)

    for i in range(num_frames):
        start = i * hop
        end = start + frame_len
        frame = samples[start:end].copy()

        for j in range(frame_len):
            frame[j] = frame[j] * w[j]

        fft_result = np.fft.fft(frame)
        mag = np.abs(fft_result[:n_bins])
        spectrogram[:, i] = 20 * np.log10(mag + 1e-10)

    times = np.arange(num_frames) * hop / sample_rate
    freqs = np.arange(n_bins) * sample_rate / n_fft

    return spectrogram, times, freqs


def compute_cepstrum(frame, sample_rate):
    n = len(frame)
    fft_result = np.fft.fft(frame)
    log_spectrum = np.log(np.abs(fft_result) + 1e-10)
    cepstrum = np.fft.ifft(log_spectrum).real
    quefrency = np.arange(n) / sample_rate
    return cepstrum, quefrency


def compute_cepstral_f0(samples, sample_rate, frame_ms=20, overlap=0.5,
                        window_name='Hamminga', f_min=50, f_max=500):
    frame_len = max(1, int(sample_rate * frame_ms / 1000))
    hop = max(1, int(frame_len * (1 - overlap)))
    num_frames = max(0, (len(samples) - frame_len) // hop + 1)

    if num_frames == 0:
        return None, None

    window_func = WINDOWS[window_name]
    w = window_func(frame_len)

    q_min = int(sample_rate / f_max)
    q_max = min(frame_len - 1, int(sample_rate / f_min))

    f0_array = np.zeros(num_frames)
    times = np.zeros(num_frames)

    for i in range(num_frames):
        start = i * hop
        end = start + frame_len
        frame = samples[start:end].copy()

        for j in range(frame_len):
            frame[j] = frame[j] * w[j]

        times[i] = (start + end) / 2.0 / sample_rate

        cepstrum, _ = compute_cepstrum(frame, sample_rate)

        if q_min >= q_max or q_max >= len(cepstrum):
            f0_array[i] = 0.0
            continue

        best_q = q_min
        best_val = -1e30
        for q in range(q_min, q_max + 1):
            if cepstrum[q] > best_val:
                best_val = cepstrum[q]
                best_q = q

        if best_val > 0.01:
            f0_array[i] = sample_rate / best_q
        else:
            f0_array[i] = 0.0

    return f0_array, times
