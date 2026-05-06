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
    end = min(start + frame_len, len(samples))
    frame = samples[start:end]
    return compute_fft(frame, sample_rate, window_name)


def compute_spectrogram(samples, sample_rate, frame_ms=20, overlap=0.5,
                        window_name='Hamminga'):
    frame_len = max(1, int(sample_rate * frame_ms / 1000))
    hop = max(1, int(frame_len * (1 - overlap)))
    num_frames = max(0, (len(samples) - frame_len) // hop + 1)

    if num_frames == 0:
        return None, None, None

    n_bins = frame_len // 2
    w = WINDOWS[window_name](frame_len)
    spectrogram = np.zeros((n_bins, num_frames))

    for i in range(num_frames):
        start = i * hop
        frame = samples[start:start + frame_len] * w
        mag = np.abs(np.fft.fft(frame)[:n_bins])
        spectrogram[:, i] = 20 * np.log10(mag + 1e-10)

    times = np.arange(num_frames) * hop / sample_rate
    freqs = np.arange(n_bins) * sample_rate / frame_len
    return spectrogram, times, freqs


def compute_cepstrum(frame, sample_rate, apply_win=True, window_name='Hamminga'):
    if apply_win:
        w = WINDOWS[window_name](len(frame))
        windowed = frame * w
    else:
        windowed = frame
    n_fft = 1
    while n_fft < len(windowed):
        n_fft *= 2
    padded = np.zeros(n_fft)
    padded[:len(windowed)] = windowed
    fft_result = np.fft.fft(padded)
    log_spectrum = np.log(np.abs(fft_result) + 1e-10)
    cepstrum = np.fft.ifft(log_spectrum).real
    quefrency = np.arange(n_fft) / sample_rate
    return cepstrum, quefrency


def compute_formants_from_spectrum(frame, sample_rate, window_name='Hamminga',
                                   smooth_k=50, f_min=90, f_max=4000,
                                   n_formants=4):

    freqs, mag, _, _, _ = compute_fft(frame, sample_rate, window_name)

    kernel = np.ones(smooth_k) / smooth_k
    smooth = np.convolve(mag, kernel, mode='same')

    is_peak = (
        (smooth[1:-1] > smooth[:-2]) &
        (smooth[1:-1] > smooth[2:])
    )
    peak_indices = np.where(is_peak)[0] + 1

    candidates = [
        (freqs[i], smooth[i])
        for i in peak_indices
        if f_min < freqs[i] < f_max
    ]
    candidates.sort(key=lambda x: -x[1])

    formants = sorted([f for f, _ in candidates[:n_formants]])
    return formants


def compute_cepstral_f0(samples, sample_rate, frame_ms=20, overlap=0.5,
                        window_name='Hamminga', f_min=50, f_max=500,
                        threshold=0.01):
    frame_len = max(1, int(sample_rate * frame_ms / 1000))
    hop = max(1, int(frame_len * (1 - overlap)))
    num_frames = max(0, (len(samples) - frame_len) // hop + 1)

    if num_frames == 0:
        return None, None

    w = WINDOWS[window_name](frame_len)
    q_min = int(sample_rate / f_max)
    q_max = min(frame_len - 1, int(sample_rate / f_min))

    f0_array = np.zeros(num_frames)
    times = np.zeros(num_frames)

    for i in range(num_frames):
        start = i * hop
        frame = samples[start:start + frame_len] * w

        times[i] = (start + start + frame_len) / 2.0 / sample_rate

        cepstrum, _ = compute_cepstrum(frame, sample_rate,
                                        apply_win=False)

        if q_min >= q_max or q_max >= len(cepstrum):
            continue

        region = cepstrum[q_min:q_max + 1]
        best_local = int(np.argmax(region))
        best_q = q_min + best_local

        f0_array[i] = sample_rate / best_q if cepstrum[best_q] > threshold else 0.0

    return f0_array, times