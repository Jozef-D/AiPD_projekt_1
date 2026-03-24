import struct
import numpy as np


def read_wav(file_bytes: bytes):
    if file_bytes[:4] != b'RIFF' or file_bytes[8:12] != b'WAVE':
        raise ValueError("To nie jest plik WAV.")

    pos = 12
    sample_rate = 0
    num_channels = 0
    bits_per_sample = 0
    audio_format = 0
    raw_data = None

    while pos < len(file_bytes) - 8:
        chunk_id = file_bytes[pos:pos + 4]
        chunk_size = struct.unpack_from('<I', file_bytes, pos + 4)[0]
        pos += 8

        if chunk_id == b'fmt ':
            audio_format = struct.unpack_from('<H', file_bytes, pos)[0]
            num_channels = struct.unpack_from('<H', file_bytes, pos + 2)[0]
            sample_rate = struct.unpack_from('<I', file_bytes, pos + 4)[0]
            bits_per_sample = struct.unpack_from('<H', file_bytes, pos + 14)[0]

        elif chunk_id == b'data':
            raw_data = file_bytes[pos:pos + chunk_size]

        pos += chunk_size
        if chunk_size % 2 != 0:
            pos += 1

    if raw_data is None:
        raise ValueError("Brak danych audio w pliku.")
    if audio_format != 1:
        raise ValueError("Obsługiwany tylko format PCM.")

    bps = bits_per_sample
    n = len(raw_data) // (bps // 8)

    if bps == 8:
        samples = np.frombuffer(raw_data, dtype=np.uint8).astype(np.float64)
        samples = (samples - 128.0) / 128.0
    elif bps == 16:
        samples = np.frombuffer(raw_data, dtype=np.int16).astype(np.float64)
        samples /= 32768.0
    elif bps == 24:
        samples = np.zeros(n, dtype=np.float64)
        for i in range(n):
            b0 = raw_data[i * 3]
            b1 = raw_data[i * 3 + 1]
            b2 = raw_data[i * 3 + 2]
            val = b0 | (b1 << 8) | (b2 << 16)
            if val >= 0x800000:
                val -= 0x1000000
            samples[i] = val / 8388608.0
    elif bps == 32:
        samples = np.frombuffer(raw_data, dtype=np.int32).astype(np.float64)
        samples /= 2147483648.0
    else:
        raise ValueError(f"Nieobsługiwana głębia: {bps} bit")

    if num_channels == 2:
        samples = (samples[0::2] + samples[1::2]) / 2.0
    elif num_channels > 2:
        mono = np.zeros(len(samples) // num_channels)
        for ch in range(num_channels):
            mono += samples[ch::num_channels]
        samples = mono / num_channels

    return samples, sample_rate, num_channels, bits_per_sample
