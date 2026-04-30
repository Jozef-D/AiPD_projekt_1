import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from src import (
    read_wav,
    compute_all_params,
    split_into_frames,
    detect_silence,
    compute_pitch,
    classify_voiced_unvoiced,
    parameters_to_csv,
    parameters_over_clip,
    WINDOWS,
    apply_window,
    compute_fft,
    compute_fft_for_frame,
    compute_spectrogram,
    compute_cepstrum,
    compute_cepstral_f0,
    compute_formants_from_spectrum,
)


st.set_page_config(page_title="Analiza audio", layout="wide")
st.title("Analiza sygnału audio")
st.markdown("**Projekt 1 + 2** — Analiza i przetwarzanie dźwięku 2025/26")

uploaded = st.file_uploader("Wczytaj plik .wav", type=["wav"])

if uploaded is None:
    st.info("Wczytaj plik .wav, aby rozpocząć analizę.")
    st.stop()

raw = uploaded.read()
try:
    samples, sr, n_ch, bps = read_wav(raw)
except Exception as e:
    st.error(f"Błąd: {e}")
    st.stop()

duration = len(samples) / sr
time_axis = np.arange(len(samples)) / sr

st.subheader("Informacje o pliku")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Sample rate", f"{sr} Hz")
c2.metric("Kanały (oryg.)", str(n_ch))
c3.metric("Bity", f"{bps} bit")
c4.metric("Czas", f"{duration:.2f} s")

st.subheader("Przebieg czasowy")
fig, ax = plt.subplots(figsize=(14, 3))
ax.plot(time_axis, samples, linewidth=0.3, color='steelblue')
ax.set_xlabel("Czas [s]")
ax.set_ylabel("Amplituda")
ax.set_xlim(0, duration)
ax.grid(True, alpha=0.3)
st.pyplot(fig)

st.subheader("Parametry w dziedzinie czasu (poziom ramki)")
col_a, col_b = st.columns(2)
frame_ms = col_a.slider("Długość ramki [ms]", 10, 100, 20, step=5)
overlap = col_b.slider("Overlap", 0.0, 0.9, 0.5, step=0.1)

include_spectral = st.checkbox("Pokaż parametry spektralne", value=True)

params = compute_all_params(samples, sr, frame_ms, overlap, include_spectral=include_spectral)
if params is None:
    st.warning("Sygnał za krótki.")
    st.stop()

ft = params['frame_times']
st.markdown(
    f"**Ramki:** {params['num_frames']}  |  "
    f"**Próbek/ramkę:** {params['frame_len']}  |  "
    f"**Hop:** {params['hop']}"
)

num_plots = 4 + (3 if include_spectral else 0)
fig2, axes = plt.subplots(num_plots, 1, figsize=(21, 9), sharex=True)

if num_plots == 1:
    axes = [axes]


i = 0

axes[i].plot(ft, params['volume'], color='#e74c3c', linewidth=0.8)
axes[i].set_ylabel("Volume (RMS)")
axes[i].set_title("Głośność")
axes[i].grid(True, alpha=0.3)
i += 1

axes[i].plot(ft, params['ste'], color='#2ecc71', linewidth=0.8)
axes[i].set_ylabel("STE")
axes[i].set_title("Energia krótkoterminowa")
axes[i].grid(True, alpha=0.3)
i += 1

axes[i].plot(ft, params['zcr'], color='#3498db', linewidth=0.8)
axes[i].set_ylabel("ZCR")
axes[i].set_title("Zero Crossing Rate")
axes[i].grid(True, alpha=0.3)
i += 1

axes[i].plot(ft, params['energy_entropy'], color='#9b59b6', linewidth=0.8)
axes[i].set_ylabel("Entropia")
axes[i].set_title("Entropia energii")
axes[i].grid(True, alpha=0.3)
i += 1

if include_spectral:
    axes[i].plot(ft, params['spectral_centroid'], color='#6bb04f', linewidth=0.8)
    axes[i].set_ylabel("Śr. Cężkości Widma")
    axes[i].set_title("Środek ciężkości widma")
    axes[i].grid(True, alpha=0.3)
    i += 1

    axes[i].plot(ft, params['spectral_rolloff'], color='#6bb04f', linewidth=0.8)
    axes[i].set_ylabel("Cz. Graniczna Widma")
    axes[i].set_title("Częstotliwość graniczna widma")
    axes[i].grid(True, alpha=0.3)
    i += 1

    axes[i].plot(ft, params['spectral_flatness'], color='#e3aca7', linewidth=0.8)
    axes[i].set_ylabel("Płaskość Widma")
    axes[i].set_xlabel("Czas [s]")
    axes[i].set_title("Płaskość Widma")
    axes[i].grid(True, alpha=0.3)
else:
    axes[i-1].set_xlabel("Czas [s]")
plt.tight_layout()
st.pyplot(fig2)

st.subheader("Statystyki parametrów")
stats_list = [
    ("Volume (RMS)", params['volume']),
    ("STE", params['ste']),
    ("ZCR", params['zcr']),
    ("Energy Entropy", params['energy_entropy']),
]

if include_spectral:
    stats_list.extend([
        ("Spectral Centroid", params['spectral_centroid']),
        ("Spectral Rolloff", params['spectral_rolloff']),
        ("Spectral Flatness", params['spectral_flatness']),
    ])

cols = st.columns(len(stats_list))

for col, (name, data) in zip(cols, stats_list):
    with col:
        st.markdown(f"**{name}**")
        st.write(f"Średnia: {np.mean(data):.6f}")
        st.write(f"Max: {np.max(data):.6f}")
        st.write(f"Min: {np.min(data):.6f}")
        st.write(f"Odch. std: {np.std(data):.6f}")

st.subheader("Detekcja ciszy")
vol_thresh_pct = st.slider("Próg ciszy (% max. głośności)", 1, 30, 5, step=1)
silence_mask, threshold = detect_silence(params, vol_thresh_pct / 100.0)

n_silent = int(np.sum(silence_mask))
n_voiced_total = params['num_frames'] - n_silent

sc1, sc2, sc3 = st.columns(3)
sc1.metric("Ramki ciche", f"{n_silent} ({n_silent / params['num_frames'] * 100:.1f}%)")
sc2.metric("Ramki z dźwiękiem", str(n_voiced_total))
sc3.metric("Próg", f"{threshold:.6f}")

fig3, ax3 = plt.subplots(figsize=(14, 3.5))
ax3.plot(time_axis, samples, linewidth=0.3, color='steelblue', label='Sygnał')
hop = params['hop']
flen = params['frame_len']
for i in range(len(silence_mask)):
    if silence_mask[i]:
        ax3.axvspan(i * hop / sr, (i * hop + flen) / sr, color='red', alpha=0.2)
red_patch = mpatches.Patch(color='red', alpha=0.2, label='Cisza')
ax3.legend(handles=[ax3.get_lines()[0], red_patch])
ax3.set_xlabel("Czas [s]")
ax3.set_ylabel("Amplituda")
ax3.set_xlim(0, duration)
ax3.grid(True, alpha=0.3)
st.pyplot(fig3)

st.markdown("---")
st.subheader("Częstotliwość tonu podstawowego (F0)")

col_f1, col_f2, col_f3 = st.columns(3)
f_min = col_f1.number_input("F0 min [Hz]", 50, 200, 50, step=10)
f_max = col_f2.number_input("F0 max [Hz]", 200, 1000, 500, step=50)
pitch_method = col_f3.selectbox("Metoda", ["autocorrelation", "amdf", "obie"])

frames, frame_times_pitch, _, _ = split_into_frames(samples, sr, frame_ms, overlap)

with st.spinner("Obliczanie F0..."):
    if pitch_method == "obie":
        f0_auto = compute_pitch(frames, sr, 'autocorrelation', f_min, f_max)
        f0_amdf = compute_pitch(frames, sr, 'amdf', f_min, f_max)
    elif pitch_method == "autocorrelation":
        f0_auto = compute_pitch(frames, sr, 'autocorrelation', f_min, f_max)
        f0_amdf = None
    else:
        f0_auto = None
        f0_amdf = compute_pitch(frames, sr, 'amdf', f_min, f_max)

fig4, ax4 = plt.subplots(figsize=(14, 4))

if f0_auto is not None:
    mask_a = f0_auto > 0
    ax4.scatter(ft[mask_a], f0_auto[mask_a], s=4, color='#e74c3c',
                label='Autokorelacja', alpha=0.7)

if f0_amdf is not None:
    mask_m = f0_amdf > 0
    ax4.scatter(ft[mask_m], f0_amdf[mask_m], s=4, color='#3498db',
                label='AMDF', alpha=0.7)

ax4.set_xlabel("Czas [s]")
ax4.set_ylabel("F0 [Hz]")
ax4.set_title("Częstotliwość tonu podstawowego (F0)")
ax4.legend()
ax4.grid(True, alpha=0.3)
ax4.set_xlim(0, duration)
st.pyplot(fig4)

if f0_auto is not None:
    voiced_a = f0_auto[f0_auto > 0]
    if len(voiced_a) > 0:
        st.markdown(f"**Autokorelacja** — średnia F0: {np.mean(voiced_a):.1f} Hz, "
                    f"mediana: {np.median(voiced_a):.1f} Hz, "
                    f"zakres: {np.min(voiced_a):.1f}–{np.max(voiced_a):.1f} Hz")

if f0_amdf is not None:
    voiced_m = f0_amdf[f0_amdf > 0]
    if len(voiced_m) > 0:
        st.markdown(f"**AMDF** — średnia F0: {np.mean(voiced_m):.1f} Hz, "
                    f"mediana: {np.median(voiced_m):.1f} Hz, "
                    f"zakres: {np.min(voiced_m):.1f}–{np.max(voiced_m):.1f} Hz")

st.subheader("Fragmenty dźwięczne / bezdźwięczne")

f0_for_class = f0_auto if f0_auto is not None else f0_amdf

col_e, col_z = st.columns(2)
en_ratio = col_e.slider("Próg energii (cisza)", 0.01, 0.30, 0.10, step=0.01)
zcr_thresh = col_z.slider("Próg ZCR (voiced vs unvoiced)", 0.05, 0.60, 0.30, step=0.05)

labels = classify_voiced_unvoiced(params, f0_for_class, en_ratio, zcr_thresh)

n_v = labels.count('V')
n_u = labels.count('U')
n_s = labels.count('S')
total = len(labels)

lc1, lc2, lc3 = st.columns(3)
lc1.metric("Dźwięczne (V)", f"{n_v} ({n_v/total*100:.1f}%)")
lc2.metric("Bezdźwięczne (U)", f"{n_u} ({n_u/total*100:.1f}%)")
lc3.metric("Cisza (S)", f"{n_s} ({n_s/total*100:.1f}%)")

fig5, ax5 = plt.subplots(figsize=(14, 3.5))
ax5.plot(time_axis, samples, linewidth=0.3, color='gray', alpha=0.5)

color_map = {'V': '#2ecc71', 'U': '#f39c12', 'S': '#e74c3c'}
for i, label in enumerate(labels):
    t_start = i * hop / sr
    t_end = (i * hop + flen) / sr
    ax5.axvspan(t_start, t_end, color=color_map[label], alpha=0.15)

patches = [
    mpatches.Patch(color='#2ecc71', alpha=0.3, label='Dźwięczne (V)'),
    mpatches.Patch(color='#f39c12', alpha=0.3, label='Bezdźwięczne (U)'),
    mpatches.Patch(color='#e74c3c', alpha=0.3, label='Cisza (S)'),
]
ax5.legend(handles=patches, loc='upper right')
ax5.set_xlabel("Czas [s]")
ax5.set_ylabel("Amplituda")
ax5.set_title("Klasyfikacja: dźwięczne / bezdźwięczne / cisza")
ax5.set_xlim(0, duration)
ax5.grid(True, alpha=0.3)
st.pyplot(fig5)

f0_for_csv = f0_auto if f0_auto is not None else f0_amdf

csv_df = parameters_to_csv(
    samples, sr, frame_ms, overlap,
    f0=f0_for_csv,
    vol_threshold=threshold,
    zcr_threshold=zcr_thresh,
    params = params,

)

st.download_button(
    label="⬇Pobierz parametry jako CSV",
    data=csv_df.to_csv(index=False).encode("utf-8"),
    file_name=f"{uploaded.name.replace('.wav', '')}_params.csv",
    mime="text/csv",
)

st.markdown("---")
st.subheader("Parametry na poziomie klipu")


clip_params = parameters_over_clip(samples, sr, frame_ms, overlap)

if clip_params is not None and params is not None:
    cp1, cp2, cp3, cp4 = st.columns(4)
    cp1.metric("LSTER", f"{clip_params['lster']:.4f}",
               help="Odsetek ramek z STE poniżej 50% średniej — wysoka wartość sugeruje mowę")
    cp2.metric("HZCRR", f"{clip_params['hzcrr']:.4f}",
               help="Odsetek ramek z ZCR powyżej 150% średniej — wysoka wartość sugeruje muzykę/szum")
    cp3.metric("ZSTD", f"{clip_params['zstd']:.6f}",
               help="Odchylenie std ZCR — miara zmienności sygnału")
    cp4.metric("Energy Entropy", f"{clip_params['energy_entropy']:.4f}",
               help="Entropia rozkładu energii między ramkami")

    st.markdown("#### Klasyfikacja muzyka / mowa")

    score = 0  # dodatni = muzyka, ujemny = mowa ( przynajmniej w teorii)

    if clip_params['lster'] > 0.13:
        score -= 1
    else:
        score += 1

    if clip_params['hzcrr'] > 0.13:
        score += 1
    else:
        score -= 1

    if np.mean(params['zcr']) > 0.05:
        score += 1
    else:
        score -= 1

    if include_spectral:

        if np.mean(params['spectral_centroid']) > 1800:
            score += 1
        else:
            score -= 1

        if np.std(params['spectral_centroid']) > 500:
            score += 1
        else:
            score -= 1

        if np.mean(params['spectral_rolloff']) > 3000:
            score += 1
        else:
            score -= 1

        if np.std(params['spectral_rolloff']) > 800:
            score += 1
        else:
            score -= 1

        if np.mean(params['spectral_flatness']) > 0.2:
            score += 2
        else:
            score -= 2


    st.metric("Wynik klasyfikacji", score, help="< 0 = mowa, > 0 = muzyka")

    if score <= -2:

        st.info("Sygnał przypomina **mowę**")
    elif score >= 2:
        st.success("Sygnał przypomina **muzykę**")
    elif score < 0:

        st.info("Prawdopodobnie **mowa** (niskie zaufanie)")
    elif score > 0:
        st.success("Prawdopodobnie **muzyka** (niskie zaufanie)")
    else:
        st.warning("Sygnał **niejednoznaczny**")


st.markdown("---")
st.markdown("## Projekt 2 — Analiza częstotliwościowa")

st.subheader("Widmo FFT całego sygnału")

window_names = list(WINDOWS.keys())
fft_window = st.selectbox("Funkcja okienkowa (cały sygnał)", window_names, key="fft_whole")

freqs_whole, mag_whole, mag_db_whole, windowed_whole, win_whole = compute_fft(
    samples, sr, fft_window
)

fig_fft_whole, (ax_fft1, ax_fft2) = plt.subplots(2, 1, figsize=(14, 6))

ax_fft1.plot(freqs_whole, mag_whole, linewidth=0.5, color='#e74c3c')
ax_fft1.set_ylabel("Amplituda")
ax_fft1.set_title(f"Widmo FFT — skala liniowa (okno: {fft_window})")
ax_fft1.grid(True, alpha=0.3)

ax_fft2.plot(freqs_whole, mag_db_whole, linewidth=0.5, color='#3498db')
ax_fft2.set_xlabel("Częstotliwość [Hz]")
ax_fft2.set_ylabel("Amplituda [dB]")
ax_fft2.set_title(f"Widmo FFT — skala dB (okno: {fft_window})")
ax_fft2.grid(True, alpha=0.3)

plt.tight_layout()
st.pyplot(fig_fft_whole)

st.subheader("Widmo FFT wybranej ramki")

col_fr, col_fw = st.columns(2)
frame_idx = col_fr.slider("Numer ramki", 0, params['num_frames'] - 1, 0, step=1)
frame_window = col_fw.selectbox("Funkcja okienkowa (ramka)", window_names, key="fft_frame")

frame_start = frame_idx * params['hop']
frame_end = frame_start + params['frame_len']
frame_time_start = frame_start / sr
frame_time_end = frame_end / sr
st.markdown(f"Ramka **{frame_idx}**: {frame_time_start:.4f} s – {frame_time_end:.4f} s")

freqs_fr, mag_fr, mag_db_fr, windowed_fr, win_fr = compute_fft_for_frame(
    samples, sr, frame_idx, params['frame_len'], params['hop'], frame_window
)

frame_signal = samples[frame_start:frame_end]

bin_size = sr / params['frame_len']
smooth_k = int(500 / bin_size)

formants = compute_formants_from_spectrum(frame_signal, sr, frame_window, smooth_k)

st.markdown("### Wykryte formanty")
st.markdown(
    f"""
    **F1:** {formants[0]:.1f} Hz  
    **F2:** {formants[1]:.1f} Hz  
    **F3:** {formants[2]:.1f} Hz  
    """ if len(formants) >= 3 else "Brak wystarczających danych"
)
frame_time_ax = np.arange(len(frame_signal)) / sr + frame_time_start

fig_frame_fft, axes_fr = plt.subplots(2, 2, figsize=(14, 7))

axes_fr[0, 0].plot(frame_time_ax, frame_signal, linewidth=0.8, color='steelblue')
axes_fr[0, 0].set_title("Ramka — oryginalna")
axes_fr[0, 0].set_xlabel("Czas [s]")
axes_fr[0, 0].set_ylabel("Amplituda")
axes_fr[0, 0].grid(True, alpha=0.3)

axes_fr[0, 1].plot(frame_time_ax, windowed_fr, linewidth=0.8, color='#2ecc71')
axes_fr[0, 1].set_title(f"Ramka po oknie: {frame_window}")
axes_fr[0, 1].set_xlabel("Czas [s]")
axes_fr[0, 1].set_ylabel("Amplituda")
axes_fr[0, 1].grid(True, alpha=0.3)

axes_fr[1, 0].plot(freqs_fr, mag_fr, linewidth=0.8, color='#e74c3c')
for f in formants[:3]:
    axes_fr[1, 0].axvline(x=f, linestyle='--', alpha=0.6)
axes_fr[1, 0].set_title("Widmo FFT — liniowe")
axes_fr[1, 0].set_xlabel("Częstotliwość [Hz]")
axes_fr[1, 0].set_ylabel("Amplituda")
axes_fr[1, 0].grid(True, alpha=0.3)

axes_fr[1, 1].plot(freqs_fr, mag_db_fr, linewidth=0.8, color='#3498db')
for f in formants[:3]:
    axes_fr[1, 1].axvline(x=f, linestyle='--', alpha=0.6)
axes_fr[1, 1].set_title("Widmo FFT — dB")
axes_fr[1, 1].set_xlabel("Częstotliwość [Hz]")
axes_fr[1, 1].set_ylabel("Amplituda [dB]")
axes_fr[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
st.pyplot(fig_frame_fft)

st.subheader("Porównanie funkcji okienkowych")

fig_win, axes_win = plt.subplots(len(WINDOWS), 2, figsize=(14, 3 * len(WINDOWS)))

for idx, (wname, wfunc) in enumerate(WINDOWS.items()):
    w = wfunc(params['frame_len'])
    w_freqs, w_mag, w_mag_db, _, _ = compute_fft(w, sr, 'Prostokątne')

    axes_win[idx, 0].plot(w, linewidth=1.2, color='#2ecc71')
    axes_win[idx, 0].set_title(f"{wname} — dziedzina czasu")
    axes_win[idx, 0].set_ylabel("Amplituda")
    axes_win[idx, 0].set_xlim(0, len(w))
    axes_win[idx, 0].grid(True, alpha=0.3)

    axes_win[idx, 1].plot(w_freqs, w_mag_db, linewidth=0.8, color='#e74c3c')
    axes_win[idx, 1].set_title(f"{wname} — widmo [dB]")
    axes_win[idx, 1].set_ylabel("dB")
    axes_win[idx, 1].grid(True, alpha=0.3)

axes_win[-1, 0].set_xlabel("Próbka")
axes_win[-1, 1].set_xlabel("Częstotliwość [Hz]")
plt.tight_layout()
st.pyplot(fig_win)

st.subheader("Wpływ okna na widmo wybranej ramki")

compare_windows = st.multiselect(
    "Wybierz okna do porównania",
    window_names,
    default=window_names,
    key="compare_win"
)

if compare_windows:
    fig_cmp, (ax_cmp_t, ax_cmp_f) = plt.subplots(1, 2, figsize=(14, 5))
    colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12', '#9b59b6']

    for i, wname in enumerate(compare_windows):
        f_fr, m_fr, m_db_fr, w_fr, _ = compute_fft_for_frame(
            samples, sr, frame_idx, params['frame_len'], params['hop'], wname
        )
        color = colors[i % len(colors)]
        ax_cmp_t.plot(w_fr, linewidth=0.8, color=color, label=wname, alpha=0.8)
        ax_cmp_f.plot(f_fr, m_db_fr, linewidth=0.8, color=color, label=wname, alpha=0.8)

    ax_cmp_t.set_title(f"Ramka {frame_idx} — po zastosowaniu okien")
    ax_cmp_t.set_xlabel("Próbka")
    ax_cmp_t.set_ylabel("Amplituda")
    ax_cmp_t.legend()
    ax_cmp_t.grid(True, alpha=0.3)

    ax_cmp_f.set_title(f"Ramka {frame_idx} — widma po oknach [dB]")
    ax_cmp_f.set_xlabel("Częstotliwość [Hz]")
    ax_cmp_f.set_ylabel("dB")
    ax_cmp_f.legend()
    ax_cmp_f.grid(True, alpha=0.3)

    plt.tight_layout()
    st.pyplot(fig_cmp)

st.markdown("---")
st.subheader("Spektrogram")

col_sp1, col_sp2, col_sp3 = st.columns(3)
spec_frame_ms = col_sp1.slider("Długość ramki spektrogramu [ms]", 5, 100, 25, step=5, key="spec_frame")
spec_overlap = col_sp2.slider("Overlap spektrogramu", 0.0, 0.95, 0.75, step=0.05, key="spec_overlap")
spec_window = col_sp3.selectbox("Okno spektrogramu", window_names, index=2, key="spec_win")

with st.spinner("Obliczanie spektrogramu..."):
    spec, spec_times, spec_freqs = compute_spectrogram(
        samples, sr, spec_frame_ms, spec_overlap, spec_window
    )

if spec is not None:
    fig_spec, ax_spec = plt.subplots(figsize=(14, 5))
    img = ax_spec.pcolormesh(spec_times, spec_freqs, spec, shading='auto', cmap='inferno')
    ax_spec.set_xlabel("Czas [s]")
    ax_spec.set_ylabel("Częstotliwość [Hz]")
    ax_spec.set_title(f"Spektrogram (okno: {spec_window}, ramka: {spec_frame_ms} ms, overlap: {spec_overlap})")
    plt.colorbar(img, ax=ax_spec, label="dB")
    plt.tight_layout()
    st.pyplot(fig_spec)

st.markdown("---")
st.subheader("Częstotliwość krtaniowa (Cepstrum)")

col_cp1, col_cp2 = st.columns(2)
cep_f_min = col_cp1.number_input("F0 min (cepstrum) [Hz]", 50, 200, 50, step=10, key="cep_fmin")
cep_f_max = col_cp2.number_input("F0 max (cepstrum) [Hz]", 200, 1000, 500, step=50, key="cep_fmax")

with st.spinner("Obliczanie F0 z cepstrum..."):
    cep_f0, cep_times = compute_cepstral_f0(
        samples, sr, frame_ms, overlap, 'Hamminga', cep_f_min, cep_f_max
    )

if cep_f0 is not None:
    fig_cep_f0, ax_cep_f0 = plt.subplots(figsize=(14, 4))
    mask_cep = cep_f0 > 0
    ax_cep_f0.scatter(cep_times[mask_cep], cep_f0[mask_cep], s=4, color='#9b59b6', alpha=0.7, label='Cepstrum F0')

    if f0_auto is not None:
        mask_a2 = f0_auto > 0
        ax_cep_f0.scatter(ft[mask_a2], f0_auto[mask_a2], s=4, color='#e74c3c', alpha=0.4, label='Autokorelacja F0')

    ax_cep_f0.set_xlabel("Czas [s]")
    ax_cep_f0.set_ylabel("F0 [Hz]")
    ax_cep_f0.set_title("Porównanie F0: Cepstrum vs Autokorelacja")
    ax_cep_f0.legend()
    ax_cep_f0.grid(True, alpha=0.3)
    ax_cep_f0.set_xlim(0, duration)
    st.pyplot(fig_cep_f0)

    voiced_cep = cep_f0[cep_f0 > 0]
    if len(voiced_cep) > 0:
        st.markdown(f"**Cepstrum** — średnia F0: {np.mean(voiced_cep):.1f} Hz, "
                    f"mediana: {np.median(voiced_cep):.1f} Hz, "
                    f"zakres: {np.min(voiced_cep):.1f}–{np.max(voiced_cep):.1f} Hz")

    st.subheader("Cepstrum wybranej ramki")
    cep_frame_idx = st.slider("Ramka (cepstrum)", 0, params['num_frames'] - 1, 0, step=1, key="cep_fr")

    cep_start = cep_frame_idx * params['hop']
    cep_end = cep_start + params['frame_len']
    cep_frame = samples[cep_start:cep_end]

    windowed_cep, _ = apply_window(cep_frame, 'Hamminga')
    cepstrum_vals, quefrency = compute_cepstrum(windowed_cep, sr)

    q_min_plot = int(sr / cep_f_max)
    q_max_plot = min(len(cepstrum_vals) - 1, int(sr / cep_f_min))

    fig_cep_detail, (ax_cd1, ax_cd2) = plt.subplots(1, 2, figsize=(14, 4))

    ax_cd1.plot(quefrency * 1000, cepstrum_vals, linewidth=0.8, color='#9b59b6')
    ax_cd1.set_xlabel("Quefrency [ms]")
    ax_cd1.set_ylabel("Amplituda")
    ax_cd1.set_title(f"Cepstrum — ramka {cep_frame_idx}")
    ax_cd1.set_xlim(0, quefrency[len(quefrency)//2] * 1000)
    ax_cd1.grid(True, alpha=0.3)

    ax_cd2.plot(quefrency[q_min_plot:q_max_plot+1] * 1000,
                cepstrum_vals[q_min_plot:q_max_plot+1],
                linewidth=1.0, color='#e74c3c')
    ax_cd2.set_xlabel("Quefrency [ms]")
    ax_cd2.set_ylabel("Amplituda")
    ax_cd2.set_title(f"Cepstrum — zakres F0 ({cep_f_min}–{cep_f_max} Hz)")
    ax_cd2.grid(True, alpha=0.3)

    if q_min_plot < q_max_plot:
        peak_q = q_min_plot + np.argmax(cepstrum_vals[q_min_plot:q_max_plot+1])
        peak_f0 = sr / peak_q if peak_q > 0 else 0
        ax_cd2.axvline(x=quefrency[peak_q] * 1000, color='black', linestyle='--', alpha=0.5)
        st.markdown(f"Peak quefrency: **{quefrency[peak_q]*1000:.2f} ms** → F0 = **{peak_f0:.1f} Hz**")

    plt.tight_layout()
    st.pyplot(fig_cep_detail)
