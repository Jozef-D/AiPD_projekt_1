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
)

st.set_page_config(page_title="Analiza audio – dziedzina czasu", layout="wide")
st.title("Cechy sygnału audio w dziedzinie czasu")
st.markdown("**Projekt 1** — Analiza i przetwarzanie dźwięku 2025/26")

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

params = compute_all_params(samples, sr, frame_ms, overlap)
if params is None:
    st.warning("Sygnał za krótki.")
    st.stop()

ft = params['frame_times']
st.markdown(
    f"**Ramki:** {params['num_frames']}  |  "
    f"**Próbek/ramkę:** {params['frame_len']}  |  "
    f"**Hop:** {params['hop']}"
)

fig2, axes = plt.subplots(3, 1, figsize=(14, 7), sharex=True)

axes[0].plot(ft, params['volume'], color='#e74c3c', linewidth=0.8)
axes[0].set_ylabel("Volume (RMS)")
axes[0].set_title("Głośność")
axes[0].grid(True, alpha=0.3)

axes[1].plot(ft, params['ste'], color='#2ecc71', linewidth=0.8)
axes[1].set_ylabel("STE")
axes[1].set_title("Energia krótkoterminowa")
axes[1].grid(True, alpha=0.3)

axes[2].plot(ft, params['zcr'], color='#3498db', linewidth=0.8)
axes[2].set_ylabel("ZCR")
axes[2].set_xlabel("Czas [s]")
axes[2].set_title("Zero Crossing Rate")
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
st.pyplot(fig2)

st.subheader("Statystyki parametrów")
s1, s2, s3 = st.columns(3)
for col, name, data in [
    (s1, "Volume (RMS)", params['volume']),
    (s2, "STE", params['ste']),
    (s3, "ZCR", params['zcr']),
]:
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
