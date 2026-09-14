import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Rectangle
import librosa
import pygame
import tkinter as tk
from tkinter import filedialog
import time


# ============================================================
# SETTINGS
# ============================================================

BASE_BRIGHTNESS = 30

# Frequency bands
LOW_MIN = 20
LOW_MAX = 100

MID_MIN = 200
MID_MAX = 300

HIGH_MIN = 800
HIGH_MAX = 20000

# FFT settings
N_FFT = 1024
HOP_LENGTH = 512

# Energy smoothing
SMOOTHING = 0.1

# ------------------------------------------------------------
# Beat / transient controls
# ------------------------------------------------------------
BEAT_GAIN = 18
TRANSIENT_GAIN = 12
BEAT_DECAY_FRAMES = 1
TRANSIENT_THRESHOLD = 0.18
OVERALL_WEIGHT = 0.15


# ============================================================
# SELECT AUDIO FILE
# ============================================================

root = tk.Tk()
root.withdraw()

audio_file = filedialog.askopenfilename(
    title="Select MP3 Song",
    filetypes=[
        ("MP3 files", "*.mp3"),
        ("Audio files", "*.wav *.mp3 *.flac"),
        ("All files", "*.*")
    ]
)

if not audio_file:
    print("No audio file selected.")
    raise SystemExit


# ============================================================
# LOAD AUDIO
# ============================================================

print("Loading audio...")

y, sr = librosa.load(
    audio_file,
    sr=None,
    mono=True
)

if np.max(np.abs(y)) > 0:
    y = y / np.max(np.abs(y))

duration = len(y) / sr

print(f"Sample rate : {sr} Hz")
print(f"Duration    : {duration:.2f} seconds")


# ============================================================
# STFT
# ============================================================

print("Calculating STFT...")

D = librosa.stft(
    y,
    n_fft=N_FFT,
    hop_length=HOP_LENGTH,
    window="hann"
)

magnitude = np.abs(D)

frequencies = librosa.fft_frequencies(
    sr=sr,
    n_fft=N_FFT
)

num_frames = magnitude.shape[1]

print(f"FFT bins    : {len(frequencies)}")
print(f"Frames      : {num_frames}")


# ============================================================
# FREQUENCY BAND MASKS
# ============================================================

nyquist = sr / 2

low_max = min(LOW_MAX, nyquist)
mid_max = min(MID_MAX, nyquist)
high_max = min(HIGH_MAX, nyquist)

low_mask = (
    (frequencies >= LOW_MIN) &
    (frequencies < low_max)
)

mid_mask = (
    (frequencies >= MID_MIN) &
    (frequencies < mid_max)
)

high_mask = (
    (frequencies >= HIGH_MIN) &
    (frequencies <= high_max)
)


# ============================================================
# BAND ENERGY
# ============================================================

print("Calculating frequency-band energy...")

low_amplitude = np.sqrt(
    np.mean(magnitude[low_mask, :] ** 2, axis=0)
)

mid_amplitude = np.sqrt(
    np.mean(magnitude[mid_mask, :] ** 2, axis=0)
)

high_amplitude = np.sqrt(
    np.mean(magnitude[high_mask, :] ** 2, axis=0)
)


def normalize_band(values):
    """Robust 95th-percentile normalization."""
    reference = np.percentile(values, 95)

    if reference <= 1e-12:
        return np.zeros_like(values)

    normalized = values / reference
    return np.clip(normalized, 0, 1)


low_norm = normalize_band(low_amplitude)
mid_norm = normalize_band(mid_amplitude)
high_norm = normalize_band(high_amplitude)


# ============================================================
# OVERALL ENERGY
# ============================================================

overall_raw = (
    0.45 * low_norm +
    0.35 * mid_norm +
    0.20 * high_norm
)

overall_norm = normalize_band(overall_raw)


# ============================================================
# BEAT STRENGTH
# ============================================================

print("Detecting beats...")

onset_envelope = librosa.onset.onset_strength(
    y=y,
    sr=sr,
    hop_length=HOP_LENGTH,
    aggregate=np.mean
)

if len(onset_envelope) < num_frames:
    onset_envelope = np.pad(
        onset_envelope,
        (0, num_frames - len(onset_envelope))
    )
else:
    onset_envelope = onset_envelope[:num_frames]

tempo, beat_frames = librosa.beat.beat_track(
    onset_envelope=onset_envelope,
    sr=sr,
    hop_length=HOP_LENGTH
)

beat_frames = np.asarray(beat_frames, dtype=int)
beat_frames = beat_frames[
    (beat_frames >= 0) &
    (beat_frames < num_frames)
]

onset_norm = normalize_band(onset_envelope)

beat_strength = np.zeros(num_frames)

for beat_frame in beat_frames:
    if beat_frame >= num_frames:
        continue

    strength = onset_norm[beat_frame]

    for k in range(BEAT_DECAY_FRAMES):
        idx = beat_frame + k

        if idx >= num_frames:
            break

        decay = np.exp(-k / (BEAT_DECAY_FRAMES / 3.0))

        beat_strength[idx] = max(
            beat_strength[idx],
            strength * decay
        )

beat_strength = np.clip(beat_strength, 0, 1)


# ============================================================
# TRANSIENT STRENGTH
# ============================================================

print("Calculating transient strength...")

kernel = 9
pad = kernel // 2

padded = np.pad(
    onset_envelope,
    (pad, pad),
    mode="edge"
)

rolling_baseline = np.array([
    np.median(padded[i:i + kernel])
    for i in range(num_frames)
])

transient_raw = np.maximum(
    onset_envelope - rolling_baseline,
    0
)

transient_norm = normalize_band(transient_raw)

transient_strength = np.where(
    transient_norm >= TRANSIENT_THRESHOLD,
    transient_norm,
    0
)

if np.max(transient_strength) > 0:
    transient_strength = (
        transient_strength /
        np.max(transient_strength)
    )

transient_strength = np.clip(transient_strength, 0, 1)


# ============================================================
# BRIGHTNESS HELPERS
# ============================================================

def calculate_energy_brightness(amplitude):
    brightness = (
        BASE_BRIGHTNESS +
        amplitude * (100 - BASE_BRIGHTNESS)
    )

    return np.clip(
        brightness,
        BASE_BRIGHTNESS,
        100
    )


low_brightness = calculate_energy_brightness(low_norm)
mid_brightness = calculate_energy_brightness(mid_norm)
high_brightness = calculate_energy_brightness(high_norm)
overall_brightness = calculate_energy_brightness(overall_norm)


# ============================================================
# SIX-CHANNEL STRICTLY SYMMETRIC LIGHTING MIXER
# ============================================================
# Channel configuration (Indices 0 to 5):
#   0 & 5 (Outer panels)      -> Highs (Treble)
#   1 & 4 (Inner-mid panels)  -> Mids
#   2 & 3 (Middle panels)     -> Low Bass
# ============================================================

channel_names = [
    "OUTER-L", "MID-L", "BASS-L", "BASS-R", "MID-R", "OUTER-R"
]

channel_targets = np.zeros(
    (6, num_frames)
)

def clamp_brightness(values):
    return np.clip(
        values,
        BASE_BRIGHTNESS,
        100
    )


for i in range(num_frames):

    L = low_norm[i]
    M = mid_norm[i]
    H = high_norm[i]
    B = beat_strength[i]
    T = transient_strength[i]

    # Calculate exact band values and explicitly enforce mirror symmetry
    # High Band (Outer panels: 0 and 5)
    high_base = BASE_BRIGHTNESS + H * (100 - BASE_BRIGHTNESS)
    high_val = high_base + (B * BEAT_GAIN * 0.45) + (T * TRANSIENT_GAIN * 1.00)

    # Mid Band (Inner-mid panels: 1 and 4)
    mid_base = BASE_BRIGHTNESS + M * (100 - BASE_BRIGHTNESS)
    mid_val = mid_base + (B * BEAT_GAIN * 0.55) + (T * TRANSIENT_GAIN * 0.45)

    # Low Bass Band (Middle panels: 2 and 3)
    low_base = BASE_BRIGHTNESS + L * (100 - BASE_BRIGHTNESS)
    low_val = low_base + (B * BEAT_GAIN * 1.00) + (T * TRANSIENT_GAIN * 0.20)

    # Assign symmetrically to guarantee identical pairing
    channel_targets[0, i] = high_val  # OUTER-L
    channel_targets[1, i] = mid_val   # MID-L
    channel_targets[2, i] = low_val   # BASS-L
    channel_targets[3, i] = low_val   # BASS-R
    channel_targets[4, i] = mid_val   # MID-R
    channel_targets[5, i] = high_val  # OUTER-R

    channel_targets[:, i] = clamp_brightness(channel_targets[:, i])


# ============================================================
# INITIAL VALUES
# ============================================================

current_channels = np.ones(6) * BASE_BRIGHTNESS

current_low = BASE_BRIGHTNESS
current_mid = BASE_BRIGHTNESS
current_high = BASE_BRIGHTNESS


# ============================================================
# PYGAME AUDIO
# ============================================================

pygame.mixer.init()
pygame.mixer.music.load(audio_file)


# ============================================================
# FIGURE
# ============================================================

fig = plt.figure(
    figsize=(16, 9)
)

fig.suptitle(
    "Music Reactive Lighting System — LIVE SIMULATION",
    fontsize=18,
    fontweight="bold"
)


# ============================================================
# LIVE SPECTRUM
# ============================================================

ax_spectrum = plt.subplot2grid(
    (3, 4),
    (0, 0),
    colspan=3
)

ax_spectrum.set_title(
    "Live Frequency Spectrum"
)

ax_spectrum.set_xlabel(
    "Frequency (Hz)"
)

ax_spectrum.set_ylabel(
    "Magnitude"
)

ax_spectrum.set_xscale("log")

ax_spectrum.set_xlim(
    20,
    min(20000, nyquist)
)

ax_spectrum.set_ylim(
    0,
    max(np.max(magnitude) * 0.5, 1e-9)
)

line_spectrum, = ax_spectrum.plot(
    frequencies,
    magnitude[:, 0]
)


# ============================================================
# STATUS PANEL
# ============================================================

ax_status = plt.subplot2grid(
    (3, 4),
    (0, 3)
)

ax_status.axis("off")

status_text = ax_status.text(
    0.05,
    0.98,
    "",
    verticalalignment="top",
    fontsize=10
)


# ============================================================
# ENERGY / EVENT GRAPH
# ============================================================

ax_events = plt.subplot2grid(
    (3, 4),
    (1, 0),
    colspan=2
)

ax_events.set_title(
    "Energy + Beat + Transient"
)

ax_events.set_ylim(
    0,
    1.05
)

ax_events.set_xlim(
    0,
    max(1, min(duration, 10))
)

ax_events.set_xlabel("Time (first 10 s)")
ax_events.set_ylabel("Normalized strength")

energy_line, = ax_events.plot([], [], label="Overall Energy")
beat_line, = ax_events.plot([], [], label="Beat")
transient_line, = ax_events.plot([], [], label="Transient")

ax_events.legend(
    loc="upper right",
    fontsize=8
)


# ============================================================
# SIX LIGHTING CHANNELS
# ============================================================

ax_channels = plt.subplot2grid(
    (3, 4),
    (1, 2),
    colspan=2
)

ax_channels.set_title(
    "Six-Channel Lighting Output"
)

ax_channels.set_ylim(
    0,
    100
)

ax_channels.set_ylabel(
    "Brightness (%)"
)

ax_channels.set_xticks(
    np.arange(6)
)

ax_channels.set_xticklabels(
    channel_names,
    fontsize=9
)

channel_bars = ax_channels.bar(
    np.arange(6),
    [BASE_BRIGHTNESS] * 6
)


# ============================================================
# SIX VIRTUAL LIGHT PANELS
# ============================================================

ax_panels = plt.subplot2grid(
    (3, 4),
    (2, 0),
    colspan=4
)

ax_panels.set_title(
    "Outer: Highs | Inner-Mid: Mids | Middle: Low Bass — Symmetric Virtual Lighting Pattern"
)

ax_panels.set_xlim(
    0,
    6
)

ax_panels.set_ylim(
    0,
    1
)

ax_panels.axis("off")

panel_rectangles = []
panel_values = []

for i, name in enumerate(channel_names):

    rectangle = Rectangle(
        (i + 0.05, 0.12),
        0.90,
        0.70,
        facecolor="black",
        edgecolor="white",
        linewidth=2
    )

    ax_panels.add_patch(rectangle)
    panel_rectangles.append(rectangle)

    ax_panels.text(
        i + 0.50,
        0.04,
        name,
        ha="center",
        fontsize=10,
        fontweight="bold"
    )

    text = ax_panels.text(
        i + 0.50,
        0.48,
        f"{BASE_BRIGHTNESS}%",
        ha="center",
        va="center",
        fontsize=12,
        color="white",
        fontweight="bold"
    )

    panel_values.append(text)


# ============================================================
# PLAYBACK / TIMING
# ============================================================

start_time = None
song_finished = False


def start_music():
    global start_time

    pygame.mixer.music.play()

    start_time = time.perf_counter()


# ============================================================
# UPDATE FUNCTION
# ============================================================

def update(frame):

    global current_channels
    global current_low
    global current_mid
    global current_high
    global song_finished

    if start_time is None:
        return

    elapsed = time.perf_counter() - start_time

    if elapsed >= duration:

        song_finished = True

        pygame.mixer.music.stop()

        plt.close(fig)

        return

    frame_index = int(
        elapsed * sr / HOP_LENGTH
    )

    frame_index = min(
        frame_index,
        num_frames - 1
    )

    current_spectrum = magnitude[
        :,
        frame_index
    ]

    line_spectrum.set_ydata(
        current_spectrum
    )

    target_low = low_brightness[
        frame_index
    ]

    target_mid = mid_brightness[
        frame_index
    ]

    target_high = high_brightness[
        frame_index
    ]

    current_low = (
        SMOOTHING * target_low +
        (1 - SMOOTHING) * current_low
    )

    current_mid = (
        SMOOTHING * target_mid +
        (1 - SMOOTHING) * current_mid
    )

    current_high = (
        SMOOTHING * target_high +
        (1 - SMOOTHING) * current_high
    )

    target_channels = channel_targets[
        :,
        frame_index
    ]

    current_channels = (
        SMOOTHING * target_channels +
        (1 - SMOOTHING) * current_channels
    )

    for bar, value in zip(
        channel_bars,
        current_channels
    ):
        bar.set_height(value)

    for i in range(6):

        brightness = (
            current_channels[i] / 100
        )

        panel_rectangles[i].set_facecolor(
            (
                brightness,
                brightness,
                brightness
            )
        )

        panel_values[i].set_text(
            f"{current_channels[i]:.0f}%"
        )

    current_beat = beat_strength[
        frame_index
    ]

    current_transient = transient_strength[
        frame_index
    ]

    current_overall = overall_norm[
        frame_index
    ]

    history_window = int(
        10 * sr / HOP_LENGTH
    )

    start_frame = max(
        0,
        frame_index - history_window
    )

    event_frames = np.arange(
        start_frame,
        frame_index + 1
    )

    event_times = (
        event_frames *
        HOP_LENGTH /
        sr
    )

    energy_values = overall_norm[
        event_frames
    ]

    beat_values = beat_strength[
        event_frames
    ]

    transient_values = transient_strength[
        event_frames
    ]

    if len(event_times) > 0:

        energy_line.set_data(
            event_times,
            energy_values
        )

        beat_line.set_data(
            event_times,
            beat_values
        )

        transient_line.set_data(
            event_times,
            transient_values
        )

        ax_events.set_xlim(
            max(0, elapsed - 10),
            max(10, elapsed)
        )

    display_mask = (
        (frequencies >= 20) &
        (frequencies <= min(20000, nyquist))
    )

    display_magnitude = current_spectrum[
        display_mask
    ]

    display_frequencies = frequencies[
        display_mask
    ]

    if len(display_magnitude) > 0:

        dominant_index = np.argmax(
            display_magnitude
        )

        dominant_frequency = (
            display_frequencies[
                dominant_index
            ]
        )

    else:

        dominant_frequency = 0

    band_values_raw = [
        current_low,
        current_mid,
        current_high
    ]

    dominant_band_index = np.argmax(
        band_values_raw
    )

    band_names = [
        "LOW",
        "MID",
        "HIGH"
    ]

    dominant_band = band_names[
        dominant_band_index
    ]

    if current_beat > 0.65:
        beat_state = "STRONG BEAT"
    elif current_beat > 0.25:
        beat_state = "BEAT"
    else:
        beat_state = "—"

    if current_transient > 0.65:
        transient_state = "STRONG IMPACT"
    elif current_transient > 0.25:
        transient_state = "IMPACT"
    else:
        transient_state = "—"

    minutes = int(elapsed // 60)
    seconds = int(elapsed % 60)

    total_minutes = int(duration // 60)
    total_seconds = int(duration % 60)

    tempo_value = np.asarray(tempo).reshape(-1)

    if len(tempo_value) > 0:
        tempo_display = float(tempo_value[0])
    else:
        tempo_display = 0

    status_text.set_text(
        f"SONG PLAYING\n\n"
        f"Time:\n"
        f"{minutes:02d}:{seconds:02d} / "
        f"{total_minutes:02d}:{total_seconds:02d}\n\n"
        f"Tempo:\n"
        f"{tempo_display:.1f} BPM\n\n"
        f"Dominant Frequency:\n"
        f"{dominant_frequency:.0f} Hz\n\n"
        f"Dominant Band:\n"
        f"{dominant_band}\n\n"
        f"Overall Energy:\n"
        f"{current_overall:.2f}\n\n"
        f"Beat Strength:\n"
        f"{current_beat:.2f}  {beat_state}\n\n"
        f"Transient:\n"
        f"{current_transient:.2f}  {transient_state}\n\n"
        f"OUTPUT MODE:\n"
        f"SIMULATED MCU"
    )

    return (
        line_spectrum,
        *channel_bars,
        *panel_rectangles,
        *panel_values,
        energy_line,
        beat_line,
        transient_line,
        status_text
    )


# ============================================================
# CLOSE EVENT
# ============================================================

def on_close(event):

    if pygame.mixer.get_init():
        pygame.mixer.music.stop()

    pygame.quit()


fig.canvas.mpl_connect(
    "close_event",
    on_close
)


# ============================================================
# ANIMATION
# ============================================================

animation = FuncAnimation(
    fig,
    update,
    interval=30,
    blit=False,
    cache_frame_data=False
)


# ============================================================
# START
# ============================================================

print()
print("Beat frames detected :", len(beat_frames))
print("Estimated tempo      :", np.asarray(tempo).reshape(-1))
print()
print("Lighting mapping:")
print("  Outer two panels   -> Highs (Treble)")
print("  Inner/Mid panels   -> Mids")
print("  Middle two panels  -> Low Bass")
print()
print("Starting music...")

start_music()

plt.tight_layout()

plt.show()