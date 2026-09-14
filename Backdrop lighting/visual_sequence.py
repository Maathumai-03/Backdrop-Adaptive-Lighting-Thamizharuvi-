"""
led_fft_sequences.py
---------------------
Reads audio (from a file by default, or a live microphone with --mic),
runs an FFT on rolling chunks, splits the spectrum into 5 common
frequency bands, and drives a distinct visual "sequence" (LED pattern)
per band. Each band is shown as its own simulated LED strip (a row of
pixels) in a live matplotlib window, so you can see the patterns
before wiring up real WS2812B strips.

RUN THIS ON YOUR OWN MACHINE (not in a sandbox) because it needs:
  - a display (matplotlib window)
  - optionally your microphone

SETUP (run once):
    pip install numpy sounddevice soundfile matplotlib

USAGE:
    # Default: plays/analyzes an audio file
    python led_fft_sequences.py --file path/to/song.wav

    # If you omit --file, it generates a short built-in test tone sweep
    python led_fft_sequences.py

    # Live microphone instead of a file
    python led_fft_sequences.py --mic

Press Ctrl+C or close the plot window to stop.
"""

import argparse
import collections
import queue
import sys
import tempfile
import os

import numpy as np
import sounddevice as sd
import soundfile as sf
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
SAMPLE_RATE = 44100
CHUNK = 1024              # samples per FFT frame
NUM_PIXELS = 30           # simulated LEDs per strip/band
SMOOTHING = 0.6           # exponential smoothing factor (0=no smoothing)

# 5 common frequency bands (Hz) covering the audible spectrum
BANDS = [
    ("Sub-Bass", 20, 60,    "Pulse"),
    ("Bass",     60, 250,   "Chase"),
    ("Low-Mid",  250, 500,  "Fill"),
    ("Mid",      500, 2000, "Sparkle"),
    ("High",     2000, 16000, "Strobe"),
]

BAND_COLORS = [
    (0.8, 0.1, 0.1),   # Sub-Bass: red
    (0.9, 0.5, 0.0),   # Bass: orange
    (0.9, 0.9, 0.0),   # Low-Mid: yellow
    (0.1, 0.8, 0.3),   # Mid: green
    (0.2, 0.4, 1.0),   # High: blue
]

audio_q = queue.Queue()


# ---------------------------------------------------------------------------
# Audio sources
# ---------------------------------------------------------------------------
def start_mic_stream():
    """Push live mic audio chunks onto audio_q."""

    def callback(indata, frames, time_info, status):
        if status:
            print(status, file=sys.stderr)
        audio_q.put(indata[:, 0].copy())

    stream = sd.InputStream(
        channels=1,
        samplerate=SAMPLE_RATE,
        blocksize=CHUNK,
        callback=callback,
    )
    stream.start()
    return stream


def start_file_stream(path):
    """Play an audio file and push the exact chunks being played onto audio_q,
    so the visualization stays in sync with what you hear."""
    data, sr = sf.read(path, dtype="float32", always_2d=True)
    data = data[:, 0]  # mono

    if sr != SAMPLE_RATE:
        # simple resample-free fallback: just play at native rate
        rate = sr
    else:
        rate = SAMPLE_RATE

    pos = 0

    def callback(outdata, frames, time_info, status):
        nonlocal pos
        if status:
            print(status, file=sys.stderr)
        chunk = data[pos:pos + frames]
        if len(chunk) < frames:
            chunk = np.pad(chunk, (0, frames - len(chunk)))
            raise sd.CallbackStop
        pos += frames
        outdata[:, 0] = chunk
        audio_q.put(chunk.copy())

    stream = sd.OutputStream(
        channels=1,
        samplerate=rate,
        blocksize=CHUNK,
        callback=callback,
    )
    stream.start()
    return stream


def make_test_tone():
    """No file/mic given: build a short sweeping tone so you can still see
    all 5 bands react in turn."""
    dur = 12.0
    t = np.linspace(0, dur, int(SAMPLE_RATE * dur), endpoint=False)
    sweep = np.sin(2 * np.pi * (30 + (8000 - 30) * (t / dur)) * t) * 0.5
    path = os.path.join(tempfile.gettempdir(), "led_test_tone.wav")
    sf.write(path, sweep.astype(np.float32), SAMPLE_RATE)
    return path


# ---------------------------------------------------------------------------
# FFT -> band energies
# ---------------------------------------------------------------------------
freqs = np.fft.rfftfreq(CHUNK, d=1.0 / SAMPLE_RATE)
band_masks = [(freqs >= lo) & (freqs < hi) for _, lo, hi, _ in BANDS]
smoothed_energy = np.zeros(len(BANDS))
band_peak = np.full(len(BANDS), 1e-6)   # running peak per band, for auto-gain
PEAK_DECAY = 0.995                      # how slowly the auto-gain ceiling relaxes

# --- beat/onset detection state ---
HISTORY_LEN = 43          # ~1 second of history at ~23ms/chunk
beat_history = [collections.deque(maxlen=HISTORY_LEN) for _ in BANDS]
beat_level = np.zeros(len(BANDS))       # what gets displayed in beat mode (attack/decay envelope)
BEAT_THRESHOLD = 1.5      # how far above the local average counts as a "hit"
BEAT_DECAY = 0.80         # how fast a flash fades each frame


def compute_band_energies(chunk):
    """Continuous mode: raw normalized loudness per band, right now."""
    global smoothed_energy, band_peak
    if len(chunk) < CHUNK:
        chunk = np.pad(chunk, (0, CHUNK - len(chunk)))
    windowed = chunk * np.hanning(len(chunk))
    spectrum = np.abs(np.fft.rfft(windowed))

    # use the strongest bin in each band (not the mean) so a single
    # dominant tone isn't diluted by mostly-empty bins in a wide band
    raw = np.array([
        spectrum[mask].max() if mask.any() else 0.0
        for mask in band_masks
    ])

    # auto-gain: track a slowly-decaying peak per band and normalize
    # against it, so this works regardless of file volume / mic gain
    band_peak = np.maximum(band_peak * PEAK_DECAY, raw)
    normalized = np.clip(raw / band_peak, 0, 1)

    smoothed_energy = SMOOTHING * smoothed_energy + (1 - SMOOTHING) * normalized
    return smoothed_energy, raw


def compute_beat_energies(raw):
    """Beat mode: flash a band only when its energy spikes well above its
    own recent (~1s) average, then let the flash decay fast. This is what
    actually tracks percussive hits instead of continuous loudness."""
    global beat_level
    for i, r in enumerate(raw):
        hist = beat_history[i]
        local_avg = (sum(hist) / len(hist)) if hist else 0.0
        is_hit = r > max(local_avg * BEAT_THRESHOLD, 1e-4)
        hist.append(r)

        if is_hit:
            beat_level[i] = 1.0
        else:
            beat_level[i] *= BEAT_DECAY
    return np.clip(beat_level, 0, 1)



# ---------------------------------------------------------------------------
# The 5 sequences: each takes (energy 0-1, frame_count) -> (NUM_PIXELS, 3) array
# ---------------------------------------------------------------------------
def seq_pulse(energy, frame, color):
    """Sub-Bass: whole strip brightens/dims together with the beat."""
    strip = np.tile(color, (NUM_PIXELS, 1)) * energy
    return strip


def seq_chase(energy, frame, color):
    """Bass: a bright dot chases down the strip, speed tied to energy."""
    strip = np.zeros((NUM_PIXELS, 3))
    pos = int(frame * (1 + energy * 4)) % NUM_PIXELS
    for i in range(NUM_PIXELS):
        dist = min(abs(i - pos), NUM_PIXELS - abs(i - pos))
        fade = max(0, 1 - dist / 4)
        strip[i] = np.array(color) * fade * energy
    return strip


def seq_fill(energy, frame, color):
    """Low-Mid: VU-meter style fill from the left."""
    lit = int(energy * NUM_PIXELS)
    strip = np.zeros((NUM_PIXELS, 3))
    strip[:lit] = color
    return strip


def seq_sparkle(energy, frame, color):
    """Mid: random pixels flicker on, density tied to energy."""
    strip = np.zeros((NUM_PIXELS, 3))
    n_sparks = int(energy * NUM_PIXELS)
    idx = np.random.choice(NUM_PIXELS, size=n_sparks, replace=False) if n_sparks else []
    strip[idx] = color
    return strip


def seq_strobe(energy, frame, color):
    """High: whole strip flashes on/off when energy crosses a threshold."""
    on = energy > 0.35 and (frame % 2 == 0)
    strip = np.tile(color, (NUM_PIXELS, 1)) if on else np.zeros((NUM_PIXELS, 3))
    return strip


SEQUENCE_FUNCS = [seq_pulse, seq_chase, seq_fill, seq_sparkle, seq_strobe]


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type=str, default=None, help="Path to a wav/audio file")
    parser.add_argument("--mic", action="store_true", help="Use live microphone instead of a file")
    parser.add_argument("--demo", action="store_true",
                         help="Skip audio entirely; drive the 5 sequences with synthetic "
                              "energy so you can preview what each pattern looks like")
    parser.add_argument("--mode", choices=["continuous", "beat"], default="continuous",
                         help="'continuous' = raw loudness per band (default). "
                              "'beat' = flash only when a band spikes above its own "
                              "recent average, i.e. actual beat/onset detection.")
    args = parser.parse_args()

    stream = None
    if args.demo:
        print("Demo mode: no audio, just previewing the 5 sequence patterns.")
    elif args.mic:
        stream = start_mic_stream()
        print("Listening to microphone... Ctrl+C to stop.")
    else:
        path = args.file or make_test_tone()
        stream = start_file_stream(path)
        print(f"Playing and analyzing: {path}")

    fig, axes = plt.subplots(2, len(BANDS), figsize=(12, 6),
                              gridspec_kw={"height_ratios": [1, 2]})
    fig.suptitle(f"LED Sequences per Frequency Band ({args.mode} mode)")

    bars = []
    strips = []
    for col, ((name, lo, hi, seq_name), color) in enumerate(zip(BANDS, BAND_COLORS)):
        bar_ax = axes[0, col]
        bar = bar_ax.bar([0], [0.0], color=color, width=0.6)
        bar_ax.set_ylim(0, 1)
        bar_ax.set_xlim(-1, 1)
        bar_ax.set_xticks([])
        bar_ax.set_title(f"{name}\n{lo}-{hi}Hz\n({seq_name})", fontsize=8)
        bars.append(bar[0])

        strip_ax = axes[1, col]
        strip_img = strip_ax.imshow(np.zeros((NUM_PIXELS, 1, 3)), aspect="auto",
                                     vmin=0, vmax=1, origin="upper")
        strip_ax.set_xticks([])
        strip_ax.set_yticks([])
        if col == 0:
            strip_ax.set_ylabel("pixel 0 (top) -> 29 (bottom)", fontsize=7)
        strips.append(strip_img)
    plt.tight_layout()

    frame_counter = [0]
    chunks_received = [0]

    def update(_):
        frame_counter[0] += 1

        if args.demo:
            # synthetic energies so you can see each pattern with no audio at all:
            # a slow sweep through the 5 bands, one at a time
            t = frame_counter[0]
            phase = (t // 20) % len(BANDS)
            energies = np.zeros(len(BANDS))
            energies[phase] = 0.5 + 0.5 * abs(np.sin(t / 5))
        else:
            try:
                chunk = audio_q.get(timeout=0.5)
            except queue.Empty:
                print("[no audio chunk received this tick]")
                return bars + strips
            chunks_received[0] += 1
            smoothed, raw = compute_band_energies(chunk)
            energies = compute_beat_energies(raw) if args.mode == "beat" else smoothed

        for bar, strip_img, fn, color, energy in zip(bars, strips, SEQUENCE_FUNCS, BAND_COLORS, energies):
            bar.set_height(energy)
            pixels = fn(energy, frame_counter[0], color)          # (NUM_PIXELS, 3)
            strip_img.set_data(pixels.reshape(NUM_PIXELS, 1, 3))

        if frame_counter[0] % 15 == 0:  # print roughly twice a second
            vals = ", ".join(f"{e:.2f}" for e in energies)
            print(f"frame={frame_counter[0]} chunks={chunks_received[0]} energies=[{vals}]")

        return bars + strips

    ani = animation.FuncAnimation(fig, update, interval=30, blit=False, cache_frame_data=False)

    try:
        plt.show()
    except KeyboardInterrupt:
        pass
    finally:
        if stream is not None:
            stream.stop()
            stream.close()


if __name__ == "__main__":
    main()