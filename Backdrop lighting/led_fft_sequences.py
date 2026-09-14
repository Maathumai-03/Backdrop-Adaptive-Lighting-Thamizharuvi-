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


def compute_band_energies(chunk):
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
    return smoothed_energy


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

    fig, axes = plt.subplots(1, len(BANDS), figsize=(11, 5))
    fig.suptitle("LED Sequences per Frequency Band (live FFT)")

    bars = []
    for ax, (name, lo, hi, seq_name), color in zip(axes, BANDS, BAND_COLORS):
        bar = ax.bar([0], [0.0], color=color, width=0.6)
        ax.set_ylim(0, 1)
        ax.set_xlim(-1, 1)
        ax.set_xticks([])
        ax.set_title(f"{name}\n{lo}-{hi}Hz\n({seq_name})", fontsize=8)
        bars.append(bar[0])
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
                return bars
            chunks_received[0] += 1
            energies = compute_band_energies(chunk)

        for bar, energy in zip(bars, energies):
            bar.set_height(energy)

        if frame_counter[0] % 15 == 0:  # print roughly twice a second
            vals = ", ".join(f"{e:.2f}" for e in energies)
            print(f"frame={frame_counter[0]} chunks={chunks_received[0]} energies=[{vals}]")

        return bars

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