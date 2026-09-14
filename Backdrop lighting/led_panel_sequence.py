"""
led_panel_sequences.py
-----------------------
Tracks ONE frequency band (Low-Mid, 250-500Hz) from an audio file or mic,
and drives a "Fill" sequence across a 6-panel LED setup (each panel has
its own row of LEDs). Since you weren't sure yet how the 6 panels should
relate to the sequence, this shows all 3 layout options side by side at
once, driven by the same audio, so you can compare them directly:

  1. COMBINED   - all 6 panels act as one continuous strip; the fill
                  sweeps across all of them together, panel boundaries
                  are just visual dividers.
  2. SEQUENTIAL - only ONE panel is active at a time; it fills, then
                  the "active" panel moves to the next one in order.
  3. MIRROR     - all 6 panels show the identical fill pattern at the
                  same time, in lockstep.

RUN THIS ON YOUR OWN MACHINE (needs a display, and optionally a mic).

SETUP (once):
    pip install numpy sounddevice soundfile matplotlib

USAGE:
    python led_panel_sequences.py --file song.wav
    python led_panel_sequences.py --file song.wav --mode beat
    python led_panel_sequences.py --mic
    python led_panel_sequences.py            # built-in test tone, no file needed
    python led_panel_sequences.py --demo      # no audio at all, synthetic pulse
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
CHUNK = 1024
SMOOTHING = 0.6

PANEL_COUNT = 6
PIXELS_PER_PANEL = 10
TOTAL_PIXELS = PANEL_COUNT * PIXELS_PER_PANEL

BAND_NAME, BAND_LO, BAND_HI = "Low-Mid", 250, 500
FILL_COLOR = (0.9, 0.9, 0.0)   # yellow, matches the earlier Low-Mid color
GAP_COLOR = (0.15, 0.15, 0.15)  # dark gray divider between panels

PANEL_SWITCH_FRAMES = 45   # how many frames the "active" panel stays active in SEQUENTIAL mode

audio_q = queue.Queue()


# ---------------------------------------------------------------------------
# Audio sources (same as before)
# ---------------------------------------------------------------------------
def start_mic_stream():
    def callback(indata, frames, time_info, status):
        if status:
            print(status, file=sys.stderr)
        audio_q.put(indata[:, 0].copy())

    stream = sd.InputStream(channels=1, samplerate=SAMPLE_RATE, blocksize=CHUNK, callback=callback)
    stream.start()
    return stream


def start_file_stream(path):
    data, sr = sf.read(path, dtype="float32", always_2d=True)
    data = data[:, 0]
    rate = sr
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

    stream = sd.OutputStream(channels=1, samplerate=rate, blocksize=CHUNK, callback=callback)
    stream.start()
    return stream


def make_test_tone():
    dur = 12.0
    t = np.linspace(0, dur, int(SAMPLE_RATE * dur), endpoint=False)
    sweep = np.sin(2 * np.pi * (200 + 200 * (0.5 + 0.5 * np.sin(t * 0.5))) * t) * 0.5
    path = os.path.join(tempfile.gettempdir(), "led_test_tone.wav")
    sf.write(path, sweep.astype(np.float32), SAMPLE_RATE)
    return path


# ---------------------------------------------------------------------------
# FFT -> single-band energy (Low-Mid only)
# ---------------------------------------------------------------------------
freqs = np.fft.rfftfreq(CHUNK, d=1.0 / SAMPLE_RATE)
band_mask = (freqs >= BAND_LO) & (freqs < BAND_HI)

smoothed_energy = 0.0
band_peak = 1e-6
PEAK_DECAY = 0.995

HISTORY_LEN = 43
beat_history = collections.deque(maxlen=HISTORY_LEN)
beat_level = 0.0
BEAT_THRESHOLD = 1.5
BEAT_DECAY = 0.80


def compute_energy(chunk):
    global smoothed_energy, band_peak
    if len(chunk) < CHUNK:
        chunk = np.pad(chunk, (0, CHUNK - len(chunk)))
    windowed = chunk * np.hanning(len(chunk))
    spectrum = np.abs(np.fft.rfft(windowed))

    raw = spectrum[band_mask].max() if band_mask.any() else 0.0
    band_peak = max(band_peak * PEAK_DECAY, raw)
    normalized = np.clip(raw / band_peak, 0, 1)
    smoothed_energy = SMOOTHING * smoothed_energy + (1 - SMOOTHING) * normalized
    return smoothed_energy, raw


def compute_beat_energy(raw):
    global beat_level
    local_avg = (sum(beat_history) / len(beat_history)) if beat_history else 0.0
    is_hit = raw > max(local_avg * BEAT_THRESHOLD, 1e-4)
    beat_history.append(raw)
    if is_hit:
        beat_level = 1.0
    else:
        beat_level *= BEAT_DECAY
    return float(np.clip(beat_level, 0, 1))


# ---------------------------------------------------------------------------
# Panel-layout renderers: each returns a (PANEL_COUNT, PIXELS_PER_PANEL, 3) array
# ---------------------------------------------------------------------------
def layout_combined(energy):
    lit = int(energy * TOTAL_PIXELS)
    flat = np.zeros((TOTAL_PIXELS, 3))
    flat[:lit] = FILL_COLOR
    return flat.reshape(PANEL_COUNT, PIXELS_PER_PANEL, 3)


def layout_sequential(energy, frame):
    active_panel = (frame // PANEL_SWITCH_FRAMES) % PANEL_COUNT
    panels = np.zeros((PANEL_COUNT, PIXELS_PER_PANEL, 3))
    lit = int(energy * PIXELS_PER_PANEL)
    panels[active_panel, :lit] = FILL_COLOR
    return panels


def layout_mirror(energy):
    lit = int(energy * PIXELS_PER_PANEL)
    panels = np.zeros((PANEL_COUNT, PIXELS_PER_PANEL, 3))
    panels[:, :lit] = FILL_COLOR
    return panels


def to_image_with_gaps(panels):
    """(PANEL_COUNT, PIXELS_PER_PANEL, 3) -> single row image with a 1px gray
    divider between each panel, purely for display clarity."""
    rows = []
    for i, panel in enumerate(panels):
        rows.append(panel)
        if i < len(panels) - 1:
            rows.append(np.array(GAP_COLOR).reshape(1, 3))
    return np.concatenate(rows, axis=0).reshape(1, -1, 3)


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type=str, default=None)
    parser.add_argument("--mic", action="store_true")
    parser.add_argument("--demo", action="store_true",
                         help="No audio; drives a synthetic pulsing energy so you can preview layouts")
    parser.add_argument("--mode", choices=["continuous", "beat"], default="continuous")
    args = parser.parse_args()

    stream = None
    if args.demo:
        print("Demo mode: no audio, synthetic energy driving the 3 layouts.")
    elif args.mic:
        stream = start_mic_stream()
        print("Listening to microphone... Ctrl+C to stop.")
    else:
        path = args.file or make_test_tone()
        stream = start_file_stream(path)
        print(f"Playing and analyzing: {path} (tracking only {BAND_NAME} {BAND_LO}-{BAND_HI}Hz)")

    fig, axes = plt.subplots(3, 1, figsize=(12, 6))
    fig.suptitle(f"Low-Mid Fill sequence across 6 panels ({args.mode} mode) - compare layouts")
    titles = [
        "1. COMBINED - all 6 panels as one strip",
        "2. SEQUENTIAL - one panel active at a time",
        "3. MIRROR - identical pattern on all panels",
    ]
    images = []
    for ax, title in zip(axes, titles):
        img = ax.imshow(np.zeros((1, TOTAL_PIXELS + PANEL_COUNT - 1, 3)), aspect="auto", vmin=0, vmax=1)
        ax.set_title(title, fontsize=10, loc="left")
        ax.set_xticks([])
        ax.set_yticks([])
        images.append(img)
    plt.tight_layout()

    frame_counter = [0]
    chunks_received = [0]

    def update(_):
        frame_counter[0] += 1

        if args.demo:
            t = frame_counter[0]
            energy = 0.5 + 0.5 * abs(np.sin(t / 15))
        else:
            try:
                chunk = audio_q.get(timeout=0.5)
            except queue.Empty:
                print("[no audio chunk received this tick]")
                return images
            chunks_received[0] += 1
            smoothed, raw = compute_energy(chunk)
            energy = compute_beat_energy(raw) if args.mode == "beat" else smoothed

        combined = to_image_with_gaps(layout_combined(energy))
        sequential = to_image_with_gaps(layout_sequential(energy, frame_counter[0]))
        mirror = to_image_with_gaps(layout_mirror(energy))

        images[0].set_data(combined)
        images[1].set_data(sequential)
        images[2].set_data(mirror)

        if frame_counter[0] % 15 == 0:
            print(f"frame={frame_counter[0]} chunks={chunks_received[0]} energy={energy:.2f}")

        return images

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