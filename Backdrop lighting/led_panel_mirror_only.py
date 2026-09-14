"""
led_panel_mirror_only.py
-------------------------
Just the MIRROR pattern: all 6 panels brighten together in lockstep,
driven by overall loudness (not tied to any one frequency range).

RUN THIS ON YOUR OWN MACHINE (needs a display, optionally a mic).

SETUP (once):
    pip install numpy sounddevice soundfile matplotlib

USAGE:
    python led_panel_mirror_only.py --file song.wav
    python led_panel_mirror_only.py --demo
    python led_panel_mirror_only.py --mic
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

SAMPLE_RATE = 44100
CHUNK = 1024
SMOOTHING = 0.6
PEAK_DECAY = 0.995
BASELINE = 0.30

PANELS = [
    ("Low",       20,   250,   (1.0, 0.15, 0.15)),
    ("Mid-Vocal", 250,  2000,  (0.15, 1.0, 0.3)),
    ("High/Inst", 2000, 16000, (0.2, 0.5, 1.0)),
    ("High/Inst", 2000, 16000, (0.2, 0.5, 1.0)),
    ("Mid-Vocal", 250,  2000,  (0.15, 1.0, 0.3)),
    ("Low",       20,   250,   (1.0, 0.15, 0.15)),
]
PANEL_COUNT = len(PANELS)

audio_q = queue.Queue()


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

    stream = sd.OutputStream(channels=1, samplerate=sr, blocksize=CHUNK, callback=callback)
    stream.start()
    return stream


def make_test_tone():
    dur = 12.0
    t = np.linspace(0, dur, int(SAMPLE_RATE * dur), endpoint=False)
    sweep = np.sin(2 * np.pi * (60 + (10000 - 60) * (t / dur)) * t) * 0.5
    path = os.path.join(tempfile.gettempdir(), "led_test_tone.wav")
    sf.write(path, sweep.astype(np.float32), SAMPLE_RATE)
    return path


freqs = np.fft.rfftfreq(CHUNK, d=1.0 / SAMPLE_RATE)
panel_masks = [(freqs >= lo) & (freqs < hi) for _, lo, hi, _ in PANELS]

band_smoothed = np.zeros(PANEL_COUNT)
band_peak = np.full(PANEL_COUNT, 1e-6)
overall_smoothed = 0.0
overall_peak = 1e-6


def compute_energies(chunk):
    global band_smoothed, band_peak, overall_smoothed, overall_peak
    if len(chunk) < CHUNK:
        chunk = np.pad(chunk, (0, CHUNK - len(chunk)))
    windowed = chunk * np.hanning(len(chunk))
    spectrum = np.abs(np.fft.rfft(windowed))

    raw = np.array([spectrum[m].max() if m.any() else 0.0 for m in panel_masks])
    band_peak = np.maximum(band_peak * PEAK_DECAY, raw)
    band_norm = np.clip(raw / band_peak, 0, 1)
    band_smoothed = SMOOTHING * band_smoothed + (1 - SMOOTHING) * band_norm

    overall_raw = spectrum.max()
    overall_peak = max(overall_peak * PEAK_DECAY, overall_raw)
    overall_norm = np.clip(overall_raw / overall_peak, 0, 1)
    overall_smoothed = SMOOTHING * overall_smoothed + (1 - SMOOTHING) * overall_norm

    return band_smoothed, overall_smoothed


def layout(band_energies, overall, frame):
    return np.full(PANEL_COUNT, BASELINE + (1 - BASELINE) * overall)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type=str, default=None)
    parser.add_argument("--mic", action="store_true")
    parser.add_argument("--demo", action="store_true")
    args = parser.parse_args()

    stream = None
    if args.demo:
        print("Demo mode: synthetic energy, no audio needed.")
    elif args.mic:
        stream = start_mic_stream()
        print("Listening to microphone... Ctrl+C to stop.")
    else:
        path = args.file or make_test_tone()
        stream = start_file_stream(path)
        print(f"Playing and analyzing: {path}")

    fig, ax = plt.subplots(figsize=(11, 3))
    fig.patch.set_facecolor("black")
    fig.suptitle("Mirror - all panels together, overall loudness", color="white")

    PANEL_GAP = 0.15
    EDGE_WIDTH = 9

    ax.set_facecolor("black")
    ax.set_xlim(0, PANEL_COUNT)
    ax.set_ylim(0, 1)
    ax.set_xticks([i + 0.5 for i in range(PANEL_COUNT)])
    ax.set_xticklabels([p[0] for p in PANELS], fontsize=8, color="white")
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    glow_patches, edge_patches = [], []
    for i, (_, _, _, color) in enumerate(PANELS):
        x0 = i + PANEL_GAP / 2
        width = 1 - PANEL_GAP
        glow = plt.Rectangle((x0, 0.05), width, 0.9, facecolor=color, edgecolor="none", alpha=0.0, zorder=1)
        edge = plt.Rectangle((x0, 0.05), width, 0.9, facecolor="none", edgecolor=color, linewidth=EDGE_WIDTH, zorder=2)
        ax.add_patch(glow)
        ax.add_patch(edge)
        glow_patches.append(glow)
        edge_patches.append(edge)

    plt.tight_layout()

    frame_counter = [0]
    chunks_received = [0]
    base_colors = [p[3] for p in PANELS]

    def render(brightness):
        for i, b in enumerate(brightness):
            b = float(np.clip(b, 0, 1))
            glow_patches[i].set_alpha(0.25 * b)
            edge_patches[i].set_edgecolor(tuple(np.array(base_colors[i]) * max(b, 0.15)))

    def update(_):
        frame_counter[0] += 1

        if args.demo:
            t = frame_counter[0]
            phase = (t // 30) % PANEL_COUNT
            band_energies = np.full(PANEL_COUNT, 0.1)
            band_energies[phase] = 0.5 + 0.5 * abs(np.sin(t / 8))
            overall = band_energies.max()
        else:
            try:
                chunk = audio_q.get(timeout=0.5)
            except queue.Empty:
                print("[no audio chunk received this tick]")
                return edge_patches
            chunks_received[0] += 1
            band_energies, overall = compute_energies(chunk)

        render(layout(band_energies, overall, frame_counter[0]))

        if frame_counter[0] % 15 == 0:
            vals = ", ".join(f"{e:.2f}" for e in band_energies)
            print(f"frame={frame_counter[0]} chunks={chunks_received[0]} bands=[{vals}] overall={overall:.2f}")

        return edge_patches + glow_patches

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
