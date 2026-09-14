"""
led_panel_layouts_compare.py
-----------------------------
Matches the real panel design: each panel is a solid edge-lit block (no
per-pixel chase inside a panel), sits at a 30% baseline brightness, and
brightens toward 100% based on audio. Shows 4 different panel-lighting
patterns stacked on one screen, driven by the same audio, so you can
compare them directly:

  1. MULTI-BAND MAPPED - each panel is permanently tied to its own
     frequency range (Low / Mid-Vocal / High). Its brightness tracks
     ONLY that range's amplitude. Multiple panels can brighten together
     for mixed-frequency content (e.g. Low + Mid at once).
  2. MIRROR         - all panels brighten together, in lockstep, driven
                       by overall loudness (not tied to one frequency).
  3. SEQUENTIAL     - only one panel brightens at a time; which panel is
                       "active" cycles over time.
  4. COMBINED (VU)  - panels light up one-by-one, left to right, based
                       on how loud the audio is right now (more panels
                       lit = louder), like a VU meter.

Panel count/frequency-range specifics are simplified (round numbers) —
tell me the real per-panel widths/ranges later and I'll match them exactly.

RUN THIS ON YOUR OWN MACHINE (needs a display, optionally a mic).

SETUP (once):
    pip install numpy sounddevice soundfile matplotlib

USAGE:
    python led_panel_layouts_compare.py --file song.wav
    python led_panel_layouts_compare.py --demo
    python led_panel_layouts_compare.py --mic
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
CHUNK = 1024
SMOOTHING = 0.6
PEAK_DECAY = 0.995
BASELINE = 0.30          # panels never go fully dark - 30% baseline glow
PANEL_SWITCH_FRAMES = 45  # how long each panel stays "active" in SEQUENTIAL mode

# 6 panels: the same 3 frequency ranges, each covering 2 panels.
# Bright, high-saturation colors chosen so brightness changes read clearly
# against a black background (matches real edge-lit LED look).
PANELS = [
    ("Low",       20,   250,   (1.0, 0.15, 0.15)),   # red
    ("Mid-Vocal", 250,  2000,  (0.15, 1.0, 0.3)),    # green
    ("High/Inst", 2000, 16000, (0.2, 0.5, 1.0)),     # blue
    ("High/Inst", 2000, 16000, (0.2, 0.5, 1.0)),     # blue
    ("Mid-Vocal", 250,  2000,  (0.15, 1.0, 0.3)),    # green
    ("Low",       20,   250,   (1.0, 0.15, 0.15)),   # red
]
PANEL_COUNT = len(PANELS)

audio_q = queue.Queue()


# ---------------------------------------------------------------------------
# Audio sources
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


# ---------------------------------------------------------------------------
# FFT -> per-panel band energies + one overall loudness value
# ---------------------------------------------------------------------------
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

    # per-panel bands
    raw = np.array([spectrum[m].max() if m.any() else 0.0 for m in panel_masks])
    band_peak = np.maximum(band_peak * PEAK_DECAY, raw)
    band_norm = np.clip(raw / band_peak, 0, 1)
    band_smoothed = SMOOTHING * band_smoothed + (1 - SMOOTHING) * band_norm

    # overall loudness (whole spectrum, not band-limited)
    overall_raw = spectrum.max()
    overall_peak = max(overall_peak * PEAK_DECAY, overall_raw)
    overall_norm = np.clip(overall_raw / overall_peak, 0, 1)
    overall_smoothed = SMOOTHING * overall_smoothed + (1 - SMOOTHING) * overall_norm

    return band_smoothed, overall_smoothed


# ---------------------------------------------------------------------------
# 4 panel-lighting patterns, each returns PANEL_COUNT brightness values (0-1)
# ---------------------------------------------------------------------------
def layout_multiband(band_energies, overall):
    return BASELINE + (1 - BASELINE) * band_energies


def layout_mirror(band_energies, overall):
    return np.full(PANEL_COUNT, BASELINE + (1 - BASELINE) * overall)


def layout_sequential(band_energies, overall, frame):
    vals = np.full(PANEL_COUNT, BASELINE)
    active = (frame // PANEL_SWITCH_FRAMES) % PANEL_COUNT
    vals[active] = BASELINE + (1 - BASELINE) * overall
    return vals


def layout_combined_vu(band_energies, overall):
    lit = int(round(overall * PANEL_COUNT))
    vals = np.full(PANEL_COUNT, BASELINE)
    vals[:lit] = BASELINE + (1 - BASELINE) * overall
    return vals


# ---------------------------------------------------------------------------
# Visualization - each panel drawn as a solid colored block scaled by brightness
# ---------------------------------------------------------------------------
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

    fig, axes = plt.subplots(4, 1, figsize=(11, 8))
    fig.patch.set_facecolor("black")
    fig.suptitle("Panel lighting patterns - compare (baseline 30% -> 100%)", color="white")
    titles = [
        "1. MULTI-BAND MAPPED - each panel tied to its own frequency range",
        "2. MIRROR - all panels together, overall loudness",
        "3. SEQUENTIAL - one panel active at a time",
        "4. COMBINED (VU) - panels fill left-to-right with loudness",
    ]

    PANEL_GAP = 0.15   # space between panels, so each reads as a separate unit
    EDGE_WIDTH = 9     # thickness of the glowing edge stroke

    glow_patches = []   # soft filled glow behind the edge
    edge_patches = []   # the bright edge-lit outline itself

    for ax, title in zip(axes, titles):
        ax.set_facecolor("black")
        ax.set_xlim(0, PANEL_COUNT)
        ax.set_ylim(0, 1)
        ax.set_title(title, fontsize=9, loc="left", color="white")
        ax.set_xticks([i + 0.5 for i in range(PANEL_COUNT)])
        ax.set_xticklabels([p[0] for p in PANELS], fontsize=7, color="white")
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

        row_glow, row_edge = [], []
        for i, (_, _, _, color) in enumerate(PANELS):
            x0 = i + PANEL_GAP / 2
            width = 1 - PANEL_GAP
            glow = plt.Rectangle((x0, 0.05), width, 0.9, facecolor=color,
                                  edgecolor="none", alpha=0.0, zorder=1)
            edge = plt.Rectangle((x0, 0.05), width, 0.9, facecolor="none",
                                  edgecolor=color, linewidth=EDGE_WIDTH, zorder=2)
            ax.add_patch(glow)
            ax.add_patch(edge)
            row_glow.append(glow)
            row_edge.append(edge)
        glow_patches.append(row_glow)
        edge_patches.append(row_edge)

    plt.tight_layout()

    frame_counter = [0]
    chunks_received = [0]
    base_colors = [p[3] for p in PANELS]

    def render(row_idx, brightness):
        for i, b in enumerate(brightness):
            b = float(np.clip(b, 0, 1))
            glow_patches[row_idx][i].set_alpha(0.25 * b)          # soft glow fill
            edge_patches[row_idx][i].set_edgecolor(
                tuple(np.array(base_colors[i]) * max(b, 0.15))    # bright edge, never fully off
            )

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
                return [p for row in edge_patches for p in row]
            chunks_received[0] += 1
            band_energies, overall = compute_energies(chunk)

        render(0, layout_multiband(band_energies, overall))
        render(1, layout_mirror(band_energies, overall))
        render(2, layout_sequential(band_energies, overall, frame_counter[0]))
        render(3, layout_combined_vu(band_energies, overall))

        if frame_counter[0] % 15 == 0:
            vals = ", ".join(f"{e:.2f}" for e in band_energies)
            print(f"frame={frame_counter[0]} chunks={chunks_received[0]} bands=[{vals}] overall={overall:.2f}")

        return [p for row in edge_patches for p in row] + [p for row in glow_patches for p in row]

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