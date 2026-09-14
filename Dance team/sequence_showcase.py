#!/usr/bin/env python3
"""
Sequence showcase — live preview of the 8 backdrop lighting patterns,
composited onto your actual stage photo, cycling automatically so you
can demo it straight to the dance team on your laptop screen.

SETUP:
    pip install numpy pillow matplotlib sounddevice soundfile

RUN (silent, timer/random-driven sequences only look their best, but
the two audio-reactive ones will just breathe gently with no input):
    python sequence_showcase.py

RUN (with the actual song — makes the reactive sequences meaningful):
    python sequence_showcase.py --file path/to/song.wav

RUN (using mic input instead of a file):
    python sequence_showcase.py --mic

Keep stage_photo.png in the same folder as this script.
Press Ctrl+C to quit.
"""

import argparse
import sys
import threading
import time

import numpy as np
from PIL import Image, ImageFilter

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# ---------------------------------------------------------------
# Panel geometry (calibrated against stage_photo.png specifically)
# ---------------------------------------------------------------
PANELS = {
    "L_front": (60, 295, 177, 510),
    "L_mid":   (228, 205, 316, 300),
    "L_back":  (124, 122, 226, 310),
    "R_front": (693, 164, 787, 338),
    "R_mid":   (763, 256, 868, 449),
    "R_back":  (873, 154, 1015, 495),
}
PAIRS = {
    "front": ("L_front", "R_front"),
    "mid":   ("L_mid", "R_mid"),
    "back":  ("L_back", "R_back"),
}
LEFT_PANELS = ["L_front", "L_mid", "L_back"]
RIGHT_PANELS = ["R_front", "R_mid", "R_back"]
CHASE_ORDER = ["L_back", "L_mid", "L_front", "R_front", "R_mid", "R_back"]

FLOOR = 0.30
DIM_COLOR = np.array([55, 38, 20])
BRIGHT_COLOR = np.array([255, 195, 110])

SCALE = 0.55          # working resolution multiplier (speed vs quality)
BLUR_RADIUS = 12       # fixed glow blur radius at working resolution


# ---------------------------------------------------------------
# Image compositing
# ---------------------------------------------------------------
def load_base(photo_path):
    img = Image.open(photo_path).convert("RGB")
    w, h = img.size
    small = img.resize((int(w * SCALE), int(h * SCALE)))
    scaled_panels = {
        name: tuple(int(v * SCALE) for v in box) for name, box in PANELS.items()
    }
    return np.array(small), scaled_panels


def yellow_fill_mask(crop_arr):
    R, G, B = crop_arr[:, :, 0].astype(int), crop_arr[:, :, 1].astype(int), crop_arr[:, :, 2].astype(int)
    return (R > 150) & (G > 120) & (B < 150) & (R - B > 40)


def screen_blend(base, glow):
    return 255 - (255 - base) * (255 - glow) / 255


def compute_frame(base_arr, panels, brightness, blackout=False):
    H, W = base_arr.shape[:2]
    out = base_arr.copy().astype(float)
    glow_layer = np.zeros_like(out)

    for name, (x0, y0, x1, y1) in panels.items():
        b = 0.0 if blackout else brightness.get(name, FLOOR)
        crop = base_arr[y0:y1, x0:x1]
        mask = yellow_fill_mask(crop)

        fill_color = DIM_COLOR + (BRIGHT_COLOR - DIM_COLOR) * b
        panel_out = crop.copy().astype(float)
        panel_out[mask] = fill_color
        out[y0:y1, x0:x1][mask] = panel_out[mask]

        if b > 0.02:
            glow_src = np.zeros((y1 - y0, x1 - x0, 3), dtype=np.uint8)
            glow_src[mask] = fill_color.astype(np.uint8)
            glow_img = Image.fromarray(glow_src).filter(ImageFilter.GaussianBlur(radius=BLUR_RADIUS))
            big_glow = Image.new("RGB", (W, H), (0, 0, 0))
            big_glow.paste(glow_img, (x0, y0))
            big_glow = big_glow.filter(ImageFilter.GaussianBlur(radius=BLUR_RADIUS))
            glow_layer += np.array(big_glow).astype(float) * (0.5 * b)

    out = screen_blend(out, glow_layer)
    return np.clip(out, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------
# Audio-reactive state (shared with sequence generators)
# ---------------------------------------------------------------
audio_state = {"amp": 0.0, "low": 0.0, "mid": 0.0, "high": 0.0,
               "beat_count": 0, "last_beat_time": time.monotonic(), "last_beat_amp": 0.0}
_peak = {"amp": 1e-4, "low": 1e-4, "mid": 1e-4, "high": 1e-4}
_rolling = {"avg": 1e-4}
AUDIO_ENABLED = False


def _normalize(key, value):
    _peak[key] = max(value, _peak[key] * 0.995)
    return float(np.clip(value / (_peak[key] + 1e-9), 0, 1))


def _detect_onset(amp):
    now = time.monotonic()
    _rolling["avg"] = _rolling["avg"] * 0.9 + amp * 0.1
    is_onset = (
        amp > _rolling["avg"] * 1.5
        and amp > 0.001
        and (now - audio_state["last_beat_time"]) > 0.12
    )
    if is_onset:
        audio_state["beat_count"] += 1
        audio_state["last_beat_time"] = now
        audio_state["last_beat_amp"] = float(np.clip(amp / (_rolling["avg"] + 1e-9) / 3, 0, 1))


def _beat_decay(decay=0.4):
    """0-1 envelope that spikes at the last detected beat and fades out."""
    dt = time.monotonic() - audio_state["last_beat_time"]
    return max(0.0, 1.0 - dt / decay)


def audio_callback_factory(rate):
    def callback(indata, frames, time_info, status):
        samples = indata[:, 0]
        amp = np.sqrt(np.mean(samples ** 2))
        audio_state["amp"] = _normalize("amp", amp)
        _detect_onset(amp)

        spectrum = np.abs(np.fft.rfft(samples))
        freqs = np.fft.rfftfreq(len(samples), 1.0 / rate)
        low = spectrum[(freqs >= 20) & (freqs < 250)].mean() if np.any((freqs >= 20) & (freqs < 250)) else 0
        mid = spectrum[(freqs >= 250) & (freqs < 2000)].mean() if np.any((freqs >= 250) & (freqs < 2000)) else 0
        high = spectrum[(freqs >= 2000) & (freqs < 16000)].mean() if np.any((freqs >= 2000) & (freqs < 16000)) else 0
        audio_state["low"] = _normalize("low", low)
        audio_state["mid"] = _normalize("mid", mid)
        audio_state["high"] = _normalize("high", high)
    return callback


def start_audio(args):
    import sounddevice as sd

    RATE = 44100
    CHUNK = 1024

    if args.mic:
        stream = sd.InputStream(channels=1, samplerate=RATE, blocksize=CHUNK,
                                 callback=audio_callback_factory(RATE))
        stream.start()
        print("[audio] using microphone input")
        return

    if args.file:
        import soundfile as sf
        data, file_rate = sf.read(args.file, always_2d=True)
        data = data[:, 0]
        pos = [0]
        cb = audio_callback_factory(file_rate)

        def player():
            with sd.OutputStream(samplerate=file_rate, channels=1) as out:
                while pos[0] < len(data):
                    chunk = data[pos[0]:pos[0] + CHUNK]
                    if len(chunk) == 0:
                        break
                    out.write(chunk.astype(np.float32).reshape(-1, 1))
                    cb(chunk.reshape(-1, 1), len(chunk), None, None)
                    pos[0] += CHUNK

        threading.Thread(target=player, daemon=True).start()
        print(f"[audio] playing file: {args.file}")
        return

    print("[audio] no --file or --mic given; reactive sequences will idle gently")


# ---------------------------------------------------------------
# Sequence generators — each returns (brightness_dict, blackout_bool)
# ---------------------------------------------------------------
def seq_baseline(t):
    return {}, False


def seq_depth_sweep(t):
    order = ["front", "mid", "back"]
    if AUDIO_ENABLED:
        idx = audio_state["beat_count"] % 3
        intensity = _beat_decay(decay=0.5) * (0.4 + 0.6 * audio_state["last_beat_amp"])
    else:
        # 4.5s total: 0-1.5 front, 1.5-3 mid, 3-4.5 back
        seg = t % 4.5
        idx = int(seg // 1.5)
        local_t = seg % 1.5
        intensity = np.sin(local_t / 1.5 * np.pi)  # ramp up then down
    active_pair = PAIRS[order[idx]]
    return {p: FLOOR + (1 - FLOOR) * intensity for p in active_pair}, False


def seq_lr_alternate(t):
    if AUDIO_ENABLED:
        panels = LEFT_PANELS if audio_state["beat_count"] % 2 == 0 else RIGHT_PANELS
        intensity = _beat_decay(decay=0.5) * (0.4 + 0.6 * audio_state["last_beat_amp"])
    else:
        seg = t % 4.0
        if seg < 2.0:
            panels = LEFT_PANELS
            intensity = np.sin(seg / 2.0 * np.pi)
        else:
            panels = RIGHT_PANELS
            intensity = np.sin((seg - 2.0) / 2.0 * np.pi)
    return {p: FLOOR + (1 - FLOOR) * intensity for p in panels}, False


def seq_chase(t):
    if AUDIO_ENABLED:
        idx = audio_state["beat_count"] % len(CHASE_ORDER)
        intensity = _beat_decay(decay=0.4) * (0.4 + 0.6 * audio_state["last_beat_amp"])
    else:
        per_panel = 0.5
        total = per_panel * len(CHASE_ORDER)
        seg = t % total
        idx = int(seg // per_panel)
        local_t = (seg % per_panel) / per_panel
        intensity = np.sin(local_t * np.pi)
    panel = CHASE_ORDER[idx]
    return {panel: FLOOR + (1 - FLOOR) * intensity}, False


_sparkle_state = {}
def seq_sparkle(t):
    b = {}
    density = max(audio_state["amp"], 0.15) if AUDIO_ENABLED else 0.35
    bucket = int(t * 4)  # changes ~4x/sec
    for name in PANELS:
        key = f"{name}_{bucket}"
        if key not in _sparkle_state:
            _sparkle_state.clear()
            for n in PANELS:
                _sparkle_state[f"{n}_{bucket}"] = np.random.uniform(0, density)
        b[name] = FLOOR + _sparkle_state.get(key, 0)
    return b, False


def seq_all_pulse(t):
    amp = audio_state["amp"]
    if amp < 0.05:
        amp = 0.5 + 0.5 * np.sin(t * 2)  # idle breathing if no audio
    return {name: FLOOR + (1 - FLOOR) * amp for name in PANELS}, False


def seq_multiband(t):
    low, mid, high = audio_state["low"], audio_state["mid"], audio_state["high"]
    if low + mid + high < 0.05:
        low = 0.5 + 0.5 * np.sin(t * 1.3)
        mid = 0.5 + 0.5 * np.sin(t * 2.1 + 1)
        high = 0.5 + 0.5 * np.sin(t * 3.4 + 2)
    b = {}
    for p in PAIRS["front"]:
        b[p] = FLOOR + (1 - FLOOR) * low
    for p in PAIRS["mid"]:
        b[p] = FLOOR + (1 - FLOOR) * mid
    for p in PAIRS["back"]:
        b[p] = FLOOR + (1 - FLOOR) * high
    return b, False


def seq_blackout(t):
    return {}, True


SCHEDULE = [
    ("1. Baseline / Idle",              4.0, seq_baseline),
    ("2. Depth-Pair Sweep",             4.5, seq_depth_sweep),
    ("3. Left <-> Right Alternate",     4.0, seq_lr_alternate),
    ("4. Chase",                        3.0, seq_chase),
    ("5. Sparkle Flicker",              4.0, seq_sparkle),
    ("6. All-Together Pulse (audio)",   6.0, seq_all_pulse),
    ("7. Multiband Pairs (audio)",      6.0, seq_multiband),
    ("8. Dialogue Blackout",            2.5, seq_blackout),
]
TOTAL_CYCLE = sum(d for _, d, _ in SCHEDULE)


def current_sequence(t):
    seg = t % TOTAL_CYCLE
    acc = 0.0
    for label, duration, fn in SCHEDULE:
        if seg < acc + duration:
            return label, seg - acc, fn
        acc += duration
    return SCHEDULE[-1][0], 0.0, SCHEDULE[-1][2]


# ---------------------------------------------------------------
# Main
# ---------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--photo", default="stage_photo.png")
    parser.add_argument("--file", help="song audio file for the reactive sequences")
    parser.add_argument("--mic", action="store_true", help="use microphone instead of --file")
    args = parser.parse_args()

    base_arr, panels = load_base(args.photo)

    global AUDIO_ENABLED
    AUDIO_ENABLED = bool(args.file or args.mic)
    if AUDIO_ENABLED:
        start_audio(args)

    fig, ax = plt.subplots(figsize=(11, 6.3))
    fig.patch.set_facecolor("black")
    ax.axis("off")
    im = ax.imshow(base_arr)
    title = ax.set_title("", color="white", fontsize=14, pad=12)

    start_time = time.monotonic()

    def update(frame_num):
        t = time.monotonic() - start_time
        label, local_t, fn = current_sequence(t)
        brightness, blackout = fn(local_t)
        frame = compute_frame(base_arr, panels, brightness, blackout=blackout)
        im.set_data(frame)
        title.set_text(label)
        return im, title

    ani = animation.FuncAnimation(fig, update, interval=90, blit=False, cache_frame_data=False)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(0)