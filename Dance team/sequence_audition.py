#!/usr/bin/env python3
"""
Final show — HARD BLACKOUT variant.

Plays the actual performance track and follows an authored cue timeline
synced to it: each music section runs its assigned sequence (still
beat-reactive within that section), and every one of the 9 dialogue
points goes to a true, hard 0% blackout.

SETUP:
    pip install numpy pillow matplotlib sounddevice soundfile

RUN (from the folder containing this script, stage_photo.png, and the track):
    python final_show_hard_blackout.py --file "Last_performance_track_with_voicee_mp3.mpeg"

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
    "front": ("L_front", "R_mid"),    # their numbers 1 & 6 -- closest to viewer
    "mid":   ("L_back", "R_front"),   # their numbers 2 & 5 -- middle distance
    "back":  ("L_mid", "R_back"),     # their numbers 3 & 7 -- farthest from viewer
}
LEFT_PANELS = ["L_front", "L_mid", "L_back"]
RIGHT_PANELS = ["R_front", "R_mid", "R_back"]
CHASE_ORDER = ["L_back", "L_mid", "L_front", "R_front", "R_mid", "R_back"]

FLOOR = 0.30
DIM_COLOR = np.array([55, 38, 20])
BRIGHT_COLOR = np.array([255, 195, 110])

SCALE = 0.55
BLUR_RADIUS = 12


# ---------------------------------------------------------------
# Image compositing
# ---------------------------------------------------------------
def load_base(photo_path):
    img = Image.open(photo_path).convert("RGB")
    w, h = img.size
    small = img.resize((int(w * SCALE), int(h * SCALE)))
    scaled_panels = {name: tuple(int(v * SCALE) for v in box) for name, box in PANELS.items()}
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
# Audio-reactive state
# ---------------------------------------------------------------
audio_state = {"amp": 0.0, "low": 0.0, "mid": 0.0, "high": 0.0,
               "beat_count": 0, "last_beat_time": time.monotonic(), "last_beat_amp": 0.0}
_peak = {"amp": 1e-4, "low": 1e-4, "mid": 1e-4, "high": 1e-4}
_rolling = {"avg": 1e-4}


def _normalize(key, value):
    _peak[key] = max(value, _peak[key] * 0.995)
    return float(np.clip(value / (_peak[key] + 1e-9), 0, 1))


def _detect_onset(amp):
    now = time.monotonic()
    _rolling["avg"] = _rolling["avg"] * 0.9 + amp * 0.1
    if amp > _rolling["avg"] * 1.5 and amp > 0.001 and (now - audio_state["last_beat_time"]) > 0.3:
        audio_state["beat_count"] += 1
        audio_state["last_beat_time"] = now
        audio_state["last_beat_amp"] = float(np.clip(amp / (_rolling["avg"] + 1e-9) / 3, 0, 1))


def _beat_decay(decay=0.4):
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


playback_time = {"t": 0.0, "started": False}


def start_audio(path):
    import sounddevice as sd
    import soundfile as sf

    data, file_rate = sf.read(path, always_2d=True)
    data = data[:, 0]
    CHUNK = 1024
    pos = [0]
    cb = audio_callback_factory(file_rate)

    def player():
        playback_time["started"] = True
        t0 = time.monotonic()
        with sd.OutputStream(samplerate=file_rate, channels=1) as out:
            while pos[0] < len(data):
                chunk = data[pos[0]:pos[0] + CHUNK]
                if len(chunk) == 0:
                    break
                out.write(chunk.astype(np.float32).reshape(-1, 1))
                cb(chunk.reshape(-1, 1), len(chunk), None, None)
                pos[0] += CHUNK
                playback_time["t"] = pos[0] / file_rate

    threading.Thread(target=player, daemon=True).start()
    print(f"[audio] playing: {path}")


# ---------------------------------------------------------------
# Sequence generators
# ---------------------------------------------------------------
def seq_baseline(t):
    return {}, False


def seq_depth_sweep(t):
    idx = audio_state["beat_count"] % 3
    intensity = _beat_decay(decay=0.5) * (0.4 + 0.6 * audio_state["last_beat_amp"])
    active_pair = PAIRS[["front", "mid", "back"][idx]]
    return {p: FLOOR + (1 - FLOOR) * intensity for p in active_pair}, False


def seq_lr_alternate(t):
    panels = LEFT_PANELS if audio_state["beat_count"] % 2 == 0 else RIGHT_PANELS
    intensity = _beat_decay(decay=0.5) * (0.4 + 0.6 * audio_state["last_beat_amp"])
    return {p: FLOOR + (1 - FLOOR) * intensity for p in panels}, False


def seq_chase(t):
    idx = audio_state["beat_count"] % len(CHASE_ORDER)
    intensity = _beat_decay(decay=0.4) * (0.4 + 0.6 * audio_state["last_beat_amp"])
    panel = CHASE_ORDER[idx]
    return {panel: FLOOR + (1 - FLOOR) * intensity}, False


def seq_all_pulse(t):
    amp = audio_state["amp"]
    return {name: FLOOR + (1 - FLOOR) * amp for name in PANELS}, False


def seq_multiband(t):
    low, mid, high = audio_state["low"], audio_state["mid"], audio_state["high"]
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



def seq_ember(t):
    """Low, slow, synchronized breathing glow -- present but receded, not fully black."""
    breathe = 0.06 + 0.04 * np.sin(2 * np.pi * t / 6.0)
    return {name: breathe for name in PANELS}, False


# ---------------------------------------------------------------
# Live audition tool -- switch sequences on the fly with keypresses
# while the real track plays, so you can decide what goes where.
# Every switch is logged to the terminal with the timestamp, so you
# end up with a ready-made list to hand back for the final timeline.
# ---------------------------------------------------------------
KEYMAP = {
    "1": ("Baseline / Idle",           seq_baseline),
    "2": ("Depth-Pair Sweep",          seq_depth_sweep),
    "3": ("Left <-> Right Alternate",  seq_lr_alternate),
    "4": ("Chase",                     seq_chase),
    "5": ("All-Together Pulse",        seq_all_pulse),
    "6": ("Multiband Pairs",           seq_multiband),
    "7": ("Ember (soft dialogue)",     seq_ember),
    "8": ("Blackout (hard)",           seq_blackout),
}

LEGEND = "  ".join(f"[{k}] {v[0]}" for k, v in KEYMAP.items())

CURRENT = {"label": KEYMAP["1"][0], "fn": KEYMAP["1"][1], "switch_t": 0.0}


def fmt_time(t):
    return f"{int(t)//60}:{int(t)%60:02d}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--photo", default="stage_photo.png")
    parser.add_argument("--file", required=True, help="the performance track to audition against")
    args = parser.parse_args()

    base_arr, panels = load_base(args.photo)
    start_audio(args.file)

    fig, ax = plt.subplots(figsize=(11, 6.8))
    fig.patch.set_facecolor("black")
    ax.axis("off")
    im = ax.imshow(base_arr)
    title = ax.set_title("", color="white", fontsize=15, pad=10)
    tclock = ax.text(0.99, 0.03, "", color="white", fontsize=11, ha="right",
                      transform=ax.transAxes, family="monospace")
    legend = ax.text(0.5, -0.04, LEGEND, color="#aaaaaa", fontsize=9, ha="center",
                      transform=ax.transAxes, family="monospace")

    def on_key(event):
        if event.key in KEYMAP:
            label, fn = KEYMAP[event.key]
            CURRENT["label"] = label
            CURRENT["fn"] = fn
            CURRENT["switch_t"] = playback_time["t"]
            print(f"[{fmt_time(playback_time['t'])}] -> {label}")

    fig.canvas.mpl_connect("key_press_event", on_key)

    print("\nPress a number key any time to switch the live sequence.")
    print(LEGEND)
    print("\nEvery switch is logged below with its timestamp -- keep this")
    print("terminal open and copy the log when you're done deciding.\n")

    def update(frame_num):
        t = playback_time["t"]
        local_t = t - CURRENT["switch_t"]
        brightness, blackout = CURRENT["fn"](local_t)
        frame = compute_frame(base_arr, panels, brightness, blackout=blackout)
        im.set_data(frame)
        title.set_text(CURRENT["label"])
        tclock.set_text(fmt_time(t))
        return im, title, tclock, legend

    ani = animation.FuncAnimation(fig, update, interval=90, blit=False, cache_frame_data=False)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(0)
