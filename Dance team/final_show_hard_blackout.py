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


# ---------------------------------------------------------------
# Authored timeline for THIS track (hard blackout on every dialogue point)
# ---------------------------------------------------------------
TIMELINE = [
    (0.00,   37.00,  "Dialogue - BLACKOUT",         seq_blackout),
    (37.00,  201.00, "Depth-Pair Sweep",            seq_depth_sweep),   # 0:37-3:21
    (201.00, 215.00, "Dialogue - BLACKOUT",         seq_blackout),      # 3:21-3:35
    (215.00, 313.00, "Multiband Pairs",              seq_multiband),     # 3:35-5:13
    (313.00, 327.00, "Dialogue - BLACKOUT",         seq_blackout),      # 5:13-5:27
    (327.00, 478.00, "All-Together Pulse",          seq_all_pulse),     # 5:27-7:58
    (478.00, 487.00, "Dialogue - BLACKOUT",         seq_blackout),      # 7:58-8:07
    (487.00, 606.00, "Multiband Pairs",             seq_multiband),     # 8:07-10:06
    (606.00, 617.00, "Dialogue - BLACKOUT",         seq_blackout),      # 10:06-10:17
    (617.00, 724.00, "Depth-Pair Sweep",            seq_depth_sweep),   # 10:17-12:04
    (724.00, 744.00, "Dialogue - BLACKOUT",         seq_blackout),      # 12:04-12:24
    (744.00, 977.00, "All-Together Pulse",          seq_all_pulse),     # 12:24-16:17
    (977.00, 984.00, "Dialogue - BLACKOUT",         seq_blackout),      # 16:17-16:24
    (984.00, 1094.00,"Depth-Pair Sweep",            seq_depth_sweep),   # 16:24-18:14
    (1094.00,1104.00,"Dialogue - BLACKOUT",         seq_blackout),      # 18:14-18:24
    (1104.00,1264.00,"Multiband Pairs",             seq_multiband),     # 18:24-21:04
    (1264.00,1287.00,"Dialogue - BLACKOUT",         seq_blackout),      # 21:04-21:27
]


def current_cue(t):
    for start, end, label, fn in TIMELINE:
        if start <= t < end:
            return label, t - start, fn
    return "End of show", 0.0, seq_blackout


# ---------------------------------------------------------------
# Main
# ---------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--photo", default="stage_photo.png")
    parser.add_argument("--file", required=True, help="the performance track (required for this authored show)")
    args = parser.parse_args()

    base_arr, panels = load_base(args.photo)
    start_audio(args.file)

    fig, ax = plt.subplots(figsize=(11, 6.3))
    fig.patch.set_facecolor("black")
    ax.axis("off")
    im = ax.imshow(base_arr)
    title = ax.set_title("", color="white", fontsize=14, pad=12)
    tclock = ax.text(0.99, 0.02, "", color="white", fontsize=10, ha="right",
                      transform=ax.transAxes, family="monospace")

    def update(frame_num):
        t = playback_time["t"]
        label, local_t, fn = current_cue(t)
        brightness, blackout = fn(local_t)
        frame = compute_frame(base_arr, panels, brightness, blackout=blackout)
        im.set_data(frame)
        title.set_text(label)
        tclock.set_text(f"{int(t)//60}:{int(t)%60:02d}")
        return im, title, tclock

    ani = animation.FuncAnimation(fig, update, interval=90, blit=False, cache_frame_data=False)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(0)