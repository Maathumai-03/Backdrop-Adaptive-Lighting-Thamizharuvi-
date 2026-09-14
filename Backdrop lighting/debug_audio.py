"""
Debug helper - run this BEFORE the main script.

1. Lists all audio input devices Windows sees, with their index numbers.
2. Streams raw (unscaled) magnitude numbers from the default device so we
   can tell if audio is reaching Python at all, before worrying about
   REFERENCE_LEVEL or band mapping.
"""

import numpy as np
import sounddevice as sd

print("=" * 60)
print("AVAILABLE AUDIO DEVICES")
print("=" * 60)
print(sd.query_devices())
print()
print("Default input device index:", sd.default.device[0])
print()
print("=" * 60)
print("RAW LEVEL TEST -- play music now, watch the numbers")
print("(Ctrl+C to stop)")
print("=" * 60)

SAMPLE_RATE = 44100
CHUNK = 1024


def callback(indata, frames, time_info, status):
    if status:
        print(status)
    chunk = indata[:, 0]
    # raw peak and average absolute amplitude -- no FFT, no scaling,
    # just "is there ANY signal coming in at all"
    peak = np.max(np.abs(chunk))
    rms = np.sqrt(np.mean(chunk**2))
    print(f"peak={peak:8.5f}   rms={rms:8.5f}")


with sd.InputStream(
    channels=1,
    samplerate=SAMPLE_RATE,
    blocksize=CHUNK,
    callback=callback,
):
    try:
        while True:
            sd.sleep(1000)
    except KeyboardInterrupt:
        print("\nStopped.")
