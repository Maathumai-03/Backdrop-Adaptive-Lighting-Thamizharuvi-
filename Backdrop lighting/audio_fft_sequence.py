"""
Step 1 of the LED prototype: audio -> FFT -> band energy sequence.

This runs on your LAPTOP (not the ESP32). It listens to your mic (or
whatever input device you pick), continuously computes the FFT, and
prints the (bass, mid, treble) byte values it *would* send over the
wire to the ESP32. No networking yet -- this is just to prove the
signal chain works and look right before we wire up UDP/serial.

Install first (on your own machine, not this sandbox):
    pip install sounddevice numpy
"""

import numpy as np
import sounddevice as sd

SAMPLE_RATE = 44100      # samples per second the mic gives us
CHUNK = 1024              # samples per FFT window (~23ms of audio @ 44.1kHz)
REFERENCE_LEVEL = 50.0    # tune this after you see real numbers from your mic

# Frequency ranges in Hz. Adjust these to taste once you see it react.
BASS_RANGE = (20, 250)
MID_RANGE = (250, 2000)
TREBLE_RANGE = (2000, 8000)


def get_band_energy(chunk, sample_rate=SAMPLE_RATE):
    """
    Take one raw audio chunk and return (bass, mid, treble) as 0-255
    integers -- this is the exact payload shape we'll eventually send
    to the ESP32 as 3 raw bytes.
    """
    # window the chunk to reduce spectral leakage (edge artifacts in the FFT)
    windowed = chunk * np.hanning(len(chunk))

    # FFT: converts the waveform into frequency-domain magnitudes
    fft_result = np.fft.rfft(windowed)
    magnitude = np.abs(fft_result)
    freqs = np.fft.rfftfreq(len(chunk), d=1.0 / sample_rate)

    def band_energy(lo, hi):
        mask = (freqs >= lo) & (freqs < hi)
        if not np.any(mask):
            return 0.0
        return np.mean(magnitude[mask])

    def to_byte(value, ref=REFERENCE_LEVEL):
        return int(np.clip((value / ref) * 255, 0, 255))

    bass = to_byte(band_energy(*BASS_RANGE))
    mid = to_byte(band_energy(*MID_RANGE))
    treble = to_byte(band_energy(*TREBLE_RANGE))
    return bass, mid, treble


def audio_callback(indata, frames, time_info, status):
    if status:
        print(status)
    chunk = indata[:, 0]  # take one channel (mono)
    bass, mid, treble = get_band_energy(chunk)

    # This is the exact sequence/packet you'll hand to the ESP32 later.
    packet = bytes([bass, mid, treble])
    print(f"bass={bass:3d}  mid={mid:3d}  treble={treble:3d}   raw_bytes={packet.hex()}")


def main():
    print("Listening... play some music. Ctrl+C to stop.")
    print("Watch the numbers -- bass should spike on kicks/bass, treble on hi-hats/cymbals.\n")
    with sd.InputStream(
        channels=1,
        samplerate=SAMPLE_RATE,
        blocksize=CHUNK,
        callback=audio_callback,
    ):
        try:
            while True:
                sd.sleep(1000)
        except KeyboardInterrupt:
            print("\nStopped.")


if __name__ == "__main__":
    main()
