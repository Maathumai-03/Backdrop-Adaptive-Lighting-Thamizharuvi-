# Backdrop-Adaptive-Lighting-Thamizharuvi-

Audio-reactive + preprogrammed LED backdrop lighting control for the
Thamizharuvi stage (6 edge-lit silhouette panels + center calligraphy
piece), driven by ESP32 + WS2812B, with a Python preview/simulation
layer for planning cues before real hardware is built.

## Status

Software-side cue design and preview tooling are working and tested
against real stage photos and the actual performance track. Physical
build (panels, ESP32, power/data wiring) has not started yet — see
`codebase_explainer.md` for the feasibility/build-considerations notes.

## What's in here

| File | Purpose |
|---|---|
| `sequence_showcase.py` | Demo reel of all 8 lighting sequence types, runs on any photo |
| `final_show_hard_blackout.py` | Full authored show for the real track — all dialogue = hard blackout |
| `final_show_soft_breaks.py` | Same show — short dialogue points use a dim "ember" glow instead |
| `stage_photo.png` | The reference stage photo (panel coordinates are calibrated to this exact file) |
| `codebase_explainer.md` | Plain-language walkthrough of how the code works, for debugging |
| `latency_prototype.py` + `latency_receiver.html` | Standalone tool for measuring audio-to-device latency over WiFi |
| `led_fft_sequences.py` | Early standalone prototype — FFT band-splitting into 5 sequences (superseded by the panel-specific work, kept for reference) |

## Setup

```bash
pip install numpy pillow matplotlib sounddevice soundfile scipy
```

## Running the real show preview

```bash
python final_show_soft_breaks.py --file "your_track.mp3"
```
(`stage_photo.png` must be in the same folder.)

## Known open items / next steps

- Physical panel build not started (see feasibility notes in `codebase_explainer.md`)
- Real show needs to *listen* to a live audio feed and sync-check against
  a rendered reference, rather than owning playback itself like the demo
  scripts do — the Sync Monitor / Offline Renderer / fallback design is
  planned but not yet built
- Latency tested only against a phone browser stand-in, not real ESP32
  hardware yet
- Panel coordinates in all scripts are hardcoded to `stage_photo.png`
  specifically — re-calibration needed if the reference photo changes
