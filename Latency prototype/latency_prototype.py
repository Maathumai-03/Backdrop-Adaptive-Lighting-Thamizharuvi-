#!/usr/bin/env python3
"""
Latency prototype — measures real capture / network / render latency
on your actual WiFi, using your phone's browser as a stand-in for the
ESP32 (no hardware needed yet).

SETUP:
    pip install websockets sounddevice numpy

RUN (do this FIRST — simplest, isolates network+render only):
    python latency_prototype.py --mode ping

RUN (once ping numbers look sane — adds mic capture+processing):
    python latency_prototype.py --mode audio

RUN (optional — tune the trigger threshold for your sound source first):
    python latency_prototype.py --mode calibrate
    # shows a live dB meter; note the peak when you trigger your sound
    # source (metronome click, clap, etc.), then pass that as e.g.:
    python latency_prototype.py --mode audio --threshold -35

Then, on your phone (same WiFi as this laptop), open the URL it prints.
Keep latency_receiver.html in the same folder as this script.

Press Ctrl+C any time to print summary stats (mean/median/p95/max/jitter).
"""

import argparse
import asyncio
import http.server
import json
import socket
import socketserver
import sys
import threading
import time

import numpy as np
import websockets

HTTP_PORT = 8000
WS_PORT = 8765

connected_clients = set()
pending = {}     # msg_id -> (t_sent, capture_ms)
results = []     # list of dicts, one per round trip


def get_local_ip():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))
        return s.getsockname()[0]
    except Exception:
        return "127.0.0.1"
    finally:
        s.close()


async def ws_handler(websocket):
    connected_clients.add(websocket)
    print(f"[phone connected] ({len(connected_clients)} client(s) total)")
    try:
        async for raw in websocket:
            data = json.loads(raw)
            if data.get("type") != "ack":
                continue
            msg_id = data.get("id")
            entry = pending.pop(msg_id, None)
            if entry is None:
                continue
            t_sent, capture_ms = entry
            rtt_ms = (time.monotonic() - t_sent) * 1000
            network_one_way_ms = rtt_ms / 2
            render_delay_ms = float(data.get("render_delay_ms", 0))
            total_ms = capture_ms + network_one_way_ms + render_delay_ms
            results.append(dict(
                rtt_ms=rtt_ms,
                network_one_way_ms=network_one_way_ms,
                render_delay_ms=render_delay_ms,
                capture_ms=capture_ms,
                total_ms=total_ms,
            ))
            print(
                f"  #{len(results):<3} rtt {rtt_ms:6.1f}ms | "
                f"network(1-way) {network_one_way_ms:6.1f}ms | "
                f"render {render_delay_ms:5.1f}ms | "
                f"capture {capture_ms:5.1f}ms | "
                f"TOTAL {total_ms:6.1f}ms"
            )
    finally:
        connected_clients.discard(websocket)
        print(f"[phone disconnected] ({len(connected_clients)} client(s) total)")


async def send_flash(capture_ms=0.0):
    if not connected_clients:
        return
    msg_id = str(time.monotonic_ns())
    pending[msg_id] = (time.monotonic(), capture_ms)
    msg = json.dumps({"type": "flash", "id": msg_id})
    dead = []
    for ws in list(connected_clients):
        try:
            await ws.send(msg)
        except Exception:
            dead.append(ws)
    for ws in dead:
        connected_clients.discard(ws)


class ReusableTCPServer(socketserver.TCPServer):
    allow_reuse_address = True


def start_http_server():
    handler = http.server.SimpleHTTPRequestHandler
    with ReusableTCPServer(("0.0.0.0", HTTP_PORT), handler) as httpd:
        httpd.serve_forever()


def print_summary():
    if not results:
        print("\nNo round trips recorded — did the phone connect and stay connected?")
        return
    print(f"\n{'='*68}\nSUMMARY ({len(results)} trials)\n{'='*68}")
    rows = [
        ("total_ms", "TOTAL (capture + network + render)"),
        ("network_one_way_ms", "Network, one-way (est. RTT/2)"),
        ("render_delay_ms", "Phone browser render delay"),
        ("capture_ms", "Audio capture + processing"),
    ]
    for key, label in rows:
        vals = np.array([r[key] for r in results])
        if key == "capture_ms" and vals.max() == 0:
            continue  # not measured in ping mode
        print(
            f"{label:38s} mean {vals.mean():6.1f}ms  median {np.median(vals):6.1f}ms  "
            f"p95 {np.percentile(vals, 95):6.1f}ms  max {vals.max():6.1f}ms  "
            f"jitter(std) {vals.std():5.1f}ms"
        )
    p95_total = np.percentile([r["total_ms"] for r in results], 95)
    print(f"\nBudget check — target is 35ms one-way:")
    if p95_total <= 35:
        print(f"  p95 total is {p95_total:.1f}ms — inside budget.")
    else:
        print(f"  p95 total is {p95_total:.1f}ms — OVER budget by {p95_total - 35:.1f}ms.")
    print("\n(Remember: this stands in your phone's WiFi+browser for the ESP32's\n"
          "WiFi+render. Real ESP32 numbers will differ, but this tells you\n"
          "whether the network leg is even in a workable range.)")


async def ping_loop(interval):
    print(f"\nPing mode: sending a message every {interval}s once your phone connects.")
    print("Watch the phone screen flash white — that round trip is what's measured.\n")
    while True:
        await send_flash(capture_ms=0.0)
        await asyncio.sleep(interval)


def audio_loop(loop, threshold_db, debounce_s=0.4):
    import sounddevice as sd

    print(f"\nAudio mode: clap near your laptop mic (threshold {threshold_db} dB).")
    print(f"Debounce: {debounce_s}s between triggers so one clap = one flash.\n")

    last_trigger = [0.0]
    RATE = 44100
    CHUNK = 512

    def callback(indata, frames, time_info, status):
        t_capture = time.monotonic()
        rms = np.sqrt(np.mean(indata ** 2)) + 1e-12
        db = 20 * np.log10(rms)
        if db > threshold_db and (t_capture - last_trigger[0]) > debounce_s:
            last_trigger[0] = t_capture
            capture_ms = (time.monotonic() - t_capture) * 1000  # processing overhead
            asyncio.run_coroutine_threadsafe(send_flash(capture_ms=capture_ms), loop)

    with sd.InputStream(channels=1, samplerate=RATE, blocksize=CHUNK, callback=callback):
        while True:
            time.sleep(0.1)


def calibrate_loop():
    import sounddevice as sd

    print("\nCalibration mode: shows live sound level. No network/flash involved.")
    print("Let it run quietly for a few seconds first to see your room's noise floor,")
    print("then trigger your sound source (metronome click, clap, etc.) a few times")
    print("and note the peak dB it reaches. Set --threshold a few dB below that peak")
    print("(and comfortably above the noise floor). Ctrl+C to stop.\n")

    RATE = 44100
    CHUNK = 512
    peak_hold = [-100.0]
    last_print = [0.0]

    def callback(indata, frames, time_info, status):
        rms = np.sqrt(np.mean(indata ** 2)) + 1e-12
        db = 20 * np.log10(rms)
        peak_hold[0] = max(peak_hold[0] * 0.98, db)  # slow-decay peak hold
        now = time.monotonic()
        if now - last_print[0] > 0.1:
            last_print[0] = now
            bar_len = max(0, min(50, int((db + 60) )))
            bar = "#" * bar_len
            print(f"\r  live: {db:6.1f}dB  peak: {peak_hold[0]:6.1f}dB  {bar:<50}", end="", flush=True)

    with sd.InputStream(channels=1, samplerate=RATE, blocksize=CHUNK, callback=callback):
        while True:
            time.sleep(0.05)


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["ping", "audio", "calibrate"], default="ping")
    parser.add_argument("--interval", type=float, default=1.0, help="ping mode interval (s)")
    parser.add_argument("--threshold", type=float, default=-25, help="audio mode trigger threshold (dB)")
    args = parser.parse_args()

    if args.mode == "calibrate":
        await asyncio.to_thread(calibrate_loop)
        return

    ip = get_local_ip()
    threading.Thread(target=start_http_server, daemon=True).start()

    print("=" * 60)
    print("LATENCY PROTOTYPE")
    print("=" * 60)
    print("1. On your phone (SAME WiFi as this laptop), open:\n")
    print(f"     http://{ip}:{HTTP_PORT}/latency_receiver.html\n")
    print("2. Wait for '[phone connected]' to print below.")
    if args.mode == "ping":
        print("3. It will start pinging automatically — just watch it run.")
    else:
        print("3. Clap near your laptop's mic, a few dozen times, varied pacing.")
    print("4. Press Ctrl+C any time to print summary stats.\n")

    await websockets.serve(ws_handler, "0.0.0.0", WS_PORT)

    if args.mode == "ping":
        await ping_loop(args.interval)
    else:
        loop = asyncio.get_running_loop()
        await asyncio.to_thread(audio_loop, loop, args.threshold)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print_summary()
        sys.exit(0)
