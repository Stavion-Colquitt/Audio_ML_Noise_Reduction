"""Live call denoiser (v4): your mic -> 10 ms streaming GRU -> a virtual audio cable. In Teams, Zoom or any app, pick
the cable's output as your microphone. CPU only (numpy + onnxruntime + sounddevice, no PyTorch).

    python v4/live.py --list                                  # show audio devices
    python v4/live.py --mic "Microphone" --out "CABLE Input"  # names (any part) or index numbers
    python v4/live.py --mic 1 --out 7 --floor 0.05            # keep a little noise (sounds more natural)

Needs call_denoiser_v4.onnx next to this file (Hugging Face: Stavion-Colquitt/Call_Denoiser_V4, or Release V4.0).
Audio runs at 16 kHz mono in 5 ms model blocks; the sound card callback is --block ms (a multiple of 5).
"""
import argparse, sys, time
import numpy as np, sounddevice as sd
from stream import StreamEngine, SR, M


def pick(spec, kind):
    """Device index from an index or a case-insensitive part of the name; kind = 'input' or 'output'."""
    if spec is None: return None
    if str(spec).isdigit(): return int(spec)
    key = "max_input_channels" if kind == "input" else "max_output_channels"
    hits = [i for i, d in enumerate(sd.query_devices()) if d[key] > 0 and str(spec).lower() in d["name"].lower()]
    if not hits: sys.exit(f"no {kind} device matching '{spec}' (run with --list)")
    return hits[0]


def make_callback(engine, stats):
    def callback(indata, outdata, frames, t, status):
        if status: stats["status"] += 1
        x = indata[:, 0]
        t0 = time.perf_counter()
        y = np.concatenate([engine.process(x[i:i + M]) for i in range(0, frames, M)])
        stats["cpu"] = max(stats["cpu"], (time.perf_counter() - t0) * 1000 / (frames / SR * 1000))
        outdata[:] = np.clip(y, -1.0, 1.0)[:, None]
    return callback


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--list", action="store_true", help="list audio devices and exit")
    ap.add_argument("--mic", help="input device: index or part of its name")
    ap.add_argument("--out", help="output device (the virtual cable's input): index or part of its name")
    ap.add_argument("--block", type=int, default=10, help="sound card block in ms, a multiple of 5 (default 10)")
    ap.add_argument("--floor", type=float, default=0.0, help="minimum mask, 0-1: keeps some noise for a natural sound")
    a = ap.parse_args()
    if a.list: print(sd.query_devices()); return
    if a.block % 5: sys.exit("--block must be a multiple of 5 ms")
    mic, out = pick(a.mic, "input"), pick(a.out, "output")
    engine, stats = StreamEngine(mask_floor=a.floor), {"status": 0, "cpu": 0.0}
    for _ in range(40): engine.process(np.zeros(M, np.float32))   # warm up ONNX Runtime so the first real block is on time
    engine.reset()
    names = sd.query_devices(mic, "input")["name"], sd.query_devices(out, "output")["name"]
    print(f"Call Denoiser v4 live: {names[0]} -> {names[1]}  ({SR} Hz, {a.block} ms blocks, 10 ms model latency)")
    print("In your call app, choose the virtual cable's output as the microphone. Ctrl+C to stop.")
    with sd.Stream(device=(mic, out), samplerate=SR, blocksize=SR * a.block // 1000, channels=1, dtype="float32",
                   latency="low", callback=make_callback(engine, stats)):
        try:
            while True:
                time.sleep(5)
                print(f"  running: worst CPU load per block {stats['cpu']:.0%}, audio glitches {stats['status']}", flush=True)
                stats["cpu"] = 0.0
        except KeyboardInterrupt:
            print("stopped")


if __name__ == "__main__":
    main()