"""Export ONE-FRAME streaming model to ONNX: (mag [1,1,241], h [2,1,256]) -> (mask [1,1,241], h_out).
The host (Python stream, or a JUCE/VST plugin later) keeps the GRU state between calls.
    py -3.12 export_onnx.py [checkpoints/best.pth]
"""
import os, sys, torch, onnx, onnxruntime as ort, numpy as np
from model import StreamDenoiser, BINS

HERE = os.path.dirname(os.path.abspath(__file__))
CKPT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "checkpoints", "best.pth")
OUT = os.path.join(HERE, "call_denoiser_v4.onnx")


class Step(torch.nn.Module):
    def __init__(self, m): super().__init__(); self.m = m
    def forward(self, mag, h): return self.m(mag, h)


def main():
    m = StreamDenoiser(); ck = torch.load(CKPT, map_location="cpu", weights_only=False)   # best.pth holds EMA weights
    m.load_state_dict(ck["model"]); m.eval(); print(f"checkpoint step {ck.get('step')}, dev SI-SDR gain {ck.get('dev_gain', 0):+.2f} dB")
    mag, h = torch.rand(1, 1, BINS), torch.zeros(m.layers, 1, m.hidden)
    torch.onnx.export(Step(m), (mag, h), OUT, input_names=["mag", "h"], output_names=["mask", "h_out"], opset_version=17,
                      do_constant_folding=True)
    onnx.checker.check_model(onnx.load(OUT))
    s = ort.InferenceSession(OUT, providers=["CPUExecutionProvider"])
    a = s.run(None, {"mag": mag.numpy(), "h": h.numpy()})[0]
    with torch.no_grad(): b = m(mag, h)[0].numpy()
    print(f"saved {OUT}  ({os.path.getsize(OUT) / 1e6:.1f} MB)  max |onnx - torch| = {np.abs(a - b).max():.2e}")


if __name__ == "__main__":
    main()
