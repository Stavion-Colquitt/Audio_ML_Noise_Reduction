"""
U-Net Denoiser Trainer  (v3: complex ratio mask)
================================================
v3 changes vs v2: the U-Net sees [log1p|S|, |S|^0.3 * cos(phase), |S|^0.3 * sin(phase)] and outputs a COMPLEX
mask (tanh-bounded real and imaginary parts), so it can correct phase instead of reusing the noisy phase.
Everything else (data, loss, 200 ms segments, model selection) is unchanged, so v2 vs v3 is a fair comparison.

v2 notes:
v1 computed SI-SNR on log1p SPECTROGRAMS. SI-SNR ignores scale, and a scale on a log spectrogram is a power
law on the audio -> the model could shrink the whole spectrum almost for free, which showed up as
over-suppression. v2 scores the RECONSTRUCTED WAVEFORM:

    mask = U-Net(log1p|STFT(noisy)|)        (mask in [0, 1])
    est  = iSTFT(mask * STFT(noisy))         (linear-magnitude mask, noisy phase)
    loss = -SI-SDR(est, clean) + W_MR * multi-resolution STFT loss(est, clean)

Checkpoints are chosen by DEV loss and DEV SI-SDR improvement (held-out files), never by the test set.
Runs are tracked in MLflow (`mlflow ui` in this folder to browse).

    py -3.12 train.py                       # full run
    py -3.12 train.py --smoke               # 1 epoch on ~500 files: checks the whole loop end to end
"""
import os, sys, time, random, argparse
import torch, torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from model import UNetDenoiser, count_parameters
from dataset import get_dataloaders, N_FFT, HOP_LENGTH, SNR_MIN_DB, SNR_MAX_DB, P_CLEAN_ONLY, SEG

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.environ.get("DNS_DATA", r"C:\Users\stavi\Projects\audio-ml-pipeline\data")   # reuse v1's download
CLEAN_DIR, NOISE_DIR = os.path.join(DATA_DIR, "clean_fullband"), os.path.join(DATA_DIR, "noise_fullband")
CKPT_DIR = os.path.join(HERE, "checkpoints"); os.makedirs(CKPT_DIR, exist_ok=True)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ap = argparse.ArgumentParser()
ap.add_argument("--epochs", type=int, default=60)
ap.add_argument("--batch", type=int, default=64)
ap.add_argument("--lr", type=float, default=1e-3)
ap.add_argument("--w_mr", type=float, default=5.0)        # weight of the multi-resolution STFT loss
ap.add_argument("--max_files", type=int, default=None)
ap.add_argument("--smoke", action="store_true")
args = ap.parse_args()
if args.smoke: args.epochs, args.max_files = 1, 500


def _win(n, x): return torch.hann_window(n, device=x.device)

COMP = 0.3   # power-law compression for the complex input channels (|S|^0.3, phase kept)

def features(S):                                             # complex [B, F, T] -> [B, 3, F, T]
    mag = S.abs(); unit = S / (mag + 1e-8); c = (mag + 1e-8) ** COMP
    return torch.stack([torch.log1p(mag), c * unit.real, c * unit.imag], dim=1)

def enhance(model, noisy):                                   # noisy [B, T] -> enhanced waveform [B, T]
    S = torch.stft(noisy, N_FFT, HOP_LENGTH, window=_win(N_FFT, noisy), center=True, return_complex=True)
    m = model(features(S))                                   # [B, 2, F, T] complex ratio mask
    return torch.istft(torch.complex(m[:, 0], m[:, 1]) * S, N_FFT, HOP_LENGTH, window=_win(N_FFT, noisy),
                       center=True, length=noisy.shape[-1])

def si_sdr(est, ref, eps=1e-8):                              # per item, dB
    est, ref = est - est.mean(-1, keepdim=True), ref - ref.mean(-1, keepdim=True)
    t = (est * ref).sum(-1, keepdim=True) / ((ref * ref).sum(-1, keepdim=True) + eps) * ref
    return 10 * torch.log10((t * t).sum(-1) / (((est - t) ** 2).sum(-1) + eps) + eps)

def mrstft_loss(est, ref):
    total = 0
    for n, h in ((256, 64), (512, 128), (1024, 256)):
        E = torch.stft(est, n, h, window=_win(n, est), return_complex=True).abs()
        R = torch.stft(ref, n, h, window=_win(n, ref), return_complex=True).abs()
        total = total + (E - R).norm() / (R.norm() + 1e-8) + (torch.log1p(E) - torch.log1p(R)).abs().mean()
    return total / 3

SDR_CAP = 40.0   # clean-only examples can reach ~100 dB; capping keeps them from dominating the loss/metric

def loss_fn(est, clean):
    return -si_sdr(est, clean).clamp(max=SDR_CAP).mean() + args.w_mr * mrstft_loss(est, clean)


def run_epoch(model, loader, opt=None):
    train = opt is not None; model.train(train)
    tot, gain, n = 0.0, 0.0, 0
    if not train: random.seed(1234); torch.manual_seed(1234)  # same dev crops/mixes every epoch
    with torch.set_grad_enabled(train):
        for i, (noisy, clean) in enumerate(loader):
            noisy, clean = noisy.to(DEVICE, non_blocking=True), clean.to(DEVICE, non_blocking=True)
            est = enhance(model, noisy); loss = loss_fn(est, clean)
            if train:
                opt.zero_grad(set_to_none=True); loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
                if i % 100 == 0: print(f"  batch {i}/{len(loader)}  loss {loss.item():.3f}", flush=True)
            else:
                gain += (si_sdr(est, clean).clamp(max=SDR_CAP) - si_sdr(noisy, clean).clamp(max=SDR_CAP)).sum().item()
            tot += loss.item() * len(noisy); n += len(noisy)
    return tot / n, (gain / n if not train else None)


def main():
    import mlflow
    mlflow.set_tracking_uri("sqlite:///" + os.path.join(HERE, "mlflow.db").replace("\\", "/"))   # `mlflow ui --backend-store-uri sqlite:///mlflow.db`
    mlflow.set_experiment("denoiser-v3")
    print(f"device: {DEVICE}" + (f" ({torch.cuda.get_device_name(0)})" if DEVICE.type == "cuda" else ""))
    tr, dv = get_dataloaders(CLEAN_DIR, NOISE_DIR, batch_size=args.batch, max_files=args.max_files)
    model = UNetDenoiser(base_ch=32).to(DEVICE)
    print(f"parameters: {count_parameters(model):,}")
    opt = optim.Adam(model.parameters(), lr=args.lr)
    sched = CosineAnnealingLR(opt, T_max=args.epochs, eta_min=1e-5)
    best = float("inf")
    with mlflow.start_run(run_name="smoke" if args.smoke else None):
        mlflow.log_params({"epochs": args.epochs, "batch": args.batch, "lr": args.lr, "w_mr": args.w_mr,
                           "snr_db": f"{SNR_MIN_DB}..{SNR_MAX_DB}", "clean_only": P_CLEAN_ONLY,
                           "segment_ms": SEG / 16, "max_files": args.max_files,
                           "mask": "complex (tanh re/im)", "input": f"log1p|S| + |S|^{COMP} re/im"})
        _, base_gain = 0, None
        for ep in range(1, args.epochs + 1):
            t0 = time.time()
            trl, _ = run_epoch(model, tr, opt)
            dvl, dvg = run_epoch(model, dv)
            sched.step()
            print(f"epoch {ep}: train {trl:.3f} | dev {dvl:.3f} | dev SI-SDR gain {dvg:+.2f} dB | "
                  f"{time.time() - t0:.0f}s", flush=True)
            mlflow.log_metrics({"train_loss": trl, "dev_loss": dvl, "dev_sisdr_gain_db": dvg}, step=ep)
            if dvl < best:
                best = dvl
                torch.save({"epoch": ep, "model_state_dict": model.state_dict(), "dev_loss": dvl,
                            "dev_sisdr_gain_db": dvg, "args": vars(args)}, os.path.join(CKPT_DIR, "best_model.pth"))
                print(f"  saved best (dev {dvl:.3f})")
        mlflow.log_metric("best_dev_loss", best)
    print("done. next: py -3.12 export_model.py")


if __name__ == "__main__":
    main()
