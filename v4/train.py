"""Train call denoiser v4 (16 kHz streaming GRU; same recipe as the DAW denoiser).

The model always sees the noisy input at its REAL level (exactly as in a DAW). Only the loss is level-normalised
(output and target are scaled by the same factor), so loud and quiet vocals count the same.
loss = compressed spectral loss (Braun & Tashev: 0.7 * |Y|^0.3 vs |X|^0.3 + 0.3 * compressed complex)  [perceptual]
     + 0.5 * multi-resolution STFT loss, all on the output waveform.
Training: AdamW, 1k-step warmup then cosine decay, grad clipping, and an exponential moving average (EMA) of the
weights, which is what gets evaluated, saved and exported (usually a little better and more stable than raw weights).
Evaluates on a fixed dev set every --eval_every steps; keeps best.pth (EMA weights, by dev loss) and latest.pth
(full state, so --resume continues after a crash or reboot). Runs logged to MLflow.

    py -3.12 train.py --hours 9            # sizes total steps to fit ~9 h on this GPU (measured in the first steps)
    py -3.12 train.py --steps 300 --smoke  # quick end-to-end check
    py -3.12 train.py --resume             # continue from checkpoints/latest.pth
"""
import os, time, math, argparse, torch
import copy
from model import StreamDenoiser, enhance, count_parameters, COMP
from dataset import loaders, SR

HERE = os.path.dirname(os.path.abspath(__file__))
CK = os.path.join(HERE, "checkpoints"); os.makedirs(CK, exist_ok=True)
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ap = argparse.ArgumentParser()
ap.add_argument("--steps", type=int, default=None)
ap.add_argument("--hours", type=float, default=None)
ap.add_argument("--batch", type=int, default=32)
ap.add_argument("--lr", type=float, default=1e-3)
ap.add_argument("--eval_every", type=int, default=1000)
ap.add_argument("--workers", type=int, default=6)
ap.add_argument("--resume", action="store_true")
ap.add_argument("--smoke", action="store_true")
args = ap.parse_args()


def _norm(est, ref):
    s = ref.pow(2).mean(-1, keepdim=True).add(1e-8).sqrt(); return est / s, ref / s

def comp_loss(Y, X):            # complex STFTs [B, T, F]
    ay, ax = Y.abs().clamp_min(1e-8) ** COMP, X.abs().clamp_min(1e-8) ** COMP
    cy, cx = ay * Y / Y.abs().clamp_min(1e-8), ax * X / X.abs().clamp_min(1e-8)
    return 0.7 * (ay - ax).pow(2).mean() + 0.3 * (cy - cx).abs().pow(2).mean()

def mrstft(est, ref):
    t = 0
    for n, h in ((256, 64), (512, 128), (1024, 256)):
        w = torch.hann_window(n, device=est.device)
        E = torch.stft(est, n, h, window=w, return_complex=True).abs(); R = torch.stft(ref, n, h, window=w, return_complex=True).abs()
        t = t + (E - R).norm() / (R.norm() + 1e-8) + (torch.log(E + 1e-5) - torch.log(R + 1e-5)).abs().mean()
    return t / 3

LOSS_WIN = torch.hann_window(512)
EMA, WARMUP = 0.999, 1000

def losses(model, noisy, clean):
    est = enhance(model, noisy)                       # model sees the real input level
    s = clean.pow(2).mean(-1, keepdim=True).add(1e-8).sqrt()
    est_n, clean_n, noisy_n = est / s, clean / s, noisy / s
    w = LOSS_WIN.to(est.device)
    Y = torch.stft(est_n, 512, 128, window=w, return_complex=True).transpose(1, 2)
    X = torch.stft(clean_n, 512, 128, window=w, return_complex=True).transpose(1, 2)
    return comp_loss(Y, X) + 0.5 * mrstft(est_n, clean_n), est_n, noisy_n, clean_n

def si_sdr(e, r, eps=1e-8):
    e, r = e - e.mean(-1, keepdim=True), r - r.mean(-1, keepdim=True)
    t = (e * r).sum(-1, keepdim=True) / (r.pow(2).sum(-1, keepdim=True) + eps) * r
    return 10 * torch.log10(t.pow(2).sum(-1) / ((e - t).pow(2).sum(-1) + eps) + eps)

@torch.no_grad()
def evaluate(model, dv):
    was = model.training; model.eval(); L = G = n = 0.0
    for noisy, clean in dv:
        noisy, clean = noisy.to(DEV), clean.to(DEV)
        loss, est, nn_, cn = losses(model, noisy, clean)
        g = (si_sdr(est, cn).clamp(max=40) - si_sdr(nn_, cn).clamp(max=40))
        L += loss.item() * len(noisy); G += g.sum().item(); n += len(noisy)
    model.train(was); return L / n, G / n


def keep_awake():
    """Ask Windows not to sleep while this process runs (released automatically when it exits; no settings change)."""
    if os.name == "nt":
        import ctypes; ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | 0x00000001)


def main():
    import mlflow
    keep_awake()
    mlflow.set_tracking_uri("sqlite:///" + os.path.join(HERE, "mlflow.db").replace("\\", "/"))
    mlflow.set_experiment("call-denoiser-v4")
    print(f"device: {DEV}" + (f" ({torch.cuda.get_device_name(0)})" if DEV.type == "cuda" else ""), flush=True)
    tr, dv = loaders(args.batch, steps_per_epoch=10 ** 6, dev_items=64 if args.smoke else 256, workers=args.workers)
    model = StreamDenoiser().to(DEV); print(f"parameters: {count_parameters(model):,}", flush=True)
    ema = copy.deepcopy(model).eval()
    for p in ema.parameters(): p.requires_grad_(False)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    step, best, total, run_id = 0, float("inf"), args.steps, None
    if args.resume and os.path.exists(os.path.join(CK, "latest.pth")):
        ck = torch.load(os.path.join(CK, "latest.pth"), map_location=DEV, weights_only=False)
        model.load_state_dict(ck["model"]); ema.load_state_dict(ck["ema"]); opt.load_state_dict(ck["opt"])
        step, best, total, run_id = ck["step"], ck["best"], ck["total"], ck.get("run_id")
        print(f"resumed at step {step}/{total}", flush=True)
    it = iter(tr); t0 = time.time(); start = step
    with mlflow.start_run(run_id=run_id, run_name="smoke" if args.smoke else None) as run:
        if run_id is None:
            mlflow.log_params({"batch": args.batch, "lr": args.lr, "hours": args.hours, "segment_s": 4, "sr": SR,
                               "window": "asym 32ms/10ms, hop 5ms", "model": "linear-2xGRU256-mask", "ema": EMA,
                               "loss": "comp0.3+0.5mrstft", "data": "vbd_train+dns_mix", "task": "call v4 16k"})
        while total is None or step < total:
            noisy, clean = next(it); noisy, clean = noisy.to(DEV, non_blocking=True), clean.to(DEV, non_blocking=True)
            if step < WARMUP: lr = args.lr * (step + 1) / WARMUP
            elif total is not None: lr = 1e-5 + 0.5 * (args.lr - 1e-5) * (1 + math.cos(math.pi * min((step - WARMUP) / max(total - WARMUP, 1), 1.0)))
            else: lr = args.lr
            for g in opt.param_groups: g["lr"] = lr
            loss, *_ = losses(model, noisy, clean)
            opt.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 3.0); opt.step(); step += 1
            with torch.no_grad():
                d = min(EMA, (1 + step) / (10 + step))
                for pe, pm in zip(ema.parameters(), model.parameters()): pe.mul_(d).add_(pm.detach(), alpha=1 - d)
            if total is None and step == start + 300: t300 = time.time()            # skip the cold-cache start
            if total is None and step == start + 800:   # size the run to the time budget from warm steps 300-800
                per = (time.time() - t300) / 500; total = int((args.hours * 3600 - (time.time() - t0)) * 0.95 / per) + step
                print(f"{per:.3f} s/step -> {total} steps for {args.hours} h", flush=True); mlflow.log_param("total_steps", total)
            if step % 100 == 0: print(f"step {step}/{total}  loss {loss.item():.4f}  lr {lr:.2e}", flush=True)
            if step % args.eval_every == 0 or step == total:
                dl, dg = evaluate(ema, dv)
                mlflow.log_metrics({"train_loss": loss.item(), "dev_loss": dl, "dev_sisdr_gain_db": dg, "lr": lr}, step=step)
                print(f"EVAL step {step}: dev loss {dl:.4f} | dev SI-SDR gain {dg:+.2f} dB | {(time.time() - t0) / 3600:.2f} h", flush=True)
                if dl < best:
                    best = dl; torch.save({"model": ema.state_dict(), "step": step, "dev_loss": dl, "dev_gain": dg}, os.path.join(CK, "best.pth"))
                    print("  saved best", flush=True)
                torch.save({"model": model.state_dict(), "ema": ema.state_dict(), "opt": opt.state_dict(), "step": step,
                            "best": best, "total": total, "run_id": run.info.run_id}, os.path.join(CK, "latest.pth"))
        mlflow.log_metric("best_dev_loss", best)
    print("done. next: py -3.12 export_onnx.py", flush=True)


if __name__ == "__main__":
    main()
