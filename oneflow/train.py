"""Masked flow matching training loop (AdamW, warmup + cosine, gradient clipping, EMA)."""

import copy
import json
import math
import time
from pathlib import Path

import numpy as np
import torch

from .data import Transform
from .masks import canonical, sample_masks, GEN, OBS
from .model import VelocityNet


def pick_device(name="auto"):
    if name != "auto":
        return torch.device(name)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def train(theta, x, out, steps=50_000, batch=512, lr=5e-4, lr_min=1e-5, warmup=2000,
          weight_decay=1e-4, clip=10.0, ema_decay=0.999, hidden=256, n_blocks=4,
          mask_weights=None, seed=0, device="auto", log_every=1000, val_frac=0.1, val_max=3000,
          patience=8):
    """patience: stop when the validation loss has not improved for this many evaluations
    (evaluations happen every log_every steps); None trains for the full schedule."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    device = pick_device(device)
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)

    # hold out a validation set for checkpoint selection
    n_val = min(val_max, int(val_frac * len(theta)))
    perm = rng.permutation(len(theta))
    theta_v, x_v = theta[perm[:n_val]], x[perm[:n_val]]
    theta, x = theta[perm[n_val:]], x[perm[n_val:]]
    tf = Transform(theta, x, rng).to(device)
    theta_t = torch.as_tensor(theta, dtype=torch.float32, device=device)
    x_t = torch.as_tensor(x, dtype=torch.float32, device=device)
    n, d = theta_t.shape[0], theta_t.shape[1] + x_t.shape[1]
    # fixed validation batch: data, dequantisation noise, masks, base noise and times
    vgen = torch.Generator(device=device).manual_seed(seed)
    z1_v = tf.joint_fwd(torch.as_tensor(theta_v, dtype=torch.float32, device=device),
                        torch.as_tensor(x_v, dtype=torch.float32, device=device), vgen)
    z0_v = torch.randn(z1_v.shape, generator=vgen, device=device)
    t_v = torch.rand(n_val, generator=vgen, device=device)
    masks_v = {"mix": torch.as_tensor(sample_masks(n_val, np.random.default_rng(seed), mask_weights), device=device),
               "posterior": torch.as_tensor(np.tile(canonical("posterior"), (n_val, 1)), device=device)}

    def fm_loss(net, z1, z0, t, mask):
        m_gen, m_obs = (mask == GEN).float(), (mask == OBS).float()
        zt = m_obs * z1 + m_gen * ((1 - t[:, None]) * z0 + t[:, None] * z1)
        r = m_gen * (net(zt, t, mask) - (z1 - z0))
        return ((r ** 2).sum(1) / m_gen.sum(1)).mean()

    @torch.no_grad()
    def validate(net):
        return {k: float(np.mean([fm_loss(net, z1_v[i:i + 1024], z0_v[i:i + 1024], t_v[i:i + 1024], m[i:i + 1024]).item()
                                  for i in range(0, n_val, 1024)])) for k, m in masks_v.items()}

    model = VelocityNet(d, hidden, n_blocks).to(device)
    ema = copy.deepcopy(model).eval()
    for p in ema.parameters():
        p.requires_grad_(False)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.999), weight_decay=weight_decay)

    def lr_at(step):
        if step < warmup:
            return (step + 1) / warmup
        f = (step - warmup) / max(1, steps - warmup)
        return (lr_min + 0.5 * (lr - lr_min) * (1 + math.cos(math.pi * f))) / lr

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_at)
    log, t0 = [], time.time()
    best = {"val": np.inf, "step": 0, "state": None}
    since_best, stopped_at = 0, steps
    for step in range(steps):
        idx = torch.as_tensor(rng.integers(0, n, batch), device=device)
        z1 = tf.joint_fwd(theta_t[idx], x_t[idx])
        mask = torch.as_tensor(sample_masks(batch, rng, mask_weights), device=device)
        loss = fm_loss(model, z1, torch.randn_like(z1), torch.rand(batch, device=device), mask)

        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        opt.step()
        sched.step()
        with torch.no_grad():
            for pe, pm in zip(ema.parameters(), model.parameters()):
                pe.lerp_(pm, 1 - ema_decay)

        if (step + 1) % log_every == 0 or step == 0:
            val = validate(ema)
            if val["mix"] < best["val"]:
                best = {"val": val["mix"], "step": step + 1, "state": copy.deepcopy(ema.state_dict())}
                since_best = 0
            else:
                since_best += 1
            log.append({"step": step + 1, "loss": loss.item(), "val_mix": val["mix"], "val_posterior": val["posterior"],
                        "time": time.time() - t0})
            print(f"step {step + 1:6d}  loss {loss.item():.4f}  val mix {val['mix']:.4f}  val post {val['posterior']:.4f}  "
                  f"lr {sched.get_last_lr()[0]:.2e}  {time.time() - t0:6.0f}s", flush=True)
            if patience is not None and since_best >= patience:
                stopped_at = step + 1
                print(f"early stop at step {stopped_at}: no improvement for {patience} evaluations", flush=True)
                break

    print(f"best validation loss {best['val']:.4f} at step {best['step']}", flush=True)
    ckpt = {
        "model": {k: v.cpu() for k, v in best["state"].items()},
        "model_last": {k: v.cpu() for k, v in ema.state_dict().items()},
        "best_step": best["step"],
        "stopped_at": stopped_at,
        "transform": {k: v.cpu() for k, v in tf.state_dict().items()},
        "config": {"d": d, "hidden": hidden, "n_blocks": n_blocks, "steps": steps, "batch": batch,
                   "lr": lr, "budget": int(n) + n_val, "n_val": n_val, "seed": seed, "mask_weights": mask_weights,
                   "patience": patience, "train_seconds": time.time() - t0},
        "log": log,
    }
    torch.save(ckpt, out / "model.pt")
    (out / "config.json").write_text(json.dumps(ckpt["config"], indent=2))
    return ckpt
