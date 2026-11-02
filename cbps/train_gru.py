"""Train the GRU decoder in PyTorch (CPU), then hand weights to the numpy GRUDecoder."""
from __future__ import annotations

import numpy as np
import torch
from torch import nn

from .data import EVAL_START
from .decoders import GRUDecoder
from .metrics import r2


class GRUNet(nn.Module):
    def __init__(self, n_in: int, hidden: int):
        super().__init__()
        self.gru = nn.GRU(n_in, hidden, batch_first=True)
        self.out = nn.Linear(hidden, 2)

    def forward(self, x):
        h, _ = self.gru(x)
        return self.out(h)


def train_gru(Xtr, Vtr, Xva, Vva, hidden: int, seed: int = 0, epochs: int = 300,
              patience: int = 25, lr: float = 3e-3, wd: float = 1e-4, in_drop: float = 0.2,
              batch: int = 64, loss_from: int = 5, verbose: bool = False,
              stream=None, crop: int = 200, crops_per_epoch: int = 1600, aug_gain: float = 0.2,
              aug_offset: float = 0.2) -> tuple[GRUDecoder, dict]:
    """Early-stops on validation R^2 over the scored window (bins EVAL_START..end).

    Default: train on onset-aligned trial windows (Xtr, Vtr).
    stream=(X [B,U], V [B,2], valid [B]): instead train on random `crop`-bin
    crops of the continuous session, loss masked to bins with kinematics, with
    per-crop per-channel gain/offset perturbations (Sussillo et al. 2016) so
    the network cannot lean on trial timing or exact channel baselines.
    """
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    flat = Xtr.reshape(-1, Xtr.shape[-1]) if stream is None else stream[0]
    mu = flat.mean(0)
    sd = np.maximum(flat.std(0), 0.1)
    v_sd = (Vtr.reshape(-1, 2) if stream is None else stream[1][stream[2]]).std(0)
    norm = lambda X: torch.tensor((X - mu) / sd, dtype=torch.float32)
    xva = norm(Xva)
    if stream is None:
        xtr = norm(Xtr)
        vtr = torch.tensor(Vtr / v_sd, dtype=torch.float32)
        mtr = torch.ones(vtr.shape[:2], dtype=torch.bool)
    else:
        xs, vs, ms = norm(stream[0]), torch.tensor(stream[1] / v_sd, dtype=torch.float32), torch.tensor(stream[2])

    net = GRUNet(Xva.shape[-1], hidden)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=wd)
    drop = nn.Dropout(in_drop)
    best, best_state, bad, hist = -np.inf, None, 0, []
    U = Xva.shape[-1]
    for ep in range(epochs):
        net.train()
        if stream is not None:  # fresh random crops + perturbations every epoch
            st = torch.tensor(rng.integers(0, len(xs) - crop, crops_per_epoch))
            ix = st[:, None] + torch.arange(crop)
            g = torch.tensor(np.exp(aug_gain * rng.standard_normal((crops_per_epoch, 1, U))), dtype=torch.float32)
            o = torch.tensor(aug_offset * rng.standard_normal((crops_per_epoch, 1, U)), dtype=torch.float32)
            xtr, vtr, mtr = xs[ix] * g + o, vs[ix], ms[ix]
            loss_from = 25
        perm = torch.randperm(len(xtr))
        for i in range(0, len(perm), batch):
            j = perm[i:i + batch]
            pred = net(drop(xtr[j]))
            m = mtr[j, loss_from:]
            if not m.any():
                continue
            loss = ((pred[:, loss_from:] - vtr[j, loss_from:]) ** 2)[m].mean()
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
        net.eval()
        with torch.no_grad():
            pv = net(xva).numpy() * v_sd
        score = r2(Vva[:, EVAL_START:], pv[:, EVAL_START:])
        hist.append(score)
        if verbose:
            print(f"  ep {ep:3d} loss {loss.item():.3f} val R2 {score:.4f}")
        if score > best + 1e-4:
            best, bad = score, 0
            best_state = {k: v.clone() for k, v in net.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    net.load_state_dict(best_state)
    sdict = {k: v.numpy().astype(np.float64) for k, v in net.state_dict().items()}
    dec = GRUDecoder(sdict["gru.weight_ih_l0"], sdict["gru.weight_hh_l0"], sdict["gru.bias_ih_l0"],
                     sdict["gru.bias_hh_l0"], sdict["out.weight"], sdict["out.bias"], mu, sd, v_sd)
    return dec, {"best_val_r2": best, "epochs_run": len(hist), "val_curve": hist}
