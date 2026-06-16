import torch
import numpy as np
import time
from pipeline import sample_batch, empirical_corr_mse, reconstruct_corr_matrix, is_psd_batch
from models import SpectralRecursiveGMAEstimator, BaseCorrEstimator, RecursiveGMAEstimator, ShrinkageCorrEstimator, StdAttn, MomentModulatedAttn, GroupedMomentAttn

def fisher_z_loss(pred, target):
    z_pred = torch.atanh(torch.clamp(pred, -0.995, 0.995))
    z_target = torch.atanh(torch.clamp(target, -0.995, 0.995))
    return torch.mean((z_pred - z_target)**2)

def spectral_loss(pred_tri, target_tri, D):
    """Penalizes deviation in eigenvalue distributions."""
    A = reconstruct_corr_matrix(pred_tri, D)
    B = reconstruct_corr_matrix(target_tri, D)
    e_a, _ = torch.linalg.eigh(A)
    e_b, _ = torch.linalg.eigh(B)
    return torch.mean((e_a - e_b)**2)

def train_and_eval(D, dist, model_cls, Xte, Yte, steps=300, T=64, B=64, lr=1e-3, loss_type='mse', **kwargs):
    torch.manual_seed(123)
    model = model_cls(D, **kwargs)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps)

    model.train()
    for step in range(steps):
        Xb, Yb = sample_batch(B, T, D, dist=dist)
        pred = model(Xb)

        if loss_type == 'mse':
            loss = torch.mean((pred - Yb)**2)
        elif loss_type == 'fisher':
            loss = fisher_z_loss(pred, Yb)
        elif loss_type == 'spectral':
            loss = 0.5 * torch.mean((pred-Yb)**2) + 0.5 * spectral_loss(pred, Yb, D)
        else:
            loss = torch.mean((pred - Yb)**2)

        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()

    model.eval()
    with torch.no_grad():
        pred = model(Xte)
        mse = torch.mean((pred - Yte)**2).item()
        psd_rate = is_psd_batch(reconstruct_corr_matrix(pred, D)).float().mean().item()

    return mse, sum(p.numel() for p in model.parameters()), psd_rate

if __name__ == '__main__':
    Ds = [4, 16]
    dists = ['gaussian', 'student_t']
    STEPS = 100
    results = []

    for D in Ds:
        Xte, Yte = sample_batch(512, 64, D, dist='gaussian')
        for dist in dists:
            row = {'D': D, 'dist': dist}
            variants = [
                ('std', BaseCorrEstimator, {}),
                ('spectral_gma', SpectralRecursiveGMAEstimator, {'n_layers': 4, 'n_groups': 2}),
            ]
            for name, cls, kwargs in variants:
                t0 = time.time()
                l_type = 'spectral' if 'spectral' in name else 'mse'
                mse, _, psd = train_and_eval(D, dist, cls, Xte, Yte, steps=STEPS, loss_type=l_type, **kwargs)
                print(f"D={D:2d} dist={dist:10s} model={name:15s} mse={mse:.5f} psd={psd:.1%}")
                row[f'{name}_mse'] = mse
            results.append(row)
