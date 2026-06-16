import torch
import os
import time
from models import SpectralRecursiveGMAEstimator
from pipeline import sample_batch, reconstruct_corr_matrix, is_psd_batch

def spectral_loss_func(pred_tri, target_tri, D):
    A = reconstruct_corr_matrix(pred_tri, D)
    B = reconstruct_corr_matrix(target_tri, D)
    e_a, _ = torch.linalg.eigh(A)
    e_b, _ = torch.linalg.eigh(B)
    return torch.mean((e_a - e_b)**2)

def train_ultra(D, steps=500, dist='gaussian'):
    print(f"\n--- ULTRA TRAINING D={D} ({dist}) ---")
    model = SpectralRecursiveGMAEstimator(D, d_model=64, n_layers=4, n_groups=2)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps)

    os.makedirs("checkpoints_ultra", exist_ok=True)
    best_mse = 1e9

    model.train()
    for s in range(steps):
        Xb, Yb = sample_batch(64, 64, D, dist=dist)
        pred = model(Xb)
        mse_loss = torch.mean((pred - Yb)**2)
        spec_loss = spectral_loss_func(pred, Yb, D)
        loss = 0.5 * mse_loss + 0.5 * spec_loss
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step()

    # Final Eval
    model.eval()
    with torch.no_grad():
        Xte, Yte = sample_batch(512, 64, D, dist=dist)
        pred = model(Xte)
        final_mse = torch.mean((pred - Yte)**2).item()
        psd_rate = is_psd_batch(reconstruct_corr_matrix(pred, D)).float().mean().item()
        print(f"Final Eval D={D}: MSE={final_mse:.6f}, PSD={psd_rate:.1%}")
    torch.save(model.state_dict(), f"checkpoints_ultra/best_d{D}.pt")

if __name__ == "__main__":
    for D in [16, 32]:
        train_ultra(D, steps=300)
