import torch
import os
import collections
from models import SpectralRecursiveGMAEstimator
from pipeline import sample_batch, reconstruct_corr_matrix, is_psd_batch

def spectral_loss_func(pred_tri, target_tri, D):
    A = reconstruct_corr_matrix(pred_tri, D)
    B = reconstruct_corr_matrix(target_tri, D)
    e_a, _ = torch.linalg.eigh(A)
    e_b, _ = torch.linalg.eigh(B)
    return torch.mean((e_a - e_b)**2)

def train_ultra(dims=[16, 32, 64], steps_per_dim=400, dist='gaussian'):
    print(f"\n--- SCALING TO D=64 ({dist}) ---")
    os.makedirs("checkpoints_ultra", exist_ok=True)

    for D in dims:
        print(f"\n>> Training D={D}")
        # Scale model capacity for higher D
        d_model = 128 if D >= 32 else 64
        n_layers = 6 if D >= 32 else 4

        model = SpectralRecursiveGMAEstimator(D, d_model=d_model, n_layers=n_layers)
        opt = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-3)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps_per_dim)

        model.train()
        for s in range(steps_per_dim):
            Xb, Yb = sample_batch(32, 64, D, dist=dist) # Reduce batch size for D=64

            pred = model(Xb)
            mse_loss = torch.mean((pred - Yb)**2)
            spec_loss = spectral_loss_func(pred, Yb, D)
            loss = 0.5 * mse_loss + 0.5 * spec_loss

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()

            if s % 100 == 0:
                print(f" Step {s}/{steps_per_dim}: Loss={loss.item():.6f} (MSE={mse_loss.item():.6f})")

        model.eval()
        with torch.no_grad():
            Xte, Yte = sample_batch(128, 64, D, dist=dist)
            pred = model(Xte)
            final_mse = torch.mean((pred - Yte)**2).item()
            psd_rate = is_psd_batch(reconstruct_corr_matrix(pred, D)).float().mean().item()
            print(f"Final Eval D={D}: MSE={final_mse:.6f}, PSD={psd_rate:.1%}")

        torch.save(model.state_dict(), f"checkpoints_ultra/best_d{D}.pt")

if __name__ == "__main__":
    # Rapid training for D=64 scaling demonstration
    train_ultra(dims=[16, 32, 64], steps_per_dim=150)
