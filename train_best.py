import torch
import os
from models import RecursiveGMAEstimator
from pipeline import sample_batch

def train_best(D=16, steps=1000, dist='gaussian'):
    model = RecursiveGMAEstimator(D, n_layers=4, n_groups=2)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps)

    os.makedirs("checkpoints", exist_ok=True)
    best_loss = 1e9

    print(f"Starting training RecursiveGMA (D={D}, {dist})...")
    model.train()
    for s in range(steps):
        Xb, Yb = sample_batch(64, 64, D, dist=dist)
        pred = model(Xb)
        loss = torch.mean((pred - Yb)**2)

        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()

        if s % 100 == 0:
            print(f"Step {s}: Loss {loss.item():.6f}")
            if loss.item() < best_loss:
                best_loss = loss.item()
                torch.save(model.state_dict(), f"checkpoints/best_gma_d{D}.pt")

    print("Training complete.")

if __name__ == "__main__":
    train_best(D=16, steps=300)
