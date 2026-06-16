import torch
import numpy as np
import time
from pipeline import sample_batch, empirical_corr_mse, reconstruct_corr_matrix, is_psd_batch
from models import RecursiveGMAEstimator,  BaseCorrEstimator, AnchoredCorrEstimator, ShrinkageCorrEstimator, StdAttn, MomentModulatedAttn, GroupedMomentAttn

def fisher_z_loss(pred, target):
    """Loss in Fisher Z-space."""
    z_pred = torch.atanh(torch.clamp(pred, -0.99, 0.99))
    z_target = torch.atanh(torch.clamp(target, -0.99, 0.99))
    return torch.mean((z_pred - z_target)**2)

def log_det_loss(pred_tri, target_tri, D):
    """
    Log-Determinant divergence (Bregman divergence for SPD matrices).
    d(A, B) = tr(A B^-1) - log det(A B^-1) - n
    """
    A = reconstruct_corr_matrix(pred_tri, D)
    B = reconstruct_corr_matrix(target_tri, D)

    # Regularize to ensure invertibility
    B = B + torch.eye(D, device=B.device) * 1e-4
    A = A + torch.eye(D, device=A.device) * 1e-4

    # tr(A B^-1)
    B_inv = torch.linalg.inv(B)
    tr_ab_inv = torch.matmul(A, B_inv).diagonal(dim1=-2, dim2=-1).sum(-1)

    # log det(A B^-1) = log det(A) - log det(B)
    log_det_a = torch.logdet(A)
    log_det_b = torch.logdet(B)

    loss = tr_ab_inv - (log_det_a - log_det_b) - D
    return loss.mean()

def train_and_eval(D, dist, model_cls, Xte, Yte, steps=800, T=64, B=64, lr=3e-3, loss_type='mse', **kwargs):
    torch.manual_seed(123)
    model = model_cls(D, **kwargs)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    # Using OneCycleLR for better convergence
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps)

    model.train()
    for step in range(steps):
        Xb, Yb = sample_batch(B, T, D, dist=dist)
        pred = model(Xb)

        if loss_type == 'mse':
            loss = torch.mean((pred - Yb)**2)
        elif loss_type == 'fisher':
            loss = fisher_z_loss(pred, Yb)
        elif loss_type == 'log_det':
            loss = log_det_loss(pred, Yb, D)
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

        # PSD Check
        full_mats = reconstruct_corr_matrix(pred, D)
        psd_flags = is_psd_batch(full_mats)
        psd_rate = psd_flags.float().mean().item()

    nparams = sum(p.numel() for p in model.parameters())
    return mse, nparams, psd_rate

if __name__ == '__main__':
    Ds = [4, 16]
    dists = ['gaussian', 'student_t']
    T_EVAL, B_EVAL = 64, 512
    STEPS = 100
    results = []

    for D in Ds:
        for dist in dists:
            torch.manual_seed(9000 + D)
            Xte, Yte = sample_batch(B_EVAL, T_EVAL, D, dist=dist)
            base_mse = empirical_corr_mse(Xte, Yte)
            row = {'D': D, 'dist': dist, 'empirical_corr_mse': base_mse}

            variants = [('recursive_gma', RecursiveGMAEstimator, {'n_layers': 4, 'n_groups': 2}),
                ('standard', BaseCorrEstimator, {'attn_cls': StdAttn}),
                ('gma', BaseCorrEstimator, {'attn_cls': GroupedMomentAttn, 'n_groups': 2}),
                ('shrinkage_mse', ShrinkageCorrEstimator, {'loss_type': 'mse'}),
                ('shrinkage_fisher', ShrinkageCorrEstimator, {'loss_type': 'fisher'}),
            ]

            for name, cls, kwargs in variants:
                t0 = time.time()
                mse, nparams, psd_rate = train_and_eval(D, dist, cls, Xte, Yte, steps=STEPS, **kwargs)
                dt = time.time() - t0
                row[f'{name}_mse'] = mse
                row[f'{name}_psd'] = psd_rate
                print(f"D={D:2d} dist={dist:10s} model={name:15s} mse={mse:.5f} psd={psd_rate:.1%} time={dt:4.1f}s", flush=True)
            results.append(row)

    print("\n=== SUMMARY ===")
    for r in results:
        improve_gma = 100 * (1 - r['gma_mse'] / r['standard_mse'])
        improve_shrk = 100 * (1 - r['shrinkage_fisher_mse'] / r['standard_mse'])
        print(f"D={r['D']:2d} {r['dist']:10s} | empirical={r['empirical_corr_mse']:.5f} | "
              f"std={r['standard_mse']:.5f} gma={r['gma_mse']:.5f} shr_fish={r['shrinkage_fisher_mse']:.5f}")
        print(f"   gma={improve_gma:+.1f}% | shr_fish={improve_shrk:+.1f}% | PSD Pass: {r['shrinkage_fisher_psd']:.1%}")
