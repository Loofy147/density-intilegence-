import torch
import numpy as np
import time
from pipeline import sample_batch, empirical_corr_mse
from models import BaseCorrEstimator, AnchoredCorrEstimator, ShrinkageCorrEstimator, StdAttn, MomentModulatedAttn, GroupedMomentAttn

def train_and_eval(D, dist, model_cls, Xte, Yte, steps=800, T=64, B=64, lr=3e-3, **kwargs):
    torch.manual_seed(123)
    model = model_cls(D, **kwargs)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for step in range(steps):
        Xb, Yb = sample_batch(B, T, D, dist=dist)
        Xb, Yb = torch.from_numpy(Xb), torch.from_numpy(Yb)
        pred = model(Xb)
        loss = ((pred - Yb) ** 2).mean()
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
    model.eval()
    with torch.no_grad():
        pred = model(torch.from_numpy(Xte))
        mse = ((pred - torch.from_numpy(Yte)) ** 2).mean().item()
    nparams = sum(p.numel() for p in model.parameters())
    return mse, nparams

if __name__ == '__main__':
    Ds = [4, 8, 16]
    dists = ['gaussian', 'student_t']
    T_EVAL, B_EVAL = 64, 512
    STEPS = 400
    results = []
    for D in Ds:
        for dist in dists:
            eval_seed = 9000 + D * 2 + (0 if dist == 'gaussian' else 1)
            data_seed = 5000 + D * 2 + (0 if dist == 'gaussian' else 1)
            np.random.seed(eval_seed)
            Xte, Yte = sample_batch(B_EVAL, T_EVAL, D, dist=dist)
            base_mse = empirical_corr_mse(Xte, Yte)
            row = {'D': D, 'dist': dist, 'empirical_corr_mse': base_mse}

            variants = [
                ('standard', BaseCorrEstimator, {'attn_cls': StdAttn}),
                ('mma_map', BaseCorrEstimator, {'attn_cls': MomentModulatedAttn}),
                ('gma', BaseCorrEstimator, {'attn_cls': GroupedMomentAttn}),
                ('anchored', AnchoredCorrEstimator, {}),
                ('shrinkage', ShrinkageCorrEstimator, {})
            ]
            for name, cls, kwargs in variants:
                np.random.seed(data_seed)
                t0 = time.time()
                mse, nparams = train_and_eval(D, dist, cls, Xte, Yte, steps=STEPS, **kwargs)
                dt = time.time() - t0
                row[f'{name}_mse'] = mse
                row[f'{name}_params'] = nparams
                row[f'{name}_time'] = dt
                print(f"D={D:2d} dist={dist:10s} model={name:8s} mse={mse:.5f} params={nparams:6d} time={dt:4.1f}s", flush=True)
            results.append(row)

    print("\n=== SUMMARY ===")
    for r in results:
        improve_mma = 100 * (1 - r['mma_map_mse'] / r['standard_mse'])
        improve_gma = 100 * (1 - r['gma_mse'] / r['standard_mse'])
        improve_anc = 100 * (1 - r['anchored_mse'] / r['standard_mse'])
        improve_shrk = 100 * (1 - r['shrinkage_mse'] / r['standard_mse'])
        print(f"D={r['D']:2d} {r['dist']:10s} | empirical={r['empirical_corr_mse']:.5f} | "
              f"std={r['standard_mse']:.5f} mma={r['mma_map_mse']:.5f} gma={r['gma_mse']:.5f} anc={r['anchored_mse']:.5f} shr={r['shrinkage_mse']:.5f}")
        print(f"   mma={improve_mma:+.1f}% | gma={improve_gma:+.1f}% | anc={improve_anc:+.1f}% | shr={improve_shrk:+.1f}%")
