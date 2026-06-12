import torch
import torch.nn as nn
import numpy as np
import time

# ---------- data: random correlation matrices via 2-factor model, sampled with Gaussian or Student-t margins ----------

def make_corr_batch(B, D, K=2):
    L = np.random.uniform(0.4, 0.9, size=(B, D, K)) * np.random.choice([-1.0, 1.0], size=(B, D, K))
    cov = L @ L.transpose(0, 2, 1)
    idio = np.random.uniform(0.2, 0.5, size=(B, D))
    cov = cov + np.einsum('bd,de->bde', idio, np.eye(D))
    d = np.sqrt(np.diagonal(cov, axis1=1, axis2=2))
    corr = cov / (d[:, :, None] * d[:, None, :])
    return corr

def sample_batch(B, T, D, dist='gaussian', dof=4, K=2):
    corr = make_corr_batch(B, D, K)
    Lc = np.linalg.cholesky(corr)
    if dist == 'gaussian':
        z = np.random.randn(B, T, D)
    else:
        g = np.random.chisquare(dof, size=(B, T, 1))
        z = np.random.randn(B, T, D) * np.sqrt(dof / g)
    x = np.einsum('bij,btj->bti', Lc, z)
    iu = np.triu_indices(D, k=1)
    Y = corr[:, iu[0], iu[1]]
    return x.astype(np.float32), Y.astype(np.float32)

def empirical_corr_mse(X, Y):
    B, T, D = X.shape
    iu = np.triu_indices(D, k=1)
    preds = np.zeros_like(Y)
    for b in range(B):
        c = np.corrcoef(X[b].T)
        preds[b] = c[iu]
    return float(((preds - Y) ** 2).mean())

# ---------- attention variants ----------

class StdAttn(nn.Module):
    """Standard scaled dot-product attention: output = attn-weighted MEAN of values."""
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.h, self.hd = n_heads, d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)
    def forward(self, x):
        B, T, D = x.shape
        qkv = self.qkv(x).view(B, T, 3, self.h, self.hd).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        a = (q @ k.transpose(-2, -1)) / self.hd ** 0.5
        a = a.softmax(-1)
        o = (a @ v).permute(0, 2, 1, 3).reshape(B, T, D)
        return self.proj(o)

class DensityAttn(nn.Module):
    """Covariance attention: each head outputs attn-weighted MEAN *and* attn-weighted
    COVARIANCE of its values, i.e. the joint shape of what it's looking at, not just
    its location."""
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.h, self.hd = n_heads, d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.cov_ln = nn.LayerNorm(self.hd * self.hd)
        self.cov_proj = nn.Linear(self.hd * self.hd, self.hd)
        self.proj = nn.Linear(2 * d_model, d_model)
    def forward(self, x):
        B, T, D = x.shape
        H, hd = self.h, self.hd
        qkv = self.qkv(x).view(B, T, 3, H, hd).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        a = (q @ k.transpose(-2, -1)) / hd ** 0.5
        a = a.softmax(-1)
        mean_o = a @ v  # B,H,T,hd  (first moment)
        v_outer = torch.einsum('bhtd,bhte->bhtde', v, v)
        second = torch.einsum('bhij,bhjde->bhide', a, v_outer)
        mean_outer = torch.einsum('bhid,bhie->bhide', mean_o, mean_o)
        cov = second - mean_outer  # attn-weighted covariance per query position
        cov_flat = cov.reshape(B, H, T, hd * hd)
        cov_flat = self.cov_ln(cov_flat)
        cov_o = self.cov_proj(cov_flat)
        mean_o = mean_o.permute(0, 2, 1, 3).reshape(B, T, D)
        cov_o = cov_o.permute(0, 2, 1, 3).reshape(B, T, D)
        combo = torch.cat([mean_o, cov_o], dim=-1)
        return self.proj(combo)

class Block(nn.Module):
    def __init__(self, d_model, n_heads, attn_cls, ff_mult=4):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = attn_cls(d_model, n_heads)
        self.ln2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(nn.Linear(d_model, ff_mult * d_model), nn.GELU(), nn.Linear(ff_mult * d_model, d_model))
    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x

class CorrEstimator(nn.Module):
    def __init__(self, D, d_model=64, n_heads=4, n_layers=2, attn_cls=StdAttn):
        super().__init__()
        self.embed = nn.Linear(D, d_model)
        self.blocks = nn.ModuleList([Block(d_model, n_heads, attn_cls) for _ in range(n_layers)])
        self.ln_f = nn.LayerNorm(d_model)
        out_dim = D * (D - 1) // 2
        self.head = nn.Sequential(nn.Linear(d_model, d_model), nn.GELU(), nn.Linear(d_model, out_dim))
    def forward(self, x):
        h = self.embed(x)
        for blk in self.blocks:
            h = blk(h)
        h = self.ln_f(h).mean(dim=1)  # pool over the T samples (set, not sequence)
        return self.head(h)

# ---------- training ----------

def train_and_eval(D, dist, attn_cls, Xte, Yte, steps=800, T=64, B=64, lr=3e-3, log_every=None):
    torch.manual_seed(123)
    model = CorrEstimator(D, attn_cls=attn_cls)
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)
    history = []
    for step in range(steps):
        Xb, Yb = sample_batch(B, T, D, dist=dist)
        Xb = torch.from_numpy(Xb); Yb = torch.from_numpy(Yb)
        pred = model(Xb)
        loss = ((pred - Yb) ** 2).mean()
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if log_every and (step % log_every == 0 or step == steps - 1):
            history.append((step, loss.item()))
    model.eval()
    with torch.no_grad():
        pred = model(torch.from_numpy(Xte))
        mse = ((pred - torch.from_numpy(Yte)) ** 2).mean().item()
    nparams = sum(p.numel() for p in model.parameters())
    return mse, nparams, history

if __name__ == '__main__':
    Ds = [4, 8, 16]
    dists = ['gaussian', 'student_t']
    T_EVAL, B_EVAL = 64, 512
    STEPS = 800
    results = []
    for D in Ds:
        for dist in dists:
            eval_seed = 9000 + D * 2 + (0 if dist == 'gaussian' else 1)
            data_seed = 5000 + D * 2 + (0 if dist == 'gaussian' else 1)

            np.random.seed(eval_seed)
            Xte, Yte = sample_batch(B_EVAL, T_EVAL, D, dist=dist)
            base_mse = empirical_corr_mse(Xte, Yte)
            target_var = float(((Yte - Yte.mean(0)) ** 2).mean())

            row = {'D': D, 'dist': dist, 'empirical_corr_mse': base_mse, 'predict_mean_mse': target_var}
            for name, cls in [('standard', StdAttn), ('density', DensityAttn)]:
                np.random.seed(data_seed)  # both models see identical training data sequence
                t0 = time.time()
                log_every = 200 if D == 8 else None
                mse, nparams, hist = train_and_eval(D, dist, cls, Xte, Yte, steps=STEPS, log_every=log_every)
                dt = time.time() - t0
                row[f'{name}_mse'] = mse
                row[f'{name}_params'] = nparams
                row[f'{name}_time'] = dt
                if hist:
                    row[f'{name}_history'] = hist
                print(f"D={D:2d} dist={dist:10s} model={name:8s} mse={mse:.5f} params={nparams:6d} time={dt:5.1f}s")
            results.append(row)

    print("\n=== SUMMARY ===")
    for r in results:
        improve = 100 * (1 - r['density_mse'] / r['standard_mse'])
        print(f"D={r['D']:2d} {r['dist']:10s} | empirical={r['empirical_corr_mse']:.5f} "
              f"predict-mean={r['predict_mean_mse']:.5f} | standard={r['standard_mse']:.5f} "
              f"density={r['density_mse']:.5f} | density improvement={improve:+.1f}%")
        if 'standard_history' in r:
            print("   std history :", r['standard_history'])
            print("   dens history:", r['density_history'])
