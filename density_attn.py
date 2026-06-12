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
        o = (a @ v).permute(0, 2, 1, 3).reshape(B, T, -1)
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
        mean_o = mean_o.permute(0, 2, 1, 3).reshape(B, T, -1)
        cov_o = cov_o.permute(0, 2, 1, 3).reshape(B, T, -1)
        combo = torch.cat([mean_o, cov_o], dim=-1)
        return self.proj(combo)

class MomentAttn(nn.Module):
    """Improved Density variant: uses weighted mean and weighted variance.
    Complexity: O(T^2 * d + T * d), avoiding full covariance O(T^2 * d + T * d^2)."""
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.h, self.hd = n_heads, d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(2 * d_model, d_model)
    def forward(self, x):
        B, T, D = x.shape
        H, hd = self.h, self.hd
        qkv = self.qkv(x).view(B, T, 3, H, hd).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        a = (q @ k.transpose(-2, -1)) / hd ** 0.5
        a = a.softmax(-1)

        mean_o = a @ v # B, H, T, hd
        mean_sq_o = a @ (v**2) # B, H, T, hd
        var_o = torch.relu(mean_sq_o - mean_o**2)

        mean_o = mean_o.permute(0, 2, 1, 3).reshape(B, T, -1)
        var_o = var_o.permute(0, 2, 1, 3).reshape(B, T, -1)
        combo = torch.cat([mean_o, var_o], dim=-1)
        return self.proj(combo)

class GroupedMomentAttn(nn.Module):
    """Grouped Moment Attention (GMA):
    Every head computes a weighted mean (full-rank), but only G groups
    compute the variance (shared moments), reducing parameter overhead."""
    def __init__(self, d_model, n_heads, n_groups=1):
        super().__init__()
        self.h, self.hd = n_heads, d_model // n_heads
        self.g = n_groups
        self.qkv = nn.Linear(d_model, 3 * d_model)
        # Final projection: h * hd (means) + g * hd (variances)
        self.proj = nn.Linear(d_model + self.g * self.hd, d_model)

    def forward(self, x):
        B, T, D = x.shape
        H, hd, G = self.h, self.hd, self.g
        qkv = self.qkv(x).view(B, T, 3, H, hd).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        a = (q @ k.transpose(-2, -1)) / hd ** 0.5
        a = a.softmax(-1)

        # 1. Full-Rank Means
        mean_o = a @ v # B, H, T, hd

        # 2. Shared Moments (Grouped Variance)
        # We take the first 'G' heads to represent our moment groups
        v_g = v[:, :G, :, :]
        a_g = a[:, :G, :, :]

        mean_sq_g = a_g @ (v_g**2)
        var_g = torch.relu(mean_sq_g - (mean_o[:, :G, :, :]**2))

        # 3. Combine
        mean_o_flat = mean_o.permute(0, 2, 1, 3).reshape(B, T, -1) # B, T, d_model
        var_g_flat = var_g.permute(0, 2, 1, 3).reshape(B, T, -1) # B, T, G * hd

        combo = torch.cat([mean_o_flat, var_g_flat], dim=-1)
        return self.proj(combo)

class MomentModulatedAttn(nn.Module):
    """Moment-Modulated Attention (MMA):
    Computes weighted mean and variance. The variance is used to modulate (gate)
    the mean output, allowing the model to suppress noisy information."""
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.h, self.hd = n_heads, d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.gate = nn.Sequential(nn.Linear(d_model, d_model), nn.Sigmoid())
        self.proj = nn.Linear(d_model, d_model)
    def forward(self, x):
        B, T, D = x.shape
        H, hd = self.h, self.hd
        qkv = self.qkv(x).view(B, T, 3, H, hd).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        a = (q @ k.transpose(-2, -1)) / hd ** 0.5
        a = a.softmax(-1)

        mean_o = a @ v
        mean_sq_o = a @ (v**2)
        var_o = torch.relu(mean_sq_o - mean_o**2)

        mean_o = mean_o.permute(0, 2, 1, 3).reshape(B, T, -1)
        var_o = var_o.permute(0, 2, 1, 3).reshape(B, T, -1)

        # Modulate mean by variance-based gate
        modulated = mean_o * self.gate(var_o)
        return self.proj(modulated)

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
    def __init__(self, D, d_model=64, n_heads=4, n_layers=2, attn_cls=StdAttn, map_pooling=True):
        super().__init__()
        self.map_pooling = map_pooling
        self.embed = nn.Linear(D, d_model)
        self.blocks = nn.ModuleList([Block(d_model, n_heads, attn_cls) for _ in range(n_layers)])
        self.ln_f = nn.LayerNorm(d_model)
        out_dim = D * (D - 1) // 2

        # If MAP, input to head is 2 * d_model (mean + var)
        head_in = 2 * d_model if map_pooling else d_model
        self.head = nn.Sequential(nn.Linear(head_in, d_model), nn.GELU(), nn.Linear(d_model, out_dim))

    def forward(self, x):
        # Input Standardization: center and scale raw features before embedding
        x = (x - x.mean(dim=1, keepdim=True)) / (x.std(dim=1, keepdim=True) + 1e-6)
        h = self.embed(x)
        for blk in self.blocks:
            h = blk(h)
        h = self.ln_f(h)

        if self.map_pooling:
            # Moment-Augmented Pooling (MAP)
            pool_mean = h.mean(dim=1)
            pool_var = h.var(dim=1)
            pool = torch.cat([pool_mean, pool_var], dim=-1)
        else:
            pool = h.mean(dim=1)

        return self.head(pool)

# ---------- training ----------

def train_and_eval(D, dist, attn_cls, Xte, Yte, steps=800, T=64, B=64, lr=3e-3, log_every=None, map_pooling=True):
    torch.manual_seed(123)
    model = CorrEstimator(D, attn_cls=attn_cls, map_pooling=map_pooling)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
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
    STEPS = 400
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
            variants = [
                ('standard', StdAttn, True),
                ('density', DensityAttn, True),
                ('moment', MomentAttn, True),
                ('gma', GroupedMomentAttn, True),
                ('mma', MomentModulatedAttn, True)
            ]
            for name, cls, map_p in variants:
                np.random.seed(data_seed)  # both models see identical training data sequence
                t0 = time.time()
                log_every = 200 if D == 8 else None
                mse, nparams, hist = train_and_eval(D, dist, cls, Xte, Yte, steps=STEPS, log_every=log_every, map_pooling=map_p)
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
        improve_dens = 100 * (1 - r['density_mse'] / r['standard_mse'])
        improve_mom = 100 * (1 - r['moment_mse'] / r['standard_mse'])
        improve_gma = 100 * (1 - r['gma_mse'] / r['standard_mse'])
        improve_mma = 100 * (1 - r['mma_mse'] / r['standard_mse'])
        print(f"D={r['D']:2d} {r['dist']:10s} | empirical={r['empirical_corr_mse']:.5f} "
              f"predict-mean={r['predict_mean_mse']:.5f} | standard={r['standard_mse']:.5f} "
              f"density={r['density_mse']:.5f} moment={r['moment_mse']:.5f} gma={r['gma_mse']:.5f} mma={r['mma_mse']:.5f}")
        print(f"   density improve={improve_dens:+.1f}% | moment={improve_mom:+.1f}% | gma={improve_gma:+.1f}% | mma={improve_mma:+.1f}%")
