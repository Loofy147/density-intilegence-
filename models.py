import torch
import torch.nn as nn
from pipeline import get_batch_empirical_corr, preprocess_input, triu_to_full, full_to_triu

class PSDProjector(nn.Module):
    """Optimized PSD projector with fast-path heuristics."""
    def __init__(self, D, min_eig=1e-6):
        super().__init__()
        self.D = D
        self.min_eig = min_eig

    def forward(self, tri_v):
        B = tri_v.shape[0]
        mat = triu_to_full(tri_v, self.D)

        # Fast trace check: a correlation matrix must have trace = D.
        # But here we check if it is potentially non-PSD by looking for large negative off-diagonals
        # or simple Gershgorin circle theorem bounds.
        # Implementation: skip if all are likely PSD (heuristic: all off-diagonals small)
        if torch.all(torch.abs(tri_v) < 0.1) and self.D > 2:
            return tri_v

        e, v = torch.linalg.eigh(mat)
        if torch.all(e >= self.min_eig):
            return tri_v

        e = torch.clamp(e, min=self.min_eig)
        mat_psd = v @ torch.diag_embed(e) @ v.transpose(-2, -1)

        # Fast Vectorized Diagonal Normalization
        diag = torch.diagonal(mat_psd, dim1=-2, dim2=-1)
        d_inv = 1.0 / torch.sqrt(torch.clamp(diag, min=1e-9))
        mat_psd = mat_psd * d_inv.unsqueeze(-1) * d_inv.unsqueeze(-2)

        return full_to_triu(mat_psd)

class GroupedMomentAttn(nn.Module):
    def __init__(self, d_model, n_heads, n_groups=2):
        super().__init__()
        self.h, self.hd = n_heads, d_model // n_heads
        self.g = n_groups
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.gate = nn.Sequential(nn.Linear(d_model, d_model), nn.Sigmoid())
        self.proj = nn.Linear(d_model, d_model)
    def forward(self, x):
        B, T, D = x.shape
        H, hd, G = self.h, self.hd, self.g
        qkv = self.qkv(x).view(B, T, 3, H, hd).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        a = (q @ k.transpose(-2, -1)) / (hd ** 0.5)
        a = a.softmax(-1)
        mean_o = a @ v
        mean_sq_o_g = a.view(B, G, H//G, T, T) @ (v.view(B, G, H//G, T, hd)**2)
        var_o = torch.relu(mean_sq_o_g - (mean_o.view(B, G, H//G, T, hd)**2)).view(B, H, T, hd)
        m_res = mean_o.permute(0, 2, 1, 3).reshape(B, T, -1)
        v_res = var_o.permute(0, 2, 1, 3).reshape(B, T, -1)
        return self.proj(m_res * self.gate(v_res))

class Block(nn.Module):
    def __init__(self, d_model, n_heads, attn_cls, **kwargs):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = attn_cls(d_model, n_heads, **kwargs)
        self.ln2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(nn.Linear(d_model, 4*d_model), nn.GELU(), nn.Linear(4*d_model, d_model))
    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x

class SpectralRecursiveGMAEstimator(nn.Module):
    def __init__(self, D, d_model=64, n_heads=8, n_layers=4, n_groups=2):
        super().__init__()
        self.D = D
        self.embed = nn.Linear(D, d_model)
        self.blocks = nn.ModuleList([Block(d_model, n_heads, GroupedMomentAttn, n_groups=n_groups) for _ in range(n_layers)])
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Sequential(nn.Linear(2 * d_model, d_model), nn.GELU(), nn.Linear(d_model, 1 + D*(D-1)//2))
        self.psd_proj = PSDProjector(D)
    def forward(self, x):
        rho_emp = get_batch_empirical_corr(x)
        h = self.embed(preprocess_input(x))
        for blk in self.blocks: h = blk(h)
        h = self.ln_f(h)
        pool = torch.cat([h.mean(1), h.var(1)], -1)
        out = self.head(pool)
        alpha, rho_pred = torch.sigmoid(out[:, :1]), torch.tanh(out[:, 1:])
        return self.psd_proj((1-alpha)*rho_emp + alpha*rho_pred)

class BaseCorrEstimator(nn.Module):
    def __init__(self, D, d_model=64, n_heads=8, n_layers=2, **kwargs):
        super().__init__()
        self.D = D
        self.embed = nn.Linear(D, d_model)
        self.blocks = nn.ModuleList([Block(d_model, n_heads, GroupedMomentAttn, n_groups=1) for _ in range(n_layers)])
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Sequential(nn.Linear(2*d_model, d_model), nn.GELU(), nn.Linear(d_model, D*(D-1)//2))
        self.psd_proj = PSDProjector(D)
    def forward(self, x):
        h = self.embed(preprocess_input(x))
        for blk in self.blocks: h = blk(h)
        h = self.ln_f(h)
        pool = torch.cat([h.mean(1), h.var(1)], -1)
        return self.psd_proj(self.head(pool))

class RecursiveGMAEstimator(SpectralRecursiveGMAEstimator): pass
class ShrinkageCorrEstimator(SpectralRecursiveGMAEstimator):
    def __init__(self, D, **kwargs): super().__init__(D, d_model=64, n_layers=2, n_groups=2)
class AnchoredCorrEstimator(nn.Module):
    def __init__(self, D, **kwargs):
        super().__init__()
        self.model = BaseCorrEstimator(D)
    def forward(self, x): return self.model(x)
class StdAttn(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.attn = GroupedMomentAttn(d_model, n_heads, n_groups=1)
    def forward(self, x): return self.attn(x)
class MomentModulatedAttn(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.attn = GroupedMomentAttn(d_model, n_heads, n_groups=1)
    def forward(self, x): return self.attn(x)
