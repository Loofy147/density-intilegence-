import torch
import torch.nn as nn
from pipeline import get_batch_empirical_corr, preprocess_input, triu_to_full, full_to_triu

class PSDProjector(nn.Module):
    """Ensures a correlation matrix is PSD by clipping eigenvalues."""
    def __init__(self, D, min_eig=1e-6):
        super().__init__()
        self.D = D
        self.min_eig = min_eig
    def forward(self, tri_v):
        B = tri_v.shape[0]
        mat = triu_to_full(tri_v, self.D)

        # Eigen-decomposition
        e, v = torch.linalg.eigh(mat)

        # Batch fast-path check
        if torch.all(e >= self.min_eig):
            return tri_v

        e_clamped = torch.clamp(e, min=self.min_eig)
        mat_psd = v @ torch.diag_embed(e_clamped) @ v.transpose(-2, -1)
        diag = torch.diagonal(mat_psd, dim1=-2, dim2=-1)
        d_inv = 1.0 / torch.sqrt(torch.clamp(diag, min=1e-9))
        mat_psd = mat_psd * d_inv.unsqueeze(-1) * d_inv.unsqueeze(-2)
        return full_to_triu(mat_psd)

class StdAttn(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.h, self.hd = n_heads, d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)
    def forward(self, x):
        B, T, D = x.shape
        qkv = self.qkv(x).view(B, T, 3, self.h, self.hd).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        a = (q @ k.transpose(-2, -1)) / (self.hd ** 0.5)
        a = a.softmax(-1)
        o = (a @ v).permute(0, 2, 1, 3).reshape(B, T, -1)
        return self.proj(o)

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
        mean_o_g = mean_o.view(B, G, H // G, T, hd)
        v_g = v.view(B, G, H // G, T, hd)
        a_g = a.view(B, G, H // G, T, T)

        mean_sq_o_g = a_g @ (v_g**2)
        var_o_g = torch.relu(mean_sq_o_g - mean_o_g**2)
        var_o = var_o_g.view(B, H, T, hd)

        mean_o_res = mean_o.permute(0, 2, 1, 3).reshape(B, T, -1)
        var_o_res = var_o.permute(0, 2, 1, 3).reshape(B, T, -1)
        modulated = mean_o_res * self.gate(var_o_res)
        return self.proj(modulated)

class Block(nn.Module):
    def __init__(self, d_model, n_heads, attn_cls, ff_mult=4, **kwargs):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = attn_cls(d_model, n_heads, **kwargs)
        self.ln2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(nn.Linear(d_model, ff_mult * d_model), nn.GELU(), nn.Linear(ff_mult * d_model, d_model))
    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x

class BaseCorrEstimator(nn.Module):
    def __init__(self, D, d_model=64, n_heads=8, n_layers=2, attn_cls=StdAttn, map_pooling=True, **kwargs):
        super().__init__()
        self.D = D
        self.map_pooling = map_pooling
        self.embed = nn.Linear(D, d_model)
        self.blocks = nn.ModuleList([Block(d_model, n_heads, attn_cls, **kwargs) for _ in range(n_layers)])
        self.ln_f = nn.LayerNorm(d_model)
        out_dim = D * (D - 1) // 2
        head_in = 2 * d_model if map_pooling else d_model
        self.head = nn.Sequential(nn.Linear(head_in, d_model), nn.GELU(), nn.Linear(d_model, out_dim))
        self.psd_proj = PSDProjector(D)
    def forward(self, x):
        x = preprocess_input(x)
        h = self.embed(x)
        for blk in self.blocks:
            h = blk(h)
        h = self.ln_f(h)
        if self.map_pooling:
            pool = torch.cat([h.mean(dim=1), h.var(dim=1)], dim=-1)
        else:
            pool = h.mean(dim=1)
        out = self.head(pool)
        return self.psd_proj(out)

class ShrinkageCorrEstimator(nn.Module):
    def __init__(self, D, d_model=64, n_heads=8, n_layers=2, attn_cls=StdAttn, **kwargs):
        super().__init__()
        self.D = D
        self.embed = nn.Linear(D, d_model)
        self.blocks = nn.ModuleList([Block(d_model, n_heads, attn_cls, **kwargs) for _ in range(n_layers)])
        self.ln_f = nn.LayerNorm(d_model)
        out_dim = D * (D - 1) // 2
        self.head = nn.Sequential(nn.Linear(2 * d_model, d_model), nn.GELU(), nn.Linear(d_model, 1 + out_dim))
        self.psd_proj = PSDProjector(D)
    def forward(self, x):
        rho_emp = get_batch_empirical_corr(x)
        x = preprocess_input(x)
        h = self.embed(x)
        for blk in self.blocks:
            h = blk(h)
        h = self.ln_f(h)
        pool = torch.cat([h.mean(dim=1), h.var(dim=1)], dim=-1)
        out = self.head(pool)
        alpha = torch.sigmoid(out[:, :1])
        rho_pred = torch.tanh(out[:, 1:])
        rho_final = (1 - alpha) * rho_emp + alpha * rho_pred
        return self.psd_proj(rho_final)

class GMA_ShrinkageCorrEstimator(ShrinkageCorrEstimator):
    def __init__(self, D, d_model=64, n_heads=8, n_layers=2, n_groups=2, **kwargs):
        super().__init__(D, d_model=d_model, n_heads=n_heads, n_layers=n_layers,
                         attn_cls=GroupedMomentAttn, n_groups=n_groups, **kwargs)

class AnchoredCorrEstimator(nn.Module):
    """Stub to keep density_attn.py happy, or implement properly if needed."""
    def __init__(self, D, d_model=64, n_heads=8, n_layers=2, attn_cls=StdAttn, **kwargs):
        super().__init__()
        self.D = D
        self.embed = nn.Linear(D, d_model)
        self.blocks = nn.ModuleList([Block(d_model, n_heads, attn_cls, **kwargs) for _ in range(n_layers)])
        self.ln_f = nn.LayerNorm(d_model)
        out_dim = D * (D - 1) // 2
        self.head = nn.Sequential(nn.Linear(2 * d_model, d_model), nn.GELU(), nn.Linear(d_model, out_dim))
        self.psd_proj = PSDProjector(D)
    def forward(self, x):
        rho_emp = get_batch_empirical_corr(x)
        x = preprocess_input(x)
        h = self.embed(x)
        for blk in self.blocks:
            h = blk(h)
        h = self.ln_f(h)
        pool = torch.cat([h.mean(dim=1), h.var(dim=1)], dim=-1)
        delta = self.head(pool)
        z_emp = torch.atanh(torch.clamp(rho_emp, -0.99, 0.99))
        out = torch.tanh(z_emp + delta)
        return self.psd_proj(out)

class MomentModulatedAttn(nn.Module):
    """Compatibility stub."""
    def __init__(self, d_model, n_heads, **kwargs):
        super().__init__()
        self.attn = GroupedMomentAttn(d_model, n_heads, n_groups=1)
    def forward(self, x):
        return self.attn(x)
