import torch
import torch.nn as nn
from pipeline import get_batch_empirical_corr, preprocess_input

def triu_to_full(tri_v, D):
    """Reconstruct a symmetric matrix from its upper triangular part (excluding diagonal)."""
    B = tri_v.shape[0]
    mat = torch.eye(D, device=tri_v.device).unsqueeze(0).repeat(B, 1, 1)
    iu = torch.triu_indices(D, D, offset=1, device=tri_v.device)
    mat[:, iu[0], iu[1]] = tri_v
    mat[:, iu[1], iu[0]] = tri_v
    return mat

def full_to_triu(mat):
    """Extract upper triangular part (excluding diagonal) from a symmetric matrix."""
    D = mat.shape[-1]
    iu = torch.triu_indices(D, D, offset=1, device=mat.device)
    return mat[:, iu[0], iu[1]]

class PSDProjector(nn.Module):
    """Ensures a correlation matrix is PSD by clipping eigenvalues or using shrinkage."""
    def __init__(self, D, min_eig=1e-6):
        super().__init__()
        self.D = D
        self.min_eig = min_eig
    def forward(self, tri_v):
        B = tri_v.shape[0]
        mat = triu_to_full(tri_v, self.D)
        # Use eigh for symmetric matrices
        e, v = torch.linalg.eigh(mat)
        e = torch.clamp(e, min=self.min_eig)
        # Reconstruct
        mat_psd = v @ torch.diag_embed(e) @ v.transpose(-2, -1)
        # Ensure it's a correlation matrix (diagonal = 1)
        d = torch.sqrt(torch.diagonal(mat_psd, dim1=-2, dim2=-1))
        mat_psd = mat_psd / (d.unsqueeze(-1) * d.unsqueeze(-2))
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
        a = (q @ k.transpose(-2, -1)) / self.hd ** 0.5
        a = a.softmax(-1)
        o = (a @ v).permute(0, 2, 1, 3).reshape(B, T, -1)
        return self.proj(o)

class MomentModulatedAttn(nn.Module):
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
        modulated = mean_o * self.gate(var_o)
        return self.proj(modulated)

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
        a = (q @ k.transpose(-2, -1)) / hd ** 0.5
        a = a.softmax(-1)

        # Mean for all heads
        mean_o = a @ v

        # Variance for groups (using first G heads as templates)
        a_g = a[:, :G]
        v_g = v[:, :G]
        mean_g = mean_o[:, :G]
        mean_sq_g = a_g @ (v_g**2)
        var_g = torch.relu(mean_sq_g - mean_g**2)

        # Broadcast group variance back to all heads using expand
        var_o = var_g.unsqueeze(2).expand(-1, -1, H // G, -1, -1).reshape(B, H, T, hd)

        mean_o = mean_o.permute(0, 2, 1, 3).reshape(B, T, -1)
        var_o = var_o.permute(0, 2, 1, 3).reshape(B, T, -1)
        modulated = mean_o * self.gate(var_o)
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

class AnchoredCorrEstimator(nn.Module):
    def __init__(self, D, d_model=64, n_heads=8, n_layers=2, attn_cls=MomentModulatedAttn, **kwargs):
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
        z_emp = torch.atanh(torch.clamp(rho_emp, -0.999, 0.999))
        out = torch.tanh(z_emp + delta)
        return self.psd_proj(out)

class ShrinkageCorrEstimator(nn.Module):
    def __init__(self, D, d_model=64, n_heads=8, n_layers=2, attn_cls=MomentModulatedAttn, **kwargs):
        super().__init__()
        self.D = D
        self.embed = nn.Linear(D, d_model)
        self.blocks = nn.ModuleList([Block(d_model, n_heads, attn_cls, **kwargs) for _ in range(n_layers)])
        self.ln_f = nn.LayerNorm(d_model)
        self.alpha_head = nn.Sequential(nn.Linear(2 * d_model, d_model), nn.GELU(), nn.Linear(d_model, 1), nn.Sigmoid())
    def forward(self, x):
        rho_emp = get_batch_empirical_corr(x)
        x = preprocess_input(x)
        h = self.embed(x)
        for blk in self.blocks:
            h = blk(h)
        h = self.ln_f(h)
        pool = torch.cat([h.mean(dim=1), h.var(dim=1)], dim=-1)
        alpha = self.alpha_head(pool)
        # Shrinkage towards Identity: (1-alpha)*R_emp + alpha*I
        # For off-diagonals, this is (1-alpha)*rho_emp + alpha*0 = (1-alpha)*rho_emp
        return (1 - alpha) * rho_emp
