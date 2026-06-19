import torch
import torch.nn as nn
from pipeline import get_batch_empirical_corr, preprocess_input, triu_to_full, full_to_triu

class PSDProjector(nn.Module):
    def __init__(self, D, min_eig=1e-4):
        super().__init__()
        self.D = D
        self.min_eig = min_eig
    def forward(self, tri_v):
        mat = triu_to_full(tri_v, self.D)
        e, v = torch.linalg.eigh(mat)
        if torch.all(e >= self.min_eig): return tri_v
        e = torch.clamp(e, min=self.min_eig)
        mat_psd = v @ torch.diag_embed(e) @ v.transpose(-2, -1)
        diag = torch.diagonal(mat_psd, dim1=-2, dim2=-1)
        d_inv = 1.0 / torch.sqrt(torch.clamp(diag, min=1e-12))
        mat_psd = mat_psd * d_inv.unsqueeze(-1) * d_inv.unsqueeze(-2)
        return full_to_triu(mat_psd)

class GroupedMomentAttn(nn.Module):
    def __init__(self, d_model, n_heads, n_groups=2):
        super().__init__()
        self.h, self.g, self.hd = n_heads, n_groups, d_model // n_heads
        self.q_proj = nn.Linear(d_model, n_groups * self.hd)
        self.k_proj = nn.Linear(d_model, n_groups * self.hd)
        self.v_proj = nn.Linear(d_model, d_model)
        self.gate = nn.Sequential(nn.Linear(d_model, d_model), nn.Sigmoid())
        self.proj = nn.Linear(d_model, d_model)
    def forward(self, x):
        B, T, _ = x.shape
        G, H, hd = self.g, self.h, self.hd
        q = self.q_proj(x).view(B, T, G, hd).transpose(1, 2)
        k = self.k_proj(x).view(B, T, G, hd).transpose(1, 2)
        v = self.v_proj(x).view(B, T, H, hd).transpose(1, 2)
        attn = (q @ k.transpose(-2, -1)) / (hd ** 0.5)
        attn = attn.softmax(-1).repeat_interleave(H // G, dim=1)
        m_o = attn @ v
        v_o = torch.clamp((attn @ (v**2)) - (m_o**2), min=1e-10)
        res = (m_o.transpose(1, 2).reshape(B, T, -1)) * self.gate(v_o.transpose(1, 2).reshape(B, T, -1))
        return self.proj(res)

class Block(nn.Module):
    def __init__(self, d_model, n_heads, n_groups=2):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = GroupedMomentAttn(d_model, n_heads, n_groups)
        self.ln2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(nn.Linear(d_model, 4*d_model), nn.GELU(), nn.Linear(4*d_model, d_model))
    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x

class SpectralRecursiveGMAEstimator(nn.Module):
    def __init__(self, D, d_model=128, n_heads=8, n_layers=6, n_groups=2):
        super().__init__()
        self.D = D
        self.embed = nn.Linear(D, d_model)
        self.blocks = nn.ModuleList([Block(d_model, n_heads, n_groups) for _ in range(n_layers)])
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Sequential(nn.Linear(2*d_model, d_model), nn.GELU(), nn.Linear(d_model, 1 + D*(D-1)//2))
        self.psd_proj = PSDProjector(D)
    def forward(self, x):
        x_n = preprocess_input(x)
        rho_emp = get_batch_empirical_corr(x_n)
        h = self.embed(x_n)
        for blk in self.blocks: h = blk(h)
        h = self.ln_f(h)
        pool = torch.cat([h.mean(1), h.var(1)], -1)
        out = self.head(pool)
        alpha, rho_pred = torch.sigmoid(out[:, :1]), torch.tanh(out[:, 1:])
        return self.psd_proj((1-alpha)*rho_emp + alpha*rho_pred)
