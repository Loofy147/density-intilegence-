import torch
import numpy as np

def make_corr_batch(B, D, K=2, device='cpu'):
    """Generates a batch of random correlation matrices using a factor model."""
    L = (torch.rand(B, D, K, device=device) * 0.5 + 0.4) * (torch.randint(0, 2, (B, D, K), device=device) * 2 - 1).float()
    cov = L @ L.transpose(-2, -1)
    # Add idiosyncratic noise to ensure PD
    idio = torch.rand(B, D, device=device) * 0.3 + 0.2
    cov = cov + torch.diag_embed(idio)

    # Normalize to correlation matrix
    d = torch.sqrt(torch.diagonal(cov, dim1=-2, dim2=-1))
    corr = cov / (d.unsqueeze(-1) * d.unsqueeze(-2))
    return corr

def sample_batch(B, T, D, dist='gaussian', dof=4, K=2, device='cpu'):
    """Samples data from a multivariate distribution with a random correlation structure."""
    corr = make_corr_batch(B, D, K, device=device)
    # Ensure it is definitely PD for Cholesky
    Lc = torch.linalg.cholesky(corr + torch.eye(D, device=device) * 1e-6)

    if dist == 'gaussian':
        z = torch.randn(B, T, D, device=device)
    elif dist == 'student_t':
        g = torch.distributions.Gamma(dof/2, 0.5).sample((B, T, 1)).to(device)
        z = torch.randn(B, T, D, device=device) * torch.sqrt(dof / g)
    else:
        raise ValueError(f"Unknown distribution: {dist}")

    x = torch.einsum('bij,btj->bti', Lc, z)

    iu = torch.triu_indices(D, D, offset=1, device=device)
    Y = corr[:, iu[0], iu[1]]
    return x, Y

def empirical_corr_mse(X, Y):
    """Calculates MSE between empirical correlation and ground truth."""
    B, T, D = X.shape
    device = X.device

    # Standardize X
    mu = X.mean(dim=1, keepdim=True)
    std = X.std(dim=1, keepdim=True) + 1e-6
    Xn = (X - mu) / std

    # Batch covariance of normalized data
    emp_corr_full = (Xn.transpose(1, 2) @ Xn) / (T - 1)

    iu = torch.triu_indices(D, D, offset=1, device=device)
    emp_tri = emp_corr_full[:, iu[0], iu[1]]

    return torch.mean((emp_tri - Y)**2).item()

def get_batch_empirical_corr(x):
    """Computes the upper triangular part of the empirical correlation matrix."""
    B, T, D = x.shape
    mu = x.mean(dim=1, keepdim=True)
    std = x.std(dim=1, keepdim=True) + 1e-6
    x_norm = (x - mu) / std
    cov = (x_norm.transpose(1, 2) @ x_norm) / (T - 1)
    iu = torch.triu_indices(D, D, offset=1, device=x.device)
    return cov[:, iu[0], iu[1]]

def preprocess_input(x):
    """Standardize input: center and scale across the time dimension."""
    return (x - x.mean(dim=1, keepdim=True)) / (x.std(dim=1, keepdim=True) + 1e-6)

def reconstruct_corr_matrix(tri_v, D):
    """Reconstruct a symmetric correlation matrix from its upper triangular part."""
    B = tri_v.shape[0]
    device = tri_v.device
    mat = torch.eye(D, device=device).unsqueeze(0).repeat(B, 1, 1)
    iu = torch.triu_indices(D, D, offset=1, device=device)
    mat[:, iu[0], iu[1]] = tri_v
    mat[:, iu[1], iu[0]] = tri_v
    return mat

def is_psd_batch(matrices, tol=1e-6):
    """Check if a batch of matrices is Positive Semi-Definite."""
    e, _ = torch.linalg.eigh(matrices)
    return torch.all(e >= -tol, dim=-1)

def full_to_triu(mat):
    """Extract upper triangular part (excluding diagonal) from a symmetric matrix."""
    D = mat.shape[-1]
    iu = torch.triu_indices(D, D, offset=1, device=mat.device)
    return mat[:, iu[0], iu[1]]

def triu_to_full(tri_v, D):
    """Reconstruct a symmetric matrix from its upper triangular part."""
    return reconstruct_corr_matrix(tri_v, D)
