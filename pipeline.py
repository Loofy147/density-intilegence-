import torch
import numpy as np

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

def get_batch_empirical_corr(x):
    B, T, D = x.shape
    mu = x.mean(dim=1, keepdim=True)
    std = x.std(dim=1, keepdim=True) + 1e-6
    x_norm = (x - mu) / std
    cov = (x_norm.transpose(1, 2) @ x_norm) / (T - 1)
    iu = torch.triu_indices(D, D, offset=1)
    return cov[:, iu[0], iu[1]]

def preprocess_input(x):
    """Standardize input: center and scale."""
    return (x - x.mean(dim=1, keepdim=True)) / (x.std(dim=1, keepdim=True) + 1e-6)
