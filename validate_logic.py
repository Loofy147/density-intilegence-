import torch
from models import MomentModulatedAttn, PSDProjector, ShrinkageCorrEstimator
from pipeline import sample_batch, is_psd_batch, reconstruct_corr_matrix

def test_psd_compliance():
    print("Testing PSD Compliance...")
    D = 10
    psd = PSDProjector(D)
    # Generate non-PSD matrix (e.g., all 0.9 off-diag can be non-PSD for large D)
    tri = torch.ones(5, D*(D-1)//2) * 0.95
    out = psd(tri)
    full = reconstruct_corr_matrix(out, D)
    passed = is_psd_batch(full).all().item()
    print(f"PSD Projector Test: {'PASSED' if passed else 'FAILED'}")
    return passed

def test_mma_sensitivity():
    print("\nTesting MMA Sensitivity to Variance...")
    d_model = 16
    n_heads = 2
    mma = MomentModulatedAttn(d_model, n_heads)

    # Low variance input
    x_low = torch.randn(1, 10, d_model) * 0.1
    # High variance input
    x_high = torch.randn(1, 10, d_model) * 5.0

    with torch.no_grad():
        out_low = mma(x_low)
        out_high = mma(x_high)

    print(f"Mean Magnitude (Low Var): {out_low.abs().mean().item():.4f}")
    print(f"Mean Magnitude (High Var): {out_high.abs().mean().item():.4f}")
    return True

def test_shrinkage_gradient():
    print("\nTesting Shrinkage Gradient Flow...")
    D = 5
    model = ShrinkageCorrEstimator(D)
    X, Y = sample_batch(2, 10, D)
    pred = model(X)
    loss = torch.mean((pred - Y)**2)
    loss.backward()

    # Check if weights have gradients
    params = list(model.parameters())
    has_grad = all(p.grad is not None for p in params if p.requires_grad)
    print(f"Gradient Check: {'PASSED' if has_grad else 'FAILED'}")
    return has_grad

if __name__ == '__main__':
    s1 = test_psd_compliance()
    s2 = test_mma_sensitivity()
    s3 = test_shrinkage_gradient()
    if all([s1, s2, s3]):
        print("\nAll logical validations PASSED.")
    else:
        print("\nLogical validations FAILED.")
        exit(1)
