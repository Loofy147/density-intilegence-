import torch
import math
import collections
from models import PSDProjector
from pipeline import triu_to_full, full_to_triu

def spectral_loss_func(A, B):
    e_a, _ = torch.linalg.eigh(A)
    e_b, _ = torch.linalg.eigh(B)
    return torch.mean((e_a - e_b)**2)

def test_spectral_logic():
    print("Testing Spectral Logic...")
    D = 4
    # Identical
    A = torch.eye(D).unsqueeze(0)
    assert spectral_loss_func(A, A) < 1e-7
    # Different
    B = torch.ones((1, D, D)) * 0.1 + torch.eye(D) * 0.9
    loss = spectral_loss_func(A, B)
    assert loss > 0
    print("Spectral Logic Passed.")

def test_psd_projector():
    print("Testing PSD Projector...")
    D = 3
    proj = PSDProjector(D)
    # Non-PSD matrix
    mat = torch.tensor([[1.0, 0.9, 0.9], [0.9, 1.0, -0.9], [0.9, -0.9, 1.0]]).unsqueeze(0)
    e, _ = torch.linalg.eigh(mat)
    assert torch.any(e < 0)

    tri = full_to_triu(mat)
    tri_psd = proj(tri)
    mat_psd = triu_to_full(tri_psd, D)
    e_psd, _ = torch.linalg.eigh(mat_psd)
    assert torch.all(e_psd >= -1e-7)
    # Check diagonal is 1
    diag = torch.diagonal(mat_psd, dim1=-2, dim2=-1)
    assert torch.all(abs(diag - 1.0) < 1e-6)
    print("PSD Projector Passed.")

def simulate_planet(p_owner, p_ships, p_prod, target_eta, events):
    curr_o, curr_s, curr_t = p_owner, float(p_ships), 0
    min_s = curr_s if curr_o != -1 else -curr_s
    for e_eta, e_o, e_s in sorted(events):
        if e_eta > target_eta: break
        dt = e_eta - curr_t
        if curr_o != -1: curr_s += p_prod * dt
        curr_t = e_eta
        if e_o == curr_o: curr_s += e_s
        else:
            if e_s > curr_s + 1e-5: curr_s, curr_o = e_s - curr_s, e_o
            elif abs(e_s - curr_s) <= 1e-5: curr_s, curr_o = 0, -1
            else: curr_s -= e_s
        if curr_o == p_owner and curr_o != -1: min_s = min(min_s, curr_s)
        elif p_owner != -1: min_s = -curr_s
    dt = target_eta - curr_t
    if curr_o != -1: curr_s += p_prod * dt
    return curr_o, curr_s, min_s

def test_arrival_sync():
    print("Testing Arrival Synchronization Simulation...")
    # Target: Neutral, 20 ships
    # Fleet 1: ETA 10, 15 ships (Not enough)
    # Fleet 2: ETA 11, 10 ships (Together they take it)
    p_dat = (-1, 20, 1) # owner, ships, prod
    events = [(10, 0, 15), (11, 0, 10)]

    # At t=10: Neutral 20 vs Ally 15 -> Neutral 5.
    # At t=11: Neutral 5 vs Ally 10 -> Ally 5.
    owner, ships, _ = simulate_planet(-1, 20, 1, 15, events)
    assert owner == 0
    # From t=11 to t=15: Ally prod 1*4=4. Total 5+4=9.
    assert abs(ships - 9.0) < 1e-5
    print("Arrival Sync Passed.")

if __name__ == "__main__":
    test_spectral_logic()
    test_psd_projector()
    test_arrival_sync()
    print("\nALL LOGIC VERIFICATION V102 PASSED.")
