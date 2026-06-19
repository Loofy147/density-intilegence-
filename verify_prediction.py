import math

def get_predicted_pos(p_dat, eta, obs):
    # p_dat: [id, owner, x, y, ..., prod]
    t_id, tx, ty = p_dat[0], p_dat[2], p_dat[3]
    av = obs.get("angular_velocity", 0.0)
    CX, CY = 50.0, 50.0
    d = math.hypot(tx-CX, ty-CY)
    if d < 1e-6 or d > 55.0: return (tx, ty)
    cur_ang = math.atan2(ty-CY, tx-CX)
    new_ang = cur_ang + av * eta
    return (CX + d * math.cos(new_ang), CY + d * math.sin(new_ang))

def test_prediction():
    print("Testing Prediction Logic...")
    obs = {"angular_velocity": 0.1}
    # Planet at (60, 50) -> distance 10, angle 0
    p_dat = [1, -1, 60.0, 50.0]
    # In eta=pi/0.2 steps, it should rotate pi/2 (90 deg)
    # wait, eta=5*pi -> angle = 0.5*pi = 90 deg. Pos should be (50, 60)
    eta = (math.pi / 2) / 0.1
    nx, ny = get_predicted_pos(p_dat, eta, obs)
    print(f"  ETA={eta:.2f}, Pos=({nx:.2f}, {ny:.2f})")
    assert abs(nx - 50.0) < 1e-5
    assert abs(ny - 60.0) < 1e-5
    print("Prediction Logic Passed.")

if __name__ == "__main__":
    test_prediction()
