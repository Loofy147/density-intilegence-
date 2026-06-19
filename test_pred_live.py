from kaggle_environments import make
import math

def get_predicted_pos(p_dat, eta, av):
    t_id, tx, ty = p_dat[0], p_dat[2], p_dat[3]
    CX, CY = 50.0, 50.0
    d = math.hypot(tx-CX, ty-CY)
    if d > 55.0: return (tx, ty)
    cur_ang = math.atan2(ty-CY, tx-CX)
    return (CX + d * math.cos(cur_ang + av*eta), CY + d * math.sin(cur_ang + av*eta))

env = make("orbit_wars")
env.reset()
obs = env.steps[0][0].observation
av = obs.angular_velocity
p0 = obs.planets[0]
print(f"Step 0: P0 at ({p0[2]:.2f}, {p0[3]:.2f})")

# Predict at step 10
pred10 = get_predicted_pos(p0, 10, av)
print(f"Pred Step 10: ({pred10[0]:.2f}, {pred10[1]:.2f})")

env.step([[], [], [], []]) # Move to step 1
# ... wait we need to step 10 times
for _ in range(9): env.step([[], [], [], []])

obs10 = env.steps[10][0].observation
p0_10 = obs10.planets[0]
print(f"Actual Step 10: ({p0_10[2]:.2f}, {p0_10[3]:.2f})")
