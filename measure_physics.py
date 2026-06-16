from kaggle_environments import make
import math

def agent(obs, config):
    if obs.step == 0:
        p = obs.planets[12]
        # Launch 10 ships straight right
        return [[p[0], 0, 10]]
    return []

env = make("orbit_wars")
env.run([agent, "random"])

for i in range(1, 10):
    obs_prev = env.steps[i-1][0].observation
    obs_curr = env.steps[i][0].observation
    if obs_curr.fleets:
        f = obs_curr.fleets[0]
        f_prev = obs_prev.fleets[0] if obs_prev.fleets else None
        if f_prev:
            dx = f[2] - f_prev[2]
            dy = f[3] - f_prev[3]
            dist = math.hypot(dx, dy)
            print(f"Step {i}: Fleet ships {f[6]} moved distance {dist:.4f}")

    p = obs_curr.planets[0]
    p_prev = obs_prev.planets[0]
    a = math.atan2(p[3]-50, p[2]-50)
    a_prev = math.atan2(p_prev[3]-50, p_prev[2]-50)
    da = (a - a_prev + math.pi) % (2*math.pi) - math.pi
    if abs(da) > 1e-6:
        print(f"Step {i}: Planet 0 moved da {da:.6f}, AV {obs_curr.angular_velocity:.6f}, ratio {da/obs_curr.angular_velocity:.4f}")
