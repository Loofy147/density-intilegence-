from kaggle_environments import make
import math

def agent(obs, config):
    if obs.step < 5:
        p0 = obs.planets[0]
        ang = math.atan2(p0[3]-50, p0[2]-50)
        print(f"Step {obs.step}: Planet 0 at ({p0[2]:.2f}, {p0[3]:.2f}), Angle {ang:.4f}, AV {obs.angular_velocity:.6f}")
    return []

env = make("orbit_wars")
env.run([agent, "random", "random", "random"])
