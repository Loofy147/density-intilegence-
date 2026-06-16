from kaggle_environments import make
import math

def agent(obs, config):
    if obs.step == 0:
        p = obs.planets[12]
        # Send 1 ship and 50 ships
        return [[p[0], 0, 1], [p[0], 0.2, 50]]
    if obs.step == 1:
        for f in obs.fleets:
            p_start = obs.planets[f[5]]
            dist = math.hypot(f[2]-p_start[2], f[3]-p_start[3])
            print(f"Step 1: Fleet {f[0]} ships {f[6]} moved {dist:.4f}")
    return []

env = make("orbit_wars")
env.run([agent, "random"])
