from kaggle_environments import make
import math

def agent(obs, config):
    if obs.step == 0:
        p = obs.planets[12]
        return [[p[0], 0, 2], [p[0], 0.1, 50]]
    if obs.step == 1:
        for f in obs.fleets:
            print(f"Step 1: Fleet {f[0]} ships {f[6]} pos ({f[2]:.4f}, {f[3]:.4f})")
    if obs.step == 2:
        for f in obs.fleets:
            print(f"Step 2: Fleet {f[0]} ships {f[6]} pos ({f[2]:.4f}, {f[3]:.4f})")
    return []

env = make("orbit_wars")
env.run([agent, "random"])
