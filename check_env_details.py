from kaggle_environments import make
import math

def agent(obs, config):
    if obs.step == 0:
        # Check configuration
        print("Config:", config)
        # Check comets
        if obs.comets:
            print("Comet sample:", obs.comets[0].keys())
            print("Path length:", len(obs.comets[0]['paths'][0]))
            print("Path index:", obs.comets[0]['path_index'])
        # Launch two fleets
        p = obs.planets[12]
        return [[p[0], 0, 2], [p[0], 0.2, 50]]

    if obs.step == 1:
        for f in obs.fleets:
            dist = math.hypot(f[2]-obs.planets[f[5]][2], f[3]-obs.planets[f[5]][3])
            print(f"Fleet {f[0]} ships {f[6]} dist from start {dist:.4f}")
    return []

env = make("orbit_wars")
env.run([agent, "random"])
