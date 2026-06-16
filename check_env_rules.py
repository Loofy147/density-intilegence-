from kaggle_environments import make
import math

def agent(obs, config):
    if obs.step == 0:
        p12 = obs.planets[12]
        p0 = obs.planets[0]
        ang = math.atan2(p0[3]-p12[3], p0[2]-p12[2])
        # Neutral P0 has 10-85 ships.
        # Send exact ships if possible.
        return [[p12[0], ang, p0[5]]]
    return []

env = make("orbit_wars")
env.run([agent, "random"])
for i in range(30):
    p0 = env.steps[i][0].observation.planets[0]
    if p0[1] != -1:
        print(f"Step {i}: Planet 0 owned by {p0[1]}, ships {p0[5]}")
        break
