from kaggle_environments import make
import math

def agent(obs, config):
    # Just a dummy
    return []

env = make("orbit_wars")
env.run([agent, "random"])
for i in range(10):
    obs = env.steps[i][0].observation
    print(f"Step {i}: AV={obs.get('angular_velocity', 'N/A')}")
    p = obs.planets[12] # Player 0 starting planet (usually)
    print(f"  P12: ({p[2]:.2f}, {p[3]:.2f})")
