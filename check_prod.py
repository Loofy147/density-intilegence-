from kaggle_environments import make
def agent(obs, config):
    if obs.step < 10:
        p = obs.planets[12]
        print(f"Step {obs.step}: Planet 12 (Owner {p[1]}) ships {p[5]}")
    return []
env = make("orbit_wars")
env.run([agent, "random"])
