from kaggle_environments import make
env = make("orbit_wars", debug=True)
env.run(["main.py", "blitz_bot.py", "random", "random"])
for i in range(10):
    print(f"Step {i}:")
    obs = env.steps[i][0].observation
    for p in obs.planets:
        if p[1] != -1:
            print(f"  P{p[0]} (Owner {p[1]}): ships={p[5]}")
    for f in obs.get("fleets", []):
        print(f"  Fleet from {f[1]} to ?? (ships={f[6]})")
