from kaggle_environments import make
env = make("orbit_wars")
obs = env.reset()[0].observation
print(f"Angular Velocity: {obs.get('angular_velocity')}")
planets = obs.planets
for p in planets[:5]:
    print(f"P{p[0]}: dist={((p[2]-50)**2+(p[3]-50)**2)**0.5:.2f}, prod={p[6]}")
