from kaggle_environments import make
import json

env = make("orbit_wars", debug=True)
env.run(["main.py", "blitz_bot.py", "random", "random"])

with open("alpha_match.json", "w") as f:
    json.dump(env.toJSON(), f)

# Print final status
print("Final Rewards:", [s.reward for s in env.steps[-1]])
print("Final Step:", len(env.steps))

# Check step 20
s20 = env.steps[20]
obs = s20[0].observation
my_planets = [p for p in obs.planets if p[1] == 0]
en_planets = [p for p in obs.planets if p[1] == 1]
print(f"Step 20: Me={len(my_planets)} planets, Enemy={len(en_planets)} planets")
