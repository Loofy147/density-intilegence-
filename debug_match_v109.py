from kaggle_environments import make
import collections

def agent_v109(obs, config):
    # Use the code from main.py
    from main import agent
    return agent(obs, config)

env = make("orbit_wars", debug=True)
env.run(["main.py", "blitz_bot.py", "spam_bot.py", "random"])

for i, agent_out in enumerate(env.steps[-1]):
    print(f"P{i} Reward: {agent_out.reward}, Status: {agent_out.status}")
    if agent_out.status == "ERROR":
        print(f"  P{i} Error: {agent_out.observation.get('error', 'Unknown')}")

# Ownership history
history = collections.defaultdict(list)
for step in env.steps:
    owners = [p[1] for p in step[0].observation.planets]
    counts = collections.Counter(owners)
    for p_idx in range(-1, 4):
        history[p_idx].append(counts[p_idx])

print("\nPlanet ownership over time (every 50 steps):")
for s in range(0, len(env.steps), 50):
    counts = {p_idx: history[p_idx][s] for p_idx in range(-1, 4)}
    print(f"Step {s:3d}: {counts}")
