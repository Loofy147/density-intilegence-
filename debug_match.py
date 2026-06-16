from kaggle_environments import make
import json

env = make("orbit_wars", debug=True)
env.run(["main.py", "blitz_bot.py", "spam_bot.py", "random"])

with open("match_debug.json", "w") as f:
    # Save the whole thing to inspect if needed, but let's print stats
    pass

for i, agent_out in enumerate(env.steps[-1]):
    print(f"Player {i} Status: {agent_out.status}, Reward: {agent_out.reward}")
    if agent_out.status != "DONE":
        print(f"  Log: {agent_out.log}")

# Analyze step 0-20
for s in range(20):
    step = env.steps[s]
    print(f"\nStep {s}:")
    for p_idx in range(4):
        action = step[p_idx].action
        if action:
            print(f"  P{p_idx} Action: {action[:2]} ... (total {len(action)})")

    # Ownership count
    obs = step[0].observation
    owners = [p[1] for p in obs.planets]
    from collections import Counter
    print(f"  Owners: {Counter(owners)}")
