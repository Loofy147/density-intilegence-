from kaggle_environments import make
import json

env = make("orbit_wars", debug=True)
env.run(["main.py", "random"])
print("Rewards:", [a.reward for a in env.steps[-1]])

# Check for errors in logs
for i, agent_out in enumerate(env.steps[-1]):
    if agent_out.status == "ERROR":
        print(f"Player {i} ERROR: {agent_out.observation.get('error', 'Unknown')}")
        print("Logs:", agent_out.log)

# Check first few steps of Player 0
for s in range(min(20, len(env.steps))):
    action = env.steps[s][0].action
    if action:
        print(f"Step {s} Action: {action}")
