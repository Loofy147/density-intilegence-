import time
from kaggle_environments import make

def run():
    env = make("orbit_wars", debug=True)
    # v79.2 vs SpamBlitz
    start = time.time()
    env.run(["main.py", "spam_blitz.py", "random", "random"])
    end = time.time()
    rewards = [a.reward for a in env.steps[-1]]
    print(f"Results: {rewards}, Total Time: {end-start:.2f}s")

    # Check for timeouts
    for step in env.steps:
        for i, agent_out in enumerate(step):
            if agent_out.status == "TIMEOUT":
                print(f"Player {i} TIMEOUT at step {agent_out.observation.step}")

if __name__ == "__main__":
    run()
