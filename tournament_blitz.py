from kaggle_environments import make

def run():
    env = make("orbit_wars")
    # v79.2 vs Blitz
    env.run(["main.py", "blitz_bot.py", "random", "random"])
    rewards = [a.reward for a in env.steps[-1]]
    print(f"v79.2 vs Blitz Results: {rewards}")

if __name__ == "__main__":
    run()
