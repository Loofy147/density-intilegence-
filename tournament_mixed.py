from kaggle_environments import make

def run():
    env = make("orbit_wars")
    # v80.1 vs Blitz vs Spam vs Random
    env.run(["main.py", "blitz_bot.py", "spam_bot.py", "random"])
    rewards = [a.reward for a in env.steps[-1]]
    print(f"Mixed Results: {rewards}")

if __name__ == "__main__":
    for i in range(5):
        run()
