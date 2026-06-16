from kaggle_environments import make

def run():
    env = make("orbit_wars")
    # v79.2 vs 3 Blitz bots
    env.run(["main.py", "blitz_bot.py", "blitz_bot.py", "blitz_bot.py"])
    rewards = [a.reward for a in env.steps[-1]]
    print(f"Results vs 3 Blitz: {rewards}")

if __name__ == "__main__":
    for i in range(5):
        run()
