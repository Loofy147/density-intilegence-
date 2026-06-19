from kaggle_environments import make
import collections

def run(n=10):
    stats = collections.Counter()
    for i in range(n):
        env = make("orbit_wars")
        # v122 (in main.py) vs Blitz bot
        env.run(["main.py", "blitz_bot.py", "random", "random"])
        rewards = [a.reward for a in env.steps[-1]]
        if rewards[0] == 1: stats['win'] += 1
        else: stats['loss'] += 1
        print(f"Game {i+1}: {rewards}")
    print(f"Stats: {stats}")

if __name__ == "__main__":
    run(10)
