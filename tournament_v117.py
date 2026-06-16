from kaggle_environments import make
import collections

def run(n=10):
    counts = collections.Counter()
    for i in range(n):
        env = make("orbit_wars")
        env.run(["main.py", "blitz_bot.py", "spam_bot.py", "random"])
        rewards = [a.reward for a in env.steps[-1]]
        if rewards[0] == 1: counts['win'] += 1
        else: counts['loss'] += 1
        print(f"Game {i+1}: {rewards}")
    print(f"v117 Stats: {counts} | Win rate: {counts['win']/n*100}%")

if __name__ == "__main__":
    run(10)
