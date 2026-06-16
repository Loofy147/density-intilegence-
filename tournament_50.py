from kaggle_environments import make
import collections

def run(n=50):
    counts = collections.Counter()
    for i in range(n):
        env = make("orbit_wars")
        # v95 vs Blitz vs Spam vs Random
        env.run(["main.py", "blitz_bot.py", "spam_bot.py", "random"])
        rewards = [a.reward for a in env.steps[-1]]
        if rewards[0] == 1: counts['win'] += 1
        elif rewards[0] == -1: counts['loss'] += 1
        else: counts['error'] += 1

        if (i+1) % 10 == 0:
            print(f"Progress {i+1}/{n}: {counts}")
    print(f"Final Stats: {counts}")

if __name__ == "__main__":
    run(50)
