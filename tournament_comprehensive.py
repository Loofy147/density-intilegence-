from kaggle_environments import make
import collections

def run(n=20):
    stats = collections.Counter()
    for i in range(n):
        env = make("orbit_wars")
        # main (v121) vs Blitz vs Spam vs Random
        env.run(["main.py", "blitz_bot.py", "spam_bot.py", "random"])
        rewards = [a.reward for a in env.steps[-1]]
        if rewards[0] == 1: stats['win'] += 1
        else: stats['loss'] += 1

        # Track who won if we lost
        if rewards[0] == -1:
            if rewards[1] == 1: stats['lost_to_blitz'] += 1
            elif rewards[2] == 1: stats['lost_to_spam'] += 1
            elif rewards[3] == 1: stats['lost_to_random'] += 1

        print(f"Game {i+1}: {rewards}")
    print(f"\nFinal Stats: {stats}")
    print(f"Win Rate: {stats['win']/n*100}%")

if __name__ == "__main__":
    run(20)
