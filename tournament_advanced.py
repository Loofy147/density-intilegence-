from kaggle_environments import make

def run(n=20):
    wins = 0
    for i in range(n):
        env = make("orbit_wars")
        # main vs 3 aggressive bots
        env.run(["main.py", "blitz_bot.py", "spam_bot.py", "spam_blitz.py"])
        rewards = [a.reward for a in env.steps[-1]]
        if rewards[0] == 1: wins += 1
        print(f"Game {i+1}: {rewards}")
    print(f"Total Wins: {wins}/{n} ({wins/n*100}%)")

if __name__ == "__main__":
    run(20)
