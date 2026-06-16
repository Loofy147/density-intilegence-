from kaggle_environments import make
import collections

def run_tournament(n_games=5):
    wins = 0
    for i in range(n_games):
        env = make("orbit_wars")
        # Player 0 is our agent
        env.run(["main.py", "random", "random", "random"])
        rewards = [a.reward for a in env.steps[-1]]
        if rewards[0] == 1:
            wins += 1
        print(f"Game {i+1}: Rewards {rewards}")
    print(f"Total Wins: {wins}/{n_games} ({wins/n_games*100}%)")

if __name__ == "__main__":
    run_tournament(5)
