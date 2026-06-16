from kaggle_environments import make
def run():
    env = make("orbit_wars")
    env.run(["main.py", "blitz_bot.py", "spam_bot.py", "random"])
    print("Result:", [a.reward for a in env.steps[-1]])
run()
run()
run()
