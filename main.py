from env import env
from DQN import agent

num_episodes = 500

def train(env, agent):
    for episode in range(num_episodes):
        state = env.reset()
        done = False
        while not done:
            action = agent.select_action(state)
            next_state, reward, done = env.step(action.item())
            action = agent.select_action(state)