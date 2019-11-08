import gym
import math
import numpy as np
from IPython.display import clear_output
from time import sleep

env = gym.make('CartPole-v1')

pos = np.around(np.arange(env.observation_space.low[0], env.observation_space.high[0], 0.5), 1)
vel = np.around(np.arange(-0.5, 0.5, 0.05), 1)
ang_pos = np.around(np.arange(env.observation_space.low[2], env.observation_space.high[2], 0.03), 2)
ang_vel = np.around(np.arange(-math.radians(50), math.radians(50), 0.04), 2)

def to_state(obs):
    state_pos = np.digitize(obs[0], pos)
    state_vel = np.digitize(obs[1], vel)
    state_ang_pos = np.digitize(obs[2], ang_pos)
    state_ang_vel = np.digitize(obs[3], ang_vel)
    return(state_pos, state_vel, state_ang_pos, state_ang_vel)

def eps_greedy_action(q_values, epsilon):
    if np.random.random()<epsilon:
        return env.action_space.sample()  # returning a random action if a random value [0,1] is less than epsilon
    else:
        return np.argmax(q_values)

Q = np.zeros(shape = (len(pos) + 1, len(vel) + 1, len(ang_pos) + 1, len(ang_vel) + 1, env.action_space.n))

alpha = 0.5  # step-size parameter
gamma = 0.9 # discounting factor [0,1]
epsilon = 0.1  # within [0,1]
epochs = 10000

for episode in range(epochs):
    G = 0  # to sum up total reward
    reward = 0
    done = False
    obs = env.reset()
    state = to_state(obs)

    while not done:
        action = eps_greedy_action(Q[state], epsilon) # taking a epsilon greedy action
        new_obs, reward, done, info = env.step(action)
        new_state = to_state(new_obs)
        Q[state][action] += alpha*(reward+gamma*np.max(Q[new_state])-Q[state][action])  # updating the q action values
        G += reward
        state = new_state

    if (episode+1)%20==0:
        print('Episode {}: Total Reward: {}'.format(episode+1, G))

print("Training Finished for {} epochs.".format(episode+1))

# Evaluating the agents performance after Q_learning
epochs = 100
for episode in range(epochs):
    G = 0  # to sum up total reward
    reward = 0
    done = False
    obs = env.reset()
    state = to_state(obs)
    frames = []
    t = 0

    while not done:
        frames.append(env.render(mode='rgb_array'))
        state = to_state(obs)
        action = np.argmax(Q[state])
        new_obs, reward, done, _ = env.step(action)
        new_state = to_state(obs)
        state = new_state