import gym
import math
import numpy as np
from IPython.display import clear_output
from time import sleep
import matplotlib.pyplot as plt

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
score = []
avg_score = []
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
    score.append(G)

    if (episode+1)%250==0:
        avg_score = np.sum(score[-250:])/250
        print('Average reward at episode {} is: {}'.format(episode+1, avg_score))

print("Training Finished for {} epochs.".format(episode+1))

plt.plot(score)
plt.xlabel('Episodes')
plt.ylabel('Score')
plt.title('Training Scores with the Progression of Episodes')
plt.show()

# Evaluating the agents performance after Q_learning
epochs_test = 10000
score_test = []
for episode in range(epochs_test):
    obs_test = env.reset()
    G = 0
    reward = 0
    done = False
    while not done:
        env.render()
        state_test = to_state(obs_test)
        action_test = np.argmax(Q[state_test])
        obs_test, reward, done, info = env.step(action_test)
        G += reward
    score_test.append(G)

    if (episode+1)%250==0:
        avg_score = np.sum(score_test[-250:])/250
        print('Average reward at episode {} is: {}'.format(episode+1, avg_score))

env.close()

plt.plot(score)
plt.xlabel('Episodes')
plt.ylabel('Score')
plt.title('Test Scores with the Progression of Episodes')
plt.show()