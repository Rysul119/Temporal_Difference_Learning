import gym
import math
import numpy as np
from IPython.display import clear_output
from time import sleep
import pandas as pd
import matplotlib.pyplot as plt

env = gym.make('CartPole-v1')
env.seed(1)
np.random.seed(2)
pos = np.around(np.arange(env.observation_space.low[0], env.observation_space.high[0], 0.5), 1)
vel = np.around(np.arange(-0.5, 0.5, 0.05), 1)
ang_pos = np.around(np.arange(env.observation_space.low[2], env.observation_space.high[2], 0.03), 2)
ang_vel = np.around(np.arange(-math.radians(50), math.radians(50), 0.04), 2)


def to_state(obs):
    state_pos = np.digitize(obs[0], pos)
    state_vel = np.digitize(obs[1], vel)
    state_ang_pos = np.digitize(obs[2], ang_pos)
    state_ang_vel = np.digitize(obs[3], ang_vel)
    return (state_pos, state_vel, state_ang_pos, state_ang_vel)


def eps_greedy_action(q_values, epsilon):
    if np.random.random() < epsilon:
        return env.action_space.sample()  # returning a random action if a random value [0,1] is less than epsilon
    else:
        return np.argmax(q_values)


Q = np.zeros(shape=(len(pos) + 1, len(vel) + 1, len(ang_pos) + 1, len(ang_vel) + 1, env.action_space.n))

epsilon = 0.1
alpha = 0.6 # step-size parameter
gamma = 0.99  # discounting factor [0,1]
epochs = 100000
score = []
avg_scores = []
for episode in range(epochs):
    G = 0
    reward = 0
    done = False
    obs = env.reset()
    state = to_state(obs)
    # epsilon = 1/(episode+1) # to make sure that all the state-action pairs are visited an infinite number of times
    action = eps_greedy_action(Q[state], epsilon)  # taking a epsilon greedy action

    while not done:
        next_obs, reward, done, info = env.step(action)
        next_state = to_state(next_obs)
        next_action = eps_greedy_action(Q[next_state], epsilon)
        Q[state][action] += alpha * (
                    reward + gamma * Q[next_state][next_action] - Q[state][action])  # updating the q action values
        G += reward
        state = next_state
        action = next_action
    score.append(G)
    if (episode + 1) % 10 == 0:
        avg_score = np.sum(score[-10:]) / 10
        avg_scores.append(avg_score)
        print('Average reward at episode {} is: {}'.format(episode + 1, avg_score))

print("Training Finished for {} epochs.".format(episode+1))
x = np.arange(1,epochs+1,10)
print(len(avg_scores))
print(len(x))
assert (len(avg_scores) == len(x))

dict = {'Episodes': x, 'Avg_rewards': avg_scores}
df = pd.DataFrame(dict)
df['alpha'] = alpha
df['gamma'] = gamma
df['epsilon'] = epsilon
path = 'Experiments'
df.to_csv(path+'/SARSA_2.csv')
dict1 = {'Reward': score}
df1 = pd.DataFrame(dict1)
df1['alpha'] = alpha
df1['gamma'] = gamma
df1['epsilon'] = epsilon
df1.to_csv(path+'/SARSA_2_all.csv')

plt.plot(x, avg_scores)
plt.xlabel('Episodes')
plt.ylabel('Score')
plt.title('Training Scores with the Progression of Episodes')
plt.show()

# Evaluating the agents performance using SARSA
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

    if (episode + 1) % 250 == 0:
        avg_score = np.sum(score_test[-250:]) / 250
        print('Average reward at episode {} is: {}'.format(episode + 1, avg_score))

env.close()

plt.plot(score)
plt.xlabel('Episodes')
plt.ylabel('Score')
plt.title('Test Scores with the Progression of Episodes')
plt.show()