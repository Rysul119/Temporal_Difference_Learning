# Training and evaluating an agent with Q_learning algorithm
import gym
import numpy as np

env = gym.make('Taxi-v2')

Q = np.zeros((env.observation_space.n, env.action_space.n))  # initializing q table (action-values) with a state x action shaped matrix

alpha = 0.5  # step-size parameter
gamma = 0.7  # discounting factor [0,1]
epsilon = 0.1  # within [0,1]
epochs = 1000
def eps_greedy_action(q_values, epsilon):
    if np.random.random()<epsilon:
        return env.action_space.sample()  # returning a random action if a random value [0,1] is less than epsilon
    else:
        return np.argmax(q_values)

for episode in range(epochs):
    G = 0  # to sum up total reward
    reward = 0
    done = False
    state = env.reset()

    while not done:
        action = eps_greedy_action(Q[state], epsilon) # taking a epsilon greedy action
        state2, reward, done, info = env.step(action)
        Q[state, action] += alpha*(reward+gamma*np.max(Q[state2])-Q[state,action])  # updating the q action values
        G += reward
        state = state2

    if (episode+1)%20==0:
        print('Episode {}: Total Reward: {}'.format(episode+1, G))

print("Training Finished for {} epochs.".format(episode+1))

# Evaluating the agents performance after Q_learning
epochs = 100
for episode in range(epochs):
    G = 0  # to sum up total reward
    reward = 0
    done = False
    state = env.reset()

    while not done:
        action = np.argmax(Q[state])  # taking a greedy action for agent evaluation
        state2, reward, done, info = env.step(action)
        G += reward
        state = state2

    if (episode+1)%10==0:
        print('Episode {}: Total Reward: {}'.format(episode+1, G))
