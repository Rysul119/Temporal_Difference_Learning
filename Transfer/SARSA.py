# Training and evaluating an agent with SARSA algorithm
import gym
import numpy as np
from IPython.display import clear_output
from time import sleep

env = gym.make('Taxi-v2')
Q = np.zeros((env.observation_space.n, env.action_space.n))  # initializing q table (action-values) with a state x action shaped matrix

alpha = 0.5  # step-size parameter
gamma = 0.9  # discounting factor [0,1]
epochs = 100000

def eps_greedy_action(q_values, epsilon):
    if np.random.random() < epsilon:
        return env.action_space.sample()  # returning a random action if a random value [0,1] is less than epsilon
    else:
        return np.argmax(q_values)

for episode in range(epochs):
    G = 0
    reward = 0
    done = False
    state = env.reset()
    epsilon = 1/(episode+1) # to make sure that all the state-action pairs are visited an infinite number of times
    action = eps_greedy_action(Q[state], epsilon)  # taking a epsilon greedy action

    while not done:
        next_state, reward, done, info = env.step(action)
        next_action = eps_greedy_action(Q[next_state], epsilon)
        Q[state, action] += alpha*(reward+gamma*Q[next_state, next_action]-Q[state, action]) # updating the q action values
        G += reward
        state = next_state
        action = next_action

    if (episode + 1) % 500 == 0:
        print('Episode {}: Total Reward: {}'.format(episode + 1, G))

print("Training Finished for {} epochs.".format(episode + 1))

# Evaluating the agents performance after Q_learning
epochs = 100
for episode in range(epochs):
    G = 0  # to sum up total reward
    reward = 0
    done = False
    state = env.reset()
    frames = []
    t = 0

    while not done:
        action = np.argmax(Q[state])  # taking a greedy action for agent evaluation
        state2, reward, done, info = env.step(action)
        G += reward
        clear_output()
        frames.append(env.render())
        print(f'Episode: {episode + 1}')
        print(f'Timestep: {t + 1}')
        print(f'State: {state2}')
        print(f'Action: {action}')
        print(f'Reward: {reward}')
        print(f'Total Reward: {G}')
        sleep(.3)
        t += 1
        state = state2