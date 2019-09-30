# Training and evaluating an agent with SARSA lambda (backward view) algorithm
import gym
import numpy as np

env = gym.make('Taxi-v2')
Q = np.zeros((env.observation_space.n, env.action_space.n))  # initializing q table (action-values) with a state x action shaped matrix

lam = 0.4  # within [0,1]
alpha = 0.6  # step-size parameter
gamma = 0.7  # discounting factor [0,1]
epsilon = 0.1
epochs = 10000

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
    E = np.zeros((env.observation_space.n, env.action_space.n))  # initializing eligibility traces for all state action pairs
    epsilon = 1/(episode+1) # to make sure that all the state-action pairs are visited an infinite number of times
    action = eps_greedy_action(Q[state], epsilon)  # taking a epsilon greedy action

    while not done:
        next_state, reward, done, info = env.step(action)
        next_action = eps_greedy_action(Q[next_state], epsilon)
        error = reward+gamma*Q[next_state, next_action]-Q[state, action] # updating the q action values
        E[state, action] += 1
        Q += alpha * error * E
        E *= gamma * lam
        G += reward
        state = next_state
        action = next_action

    if (episode + 1) % 500 == 0:
        print('Episode {}: Total Reward: {}'.format(episode + 1, G))

print("Training Finished for {} epochs.".format(episode + 1))

# Evaluating the agents performance after the implementation of SARSA_Lambda
epochs = 100
for episode in range(epochs):
    G = 0  # to sum up total reward
    reward = 0
    done = False
    state = env.reset()

    while not done:
        action = np.argmax(Q[state])  # taking a greedy action for agent evaluation
        next_state, reward, done, info = env.step(action)
        G += reward
        state = next_state

    if (episode+1)%10==0:
        print('Episode {}: Total Reward: {}'.format(episode+1, G))