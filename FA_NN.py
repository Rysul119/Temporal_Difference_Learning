import gym
import tensorflow as tf
import numpy as np
from collections import deque
import random

env = gym.make('CartPole-v1')

gamma = 0.9  # discounting factor [0,1]
epsilon = 0.1  # within [0,1]
batch_size = 25
epochs = 200

memory = deque(maxlen=2000)

state_size = env.observation_space.shape[0]
action_size = env.action_space.n


def Neural_Net(in_size, out_size):

    model = tf.keras.Sequential([
        tf.keras.layers.Dense(24, input_shape = [in_size], activation = 'relu'),
        tf.keras.layers.Dense(24, activation = 'relu'),
        tf.keras.layers.Dense(out_size, activation = 'relu')
    ])

    model.compile(optimizer = 'adam', loss = 'mse', metrics = ['accuracy'])

    return  model

def Eps_greedy_action(model, state, epsilon):
    if np.random.random()<epsilon:
        return env.action_space.sample()  # returning a random action if a random value [0,1] is less than epsilon
    else:
        return np.argmax(model.predict(state)[0])

def remember(state, action, reward, next_state, done):

    memory.append((state, action, reward, next_state, done))

def replay(model, memory, batch_size):

    curr_batch = random.sample(memory, batch_size)
    state = np.zeros((batch_size, state_size))
    next_state = np.zeros((batch_size, state_size))
    action = []
    reward = []
    done = []

    for i in range(batch_size):
        state[i] = curr_batch[i][0]
        action.append(curr_batch[i][1])
        reward.append(curr_batch[i][2])
        next_state[i] = curr_batch[i][3]
        done.append(curr_batch[i][4])

    target = model.predict(state)
    target_val = model.predict(next_state)

    for i in range(batch_size):
        if done[i]:
            target[i][action[i]] = reward[i]
        else:
            target[i][action[i]] = reward[i] + gamma * (np.max(target_val[i]))

    #fitting over the whole batch
    model.fit(state, target, batch_size = batch_size, epochs = 1, verbose = 0)

Model = Neural_Net(state_size, action_size)

for episode in range(epochs):
    G = 0
    reward = 0
    done = False

    state = env.reset()
    state = np.reshape(state, (1,-1))

    while not done:
        action = Eps_greedy_action(Model, state, epsilon)
        next_state, reward, done, info = env.step(action)
        next_state = np.reshape(next_state, (1,-1))
        remember(state, action, reward, next_state, done)
        G += reward
        state = next_state
        if len(memory) > batch_size:
            replay(Model, memory, batch_size)

    print('Episode: {}/{} had total reward: {}.'.format(episode + 1, epochs, G))


# Evaluating the DQN model
epochs_test = 500
for episode in range(epochs_test):
    obs_test = env.reset()
    G = 0
    reward = 0
    done = False
    while not done:
        env.render()
        obs_test = np.reshape(obs_test, (1, -1))
        action_test = np.argmax(Model.predict(obs_test)[0])
        obs_test, reward, done, info = env.step(action_test)
        G += reward

    print('Episode: {}/{} had total reward: {}.'.format(episode + 1, epochs_test, G))
env.close()