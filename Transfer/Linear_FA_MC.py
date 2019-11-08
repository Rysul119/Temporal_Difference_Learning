import gym
import numpy as np
import matplotlib.pyplot as plt
import math
from IPython.display import clear_output
from time import sleep
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDRegressor



env = gym.make('CartPole-v1')

sample_size = 1000
#state_space = env.observation_space.n
action_space = env.action_space.n

observation_examples = np.array([env.observation_space.sample() for x in range(sample_size)])
print(observation_examples)


w = np.zeros((sample_size, action_space))
Q_est = np.dot(w.T, observation_examples)

alpha = 0.5  # step-size parameter
gamma = 0.7  # discounting factor [0,1]
epsilon = 0.2  # within [0,1]
epochs = 10000

def Approximator(state, action):

    x_train =
    y_train =
    x_test =
    sc_x = StandardScaler()
    sc_x_train = sc_x.fit_transform(x_train)
    sc_x_test = sc_x.fit_transform(x_test)
    regressor = SGDRegressor()
    regressor.fit(X_train, y_train)

    return  regressor.predict(X_test)

def eps_greedy_action(q_values, epsilon):
    if np.random.random()<epsilon:
        return env.action_space.sample()  # returning a random action if a random value [0,1] is less than epsilon
    else:
        return np.argmax(q_values)

for episode in range(epochs):
    state = env.reset()
    G = 0
    reward = 0
    done = False
