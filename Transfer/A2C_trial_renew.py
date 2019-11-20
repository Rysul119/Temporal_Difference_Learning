import gym
from collections import deque
import numpy as np
import pandas as pd
import random
import tensorflow as tf
import tensorflow.keras.layers as kl
import tensorflow.keras.losses as kls
import tensorflow.keras.optimizers as ko

import matplotlib
import matplotlib.pyplot as plt


#policy and vlaue model
class ProbabilityDistribution(tf.keras.Model):
    def call(self, logits):
        # sample a random categorical action from given logits
        return tf.squeeze(tf.random.categorical(logits, 1), axis=-1)

class Model(tf.keras.Model):
    def __init__(self, num_actions):
        super().__init__('mlp_policy')
        # no tf.get_variable(), just simple Keras API
        self.hidden1 = kl.Dense(128, activation='relu')
        self.hidden2 = kl.Dense(128, activation='relu')
        self.value = kl.Dense(1, name='value')
        # logits are unnormalized log probabilities
        self.logits = kl.Dense(num_actions, name='policy_logits')
        self.dist = ProbabilityDistribution()

    def call(self, inputs):
        # inputs is a numpy array, convert to Tensor
        x = tf.convert_to_tensor(inputs)
        # separate hidden layers from the same input tensor
        hidden_logs = self.hidden1(x)
        hidden_vals = self.hidden2(x)
        return self.logits(hidden_logs), self.value(hidden_vals)

    def action_value(self, obs):
        # executes call() under the hood
        logits, value = self.predict(obs)
        action = self.dist.predict(logits)
        # a simpler option, will become clear later why we don't use it
        # action = tf.random.categorical(logits, 1)
        return np.squeeze(action, axis=-1), np.squeeze(value, axis=-1)

#A2C class
class A2CAgent:
    def __init__(self, model):
        # hyperparameters for loss terms, gamma is the discount coefficient
        self.params = {
            'gamma': 0.99,
            'value': 0.5,
            'entropy': 0.0001
        }
        self.model = model
        self.model.compile(
            optimizer=ko.Adam(lr=0.007),
            # define separate losses for policy logits and value estimate
            loss=[self._logits_loss, self._value_loss]
        )
        self.memory = deque(maxlen = 2000)

    def remember(self, state, action, value, reward, next_state, next_value, done):
        self.memory.append((state, action, value, reward, next_state, next_value, done))

    def replay(self, memory, batch_size, state_shape):

        curr_batch = random.sample(memory, batch_size)
        actions = np.empty((batch_size,), dtype=np.int32)
        rewards, dones, values, next_values = np.empty((3, batch_size))
        states = np.empty((batch_size,) + state_shape)
        for i in range(batch_size):
            states[i] = curr_batch[i][0]
            actions[i] = curr_batch[i][1]
            values[i] = curr_batch[i][2]
            rewards[i] = curr_batch[i][3]
            next_values[i] = curr_batch[i][5]
            dones[i] = curr_batch[i][6]

        returns, advs = self._returns_advantages(rewards, dones, values, next_values)
        # a trick to input actions and advantages through same API
        print("States shape: {}, Actions shape: {}, Rewards shape: {}, Values shape: {}, dones shape:{}, returns shape: {}, Advantage shape: {}".format(states.shape, actions.shape, rewards.shape,
                                                                                           values.shape, dones.shape, returns.shape, advs.shape))
        acts_and_advs = np.concatenate([actions[:, None], advs[:, None]], axis=-1)
        # performs a full training step on the collected batch
        # note: no need to mess around with gradients, Keras API handles it
        losses = self.model.train_on_batch(states, [acts_and_advs, returns])

    def train(self, env, batch_sz=32, episodes=1000):
        # storage helpers for a single batch of data
        score = []
        values =[]
        rewards = []
        dones = []
        actions =[]
        states = []
        for episode in range(episodes):
            G = 0
            reward = 0
            done = False
            state = env.reset()
            while not done:
                states.append(state)
                action, value = self.model.action_value(state[None, :])
                next_state, reward, done, _ = env.step(action)
                G += reward
                values.append(value)
                rewards.append(reward)
                dones.append(done)
                state = next_state

            score.append(G)

            _, next_value = self.model.action_value(next_state[None, :])
            values = np.array(values)
            rewards = np.array(rewards)
            dones = np.array(dones)
            actions = np.array(actions)
            states = np.array(states)
            returns, advs = self._returns_advantages(rewards, dones, values, next_value)
            acts_and_advs = np.concatenate([actions[:, None], advs[:, None]], axis = -1 )
            losses = self.model.train_on_batch(states, [acts_and_advs, returns])

        print('Episode: {}/{} Reward: {} Losses: {}'.format(episode+1 , episode, G, losses))

        return score

    def test(self, env, render=False):
        obs, done, ep_reward = env.reset(), False, 0
        while not done:
            action, _ = self.model.action_value(obs[None, :])
            obs, reward, done, _ = env.step(action)
            ep_reward += reward
            if render:
                env.render()
        return ep_reward

    def _returns_advantages(self, rewards, dones, values, next_value):
        # next_value is the bootstrap value estimate of a future state (the critic)
        returns = np.append(np.zeros_like(rewards), next_value, axis=-1)
        # returns are calculated as discounted sum of future rewards
        for t in reversed(range(rewards.shape[0])):
            returns[t] = rewards[t] + self.params['gamma'] * returns[t + 1] * (1 - dones[t])
        returns = returns[:-1]
        # advantages are returns - baseline, value estimates in our case
        advantages = returns - values
        return returns, advantages

    def _value_loss(self, returns, value):
        # value loss is typically MSE between value estimates and returns
        return self.params['value'] * kls.mean_squared_error(returns, value)

    def _logits_loss(self, acts_and_advs, logits):
        # a trick to input actions and advantages through same API
        actions, advantages = tf.split(acts_and_advs, 2, axis=-1)
        # sparse categorical CE loss obj that supports sample_weight arg on call()
        # from_logits argument ensures transformation into normalized probabilities
        weighted_sparse_ce = kls.SparseCategoricalCrossentropy(from_logits=True)
        # policy loss is defined by policy gradients, weighted by advantages
        # note: we only calculate the loss on the actions we've actually taken
        actions = tf.cast(actions, tf.int32)
        policy_loss = weighted_sparse_ce(actions, logits, sample_weight=advantages)
        # entropy loss can be calculated via CE over itself
        entropy_loss = kls.categorical_crossentropy(logits, logits, from_logits=True)
        # here signs are flipped because optimizer minimizes
        return policy_loss - self.params['entropy'] * entropy_loss

env = gym.make('CartPole-v1')
env.seed(1)
np.random.seed(2)
model = Model(num_actions = env.action_space.n)
agent = A2CAgent(model)
#logging.getLogger().setLevel(logging.INFO)

rewards_history = agent.train(env)
dict = {'Score': rewards_history}
df = pd.DataFrame(dict)
df.to_csv('Experiments_ubuntu/A2C_trial2.csv')
plt.style.use('seaborn')
plt.plot(np.arange(0, len(rewards_history), 1), rewards_history[::1])
plt.xlabel('Episode')
plt.ylabel('Total Reward')
plt.show()
