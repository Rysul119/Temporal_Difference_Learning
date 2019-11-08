import gym
import tensorflow as tf
import numpy as np
import os
from collections import  deque

tf.keras.backend.set_floatx('float64')
env = gym.make('CartPole-v1')

gamma = 0.99 # discounting factor [0,1]
lmda = 0.95
alpha =  0.05 # actor learning rate
beta = 0.04 # critic learning rate
epochs = 200
ppo_steps = 200
clip_coeff = 0.2


state_size = env.observation_space.shape[0]
action_size = env.action_space.n
action_space = [0, 1]

def Critic_NN(in_size, out_size): ## A regression NN

    model = tf.keras.Sequential([
        tf.keras.layers.Dense(30, input_shape = [in_size], activation = 'relu'),
        tf.keras.layers.Dense(30, activation = 'relu'),
        tf.keras.layers.Dense(out_size, activation = None)
    ])

    return model


def Actor_NN(in_size, out_size): ## A categorial NN

    model = tf.keras.Sequential([
        tf.keras.layers.Dense(30, input_shape=[in_size], activation='relu'),
        tf.keras.layers.Dense(30, activation = 'relu'),
        tf.keras.layers.Dense(out_size, activation = 'softmax')
    ])

    return model

def Critic_loss(loss):

    return tf.math.reduce_mean(tf.math.square(loss))

def Actor_loss(advantage, ratio, clip_coeff):

    surr_loss = ratio * advantage
    loss = - tf.math.reduce_mean(tf.minimum(surr_loss, tf.clip_by_value(ratio, 1 - clip_coeff, 1 + clip_coeff) * advantage)) # include entropy?

    return  loss

def Critic_update(model, states, returns, values):

    states = np.array(states).reshape(ppo_steps, -1)
    returns = np.array(returns).flatten()
    model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])
    model.fit(states, returns, epochs = 5, shuffle = True, verbose = 0)

    #optimizer = tf.keras.optimizers.Adam(beta)
    #with tf.GradientTape() as tape:
        #q_values = model(states, training = True)
        #adv = np.array(rewards) - np.array(q_values)
        #loss = Critic_loss(adv)

    #gradients = tape.gradient(loss, model.trainable_weights)
    #optimizer.apply_gradients(zip(gradients, model.trainable_weights))




def Actor_update(model, model_old, advantage, states, clip_coeff):

    optimizer = tf.keras.optimizers.Adam(alpha)
    with tf.GradientTape() as tape:
        pi_prob = model(states, training=True)
        pi_old_prob = model_old(states, training=False)
        ratio = pi_prob / pi_old_prob
        loss = Actor_loss(advantage, ratio, clip_coeff)

    gradients = tape.gradient(loss, model.trainable_weights)
    optimizer.apply_gradients(zip(gradients, model.trainable_weights))


def Advantage(values, masks, rewards, lmbda):

    returns = []
    gae = 0

    for i in reversed(range(len(rewards))):
        delta = rewards[i] + gamma * values[i+1] * masks[i] - values[i]
        gae = delta + gamma * lmbda * masks[i] * gae
        returns.insert(0, gae + values[i])

    adv = np.array(returns) - values[:-1]

    return returns, (adv - np.mean(adv)) / (np.std(adv) + 1e-10)

def test_reward():
    state = env.reset()
    done = False
    total_reward = 0
    print('testing...')
    iter = 0
    while not done:
        state = np.reshape(state, (1, -1))
        action_probs = actor.predict(state)
        action = np.argmax(action_probs)
        next_state, reward, done, _ = env.step(action)
        state = next_state
        total_reward += reward
        iter += 1
        if iter > 50:
            break
    return total_reward

critic = Critic_NN(state_size, 1)

actor = Actor_NN(state_size, action_size)
actor_old = Actor_NN(state_size, action_size)


for episode in range(epochs):
    states = []
    actions = []
    rewards = []
    values = []
    masks = []
    action_probs = []
    action_onehots = []
    state = env.reset()
    state = np.reshape(state, (1, -1))

    for steps in range(ppo_steps):
        action_prob = actor.predict(state)
        q_value = critic.predict(state)
        action = int(np.random.choice(action_space, 1, p = action_prob[0]))
        action_onehot = np.zeros(action_size)
        action_onehot[action] = 1

        next_state, reward, done, info = env.step(action)
        mask = not done

        states.append(state)
        actions.append(action)
        rewards.append(reward)
        values.append(q_value)
        masks.append(mask)
        action_probs.append(action_prob)
        action_onehots.append(action_onehot)

        state = next_state
        state = np.reshape(state, (1, -1))
        if done:
            env.reset()

    q_value = critic.predict(state)
    values.append(q_value)
    returns, advantages = Advantage(values, masks, rewards, lmda)

    Critic_update(critic, states, returns, values)
    Actor_update(actor, actor_old, advantages, states, clip_coeff)

    avg_reward = np.mean([test_reward() for _ in range(5)])
    print('Episode: {} '.format(episode+1))
    print('Total test reward {}'.format(avg_reward))

env.close()