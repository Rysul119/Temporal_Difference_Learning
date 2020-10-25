import gym
import numpy as np
import tensorflow as tf
import os
import random
from collections import  deque
tf.keras.backend.set_floatx('float64')
env = gym.make('CartPole-v1')

gamma = 0.9 # discounting factor [0,1]
epsilon = 0.4  # within [0,1]
alpha =  0.001 # actor learning rate
beta = 0.02 # critic learning rate
batch_size = 60
epochs = 200
train_epochs = 1
tau = 0.01 # soft update
memory = deque(maxlen=2000)

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
        tf.keras.layers.Dense(out_size, activation = 'softmax') # no activation = 'softmax' as sparse_softmax_cross_entropy_logit is gonna be used for loss calculation
    ])

    return model

def remember(state, action, action_prob, next_action_prob, reward, next_state, done):

    memory.append((state, action, action_prob, next_action_prob, reward, next_state, done))

def replay(memory, batch_size):

    curr_batch = random.sample(memory, batch_size)
    state = np.zeros((batch_size, state_size))
    next_state = np.zeros((batch_size, state_size))
    action_prob = np.zeros((batch_size, action_size))
    next_action_prob = np.zeros((batch_size, action_size))
    action = []
    reward = []
    done = []

    for i in range(batch_size):
        state[i] = curr_batch[i][0]
        action.append(curr_batch[i][1])
        action_prob[i] = curr_batch[i][2]
        next_action_prob[i] = curr_batch[i][3]
        reward.append(curr_batch[i][4])
        next_state[i] = curr_batch[i][5]
        done.append(curr_batch[i][6])

    return state, action, action_prob, next_action_prob, reward, next_state, done

def Action_choice(model, state, epsilon):

    action_prob = model.predict(state)[0]
    if np.random.random() < epsilon:
        choice = env.action_space.sample()
    else:
        choice = int(np.random.choice(action_space, 1, p=action_prob))

    return choice, action_prob


def Update(actor, actor_target, critic, critic_target,memory, batch_size):

    states, actions, action_probs, next_action_probs, rewards, next_states, _ = replay(memory, batch_size)
    x = tf.concat([states, action_probs], axis=1)
    y = tf.concat([next_states, next_action_probs], axis = 1)
    # gradients for update for critic...
    critic_optimizer = tf.keras.optimizers.Adam(beta)
    actor_optimizer = tf.keras.optimizers.Adam(alpha)

    for i in range(train_epochs):
        with tf.GradientTape() as tape:
            Q = critic(x, training=True)
            td_target = rewards + gamma * critic_target(y)  # TD_target calculated from critic
            td_target = tf.stop_gradient(td_target)
            critic_loss = tf.math.reduce_mean(tf.math.squared_difference(td_target, Q))

        critic_gradients = tape.gradient(critic_loss, critic.trainable_variables)
        critic_optimizer.apply_gradients(zip(critic_gradients, critic.trainable_variables))

        # gradients for update for actor...
    for i in range(train_epochs):
        with tf.GradientTape() as tape2:
            action_prob = actor(states, training=True)
            z = tf.concat([states, action_prob], axis=1)
            # action = actor(state, training=True)
            z_ = critic(z)
            actor_loss = - tf.math.reduce_mean(z_)

        actor_gradients = tape2.gradient(actor_loss, actor.trainable_variables)
        actor_optimizer.apply_gradients(zip(actor_gradients, actor.trainable_variables))

        # params = critic.trainable_variables
        # params_target = critic_target.trainable_variables
        # Soft-updating the target parameters online
        # params_target_update = [
        #   params_target[i].assign(tf.math.multiply(params[i], tau) + tf.math.multiply(params_target[i], (1.0 - tau))) for
        #  i in range(len(params_target))]

        # params = actor.trainable_variables
        # params_target = actor_target.trainable_variables
        # print(type(params[1]))
        # Soft-updating the target parameters online
        # params_target_update = [
        #   params_target[i].assign(tf.math.multiply(params[i], tau) + tf.math.multiply(params_target[i], (1.0 - tau))) for
        #  i in range(len(params_target))]
        # actor_trainable_vars = len(params) + len(params_target)

        for params, target_params in zip(critic.trainable_variables, critic_target.trainable_variables):
            target_params.assign((tf.math.multiply(params, tau) + tf.math.multiply(target_params, (1.0 - tau))))

        for params, target_params in zip(actor.trainable_variables, actor_target.trainable_variables):
            target_params.assign((tf.math.multiply(params, tau) + tf.math.multiply(target_params, (1.0 - tau))))


critic = Critic_NN(state_size + action_size, 1)
critic_target = Critic_NN(state_size + action_size, 1)

actor = Actor_NN(state_size, action_size)
actor_target = Actor_NN(state_size, action_size)

for params, target_params in zip(critic.trainable_variables, critic_target.trainable_variables):
    target_params.assign(params)

for params, target_params in zip(actor.trainable_variables, actor_target.trainable_variables):
    target_params.assign(params)


for episode in range(epochs):
    G = 0
    reward = 0
    done = False

    state = env.reset()
    state = np.reshape(state, (1,-1))

    while not done:
        action, action_prob = Action_choice(actor, state, epsilon)
        next_state, reward, done, info = env.step(action)
        next_state = np.reshape(next_state, (1,-1))
        _, next_action_prob = Action_choice(actor_target, next_state, epsilon)
        remember(state, action, action_prob, next_action_prob, reward, next_state, done)

        if len(memory) > batch_size:
            Update(actor, actor_target, critic, critic_target, memory, batch_size)

        G += reward
        state = next_state


    print('Episode: {}/{} had total reward: {}.'.format(episode + 1, epochs, G))

print('Training is done after {} episodes'.format(epochs) )


path = 'saved_model/Actor_network_DDPG'
os.makedirs(path)
actor.save(path)

path = 'saved_model/Critic_network_DDPG'
os.makedirs(path)
critic.save(path)

epochs_test = 200
for episode in range(epochs_test):
    obs_test = env.reset()
    G = 0
    reward = 0
    done = False
    while not done:
        env.render()
        obs_test = np.reshape(obs_test, (1, -1))
        action_pred = actor.predict(obs_test)
        #print(action_pred)
        #print(type(action_pred))
        #action_pred = tf.convert_to_tensor(action_pred)
        #action_pred = tf.reshape(action_pred, [1, ])
        action_test = np.argmax(action_pred)
        obs_test, reward, done, info = env.step(action_test)
        G += reward

    print('Episode: {}/{} had total reward: {}.'.format(episode + 1, epochs_test, G))
env.close()