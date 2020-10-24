import gym
import tensorflow as tf
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
from timeit import default_timer as timer
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

gamma = 0.99
env = gym.make("CartPole-v1")
env.seed(0)
np.random.seed(0)
tf.random.set_seed(0)

#create the neural net model for action-value function
model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(64, input_shape = (4,), activation = 'tanh', kernel_initializer= 'RandomNormal'))
#model.add(tf.keras.layers.Dense(64, activation = 'tanh', kernel_initializer='RandomNormal'))
model.add(tf.keras.layers.Dense(2, activation = 'linear', kernel_initializer='RandomNormal'))
model.build()
optimizer = tf.keras.optimizers.Adam(lr = 0.0001)

batch_size = 64
tot_experience = {'s': [], 'a': [], 'r': [], 's2': [], 'done':[]}
max_experience = 1000
min_experience = 32
ep_score = 0
ep_step = 0
episode = 0
scores = []
steps = []
tot_steps = 1000000
epsilon = 0.99
decay = 0.99999
min_epsilon = 0.1

s = env.reset()
#s = s.reshape([1, 4])
for step in range(tot_steps):

    #reduces the value of epsilon
    epsilon = max(min_epsilon, decay * epsilon)
    #gets an action
    if  np.random.random() < epsilon:
        a = np.random.choice(env.action_space.n)
    else:
        a = np.argmax(model(np.atleast_2d(s)))
    #takes the action
    next_s, r, done, _ = env.step(a)
    #next_s = next_s.reshape([1, 4])
    ep_score += r
    ep_step += 1
    exp = {"s": s, 'a': a, 'r': r, 's2': next_s, 'done': done}
    s = next_s

    if done:
        scores.append(ep_score)
        steps.append(ep_step)
        episode += 1
        if episode % 100 == 0:
            print("Episode: {} episode score: {} epsilon: {} avg score: {}".format(episode, ep_score,epsilon, np.mean(scores[-100:])))
        ep_score = 0
        ep_step = 0
        s = env.reset()
    # Deletes the other experiences if it reaches to the maximum
    if (len(tot_experience["s"])>=max_experience):
        for key in tot_experience:
            tot_experience[key].pop(0)
    # Adds to the total experience and
    for key, value in exp.items():
        tot_experience[key].append(value)
    #trains the nn with the stored values in tot_experience using a condition
    if (len(tot_experience["s"]) > min_experience):
        #gets random indices for the lists of experiences
        inds = np.random.randint(low = 0, high= len(tot_experience['s']), size = batch_size)
        #using python list comprehension
        states = np.asarray([tot_experience['s'][i] for i in inds])
        actions = np.asarray([tot_experience['a'][i] for i in inds])
        rewards = np.asarray([tot_experience['r'][i] for i in inds])
        next_states = np.asarray([tot_experience['s2'][i] for i in inds])
        dones = np.asarray([tot_experience['done'][i] for i in inds])

        #getting the max action-value function for the next state
        nexts_vals = np.max(model(np.atleast_2d(next_states)), axis = 1)
        #print(nexts_vals.shape)
        #getting the actual/discounted values using np.where << very quick method
        actual_vals = np.where(dones, rewards, rewards + gamma * nexts_vals)

        #Calculating the loss
        with tf.GradientTape() as tape:
            actvals = model(np.atleast_2d(states))
            selected_actions = tf.one_hot(actions, env.action_space.n)
            selected_actvals = tf.math.reduce_sum(actvals * selected_actions, axis = 1)
            loss = tf.reduce_sum(tf.math.square(actual_vals - selected_actvals))
        #getting the gradients
        gradients = tape.gradient(loss, model.trainable_variables)
        #applying the gradients
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))


dict = {'r': scores, 'l': steps}
df = pd.DataFrame(dict)
log_dir = "logs/CartPole-v1/DQN"
os.makedirs(log_dir, exist_ok=True)
file_name = 'monitor.csv'
df.to_csv(log_dir + '/' + file_name)
