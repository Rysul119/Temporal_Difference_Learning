import numpy as np
import gym
import tensorflow as tf
from scipy.special import softmax
import os

tf.keras.backend.set_floatx('float64')
env = gym.make('CartPole-v1')

alpha = 0.002  # learning rate for actor
beta = 0.06 # learning rate for critic
gamma = 0.95  # discounting factor [0,1]
epsilon = 0.4 # within [0,1]
batch_size = 25
epochs = 200
train_epochs = 1
h_layer_size = 20
state_size = env.observation_space.shape[0]
action_size = env.action_space.n

def Critic_NN(in_size, out_size): ## A regression NN

    model = tf.keras.Sequential([
        tf.keras.layers.Dense(h_layer_size, input_shape = [in_size], activation = 'relu'),
        tf.keras.layers.Dense(h_layer_size, activation='relu'),
        tf.keras.layers.Dense(out_size, activation = None)
    ])

    return model


def Actor_NN(in_size, out_size): ## A categorial NN

    model = tf.keras.Sequential([
        tf.keras.layers.Dense(h_layer_size, input_shape=[in_size], activation='relu'),
        tf.keras.layers.Dense(h_layer_size, activation='relu'),
        tf.keras.layers.Dense(out_size, activation = 'softmax')
    ])

    return model

def Critic_loss(TD_error):
    return tf.math.reduce_mean(tf.math.square(TD_error))

def Actor_loss(TD_error, actor_pred, action):
    log_prob = tf.math.log(actor_pred)
    return -tf.math.reduce_mean(TD_error * log_prob)

def Critic_Update(model, state, reward, next_state, train_epochs):

    for i in range(train_epochs):
        optimizer = tf.keras.optimizers.Adam(beta)
        with tf.GradientTape() as tape:
            TD_target = reward + gamma * tf.squeeze(model(next_state, training = True))
            TD_error = TD_target - tf.squeeze(model(state, training = True))
            loss = Critic_loss(TD_error)

        gradients = tape.gradient(loss, model.trainable_weights)
        optimizer.apply_gradients(zip(gradients, model.trainable_weights))

    return TD_error


def Actor_Update(model, TD_error, state, action,train_epochs):

    for i in range(train_epochs):
        optimizer = tf.keras.optimizers.Adam(alpha)
        with tf.GradientTape() as tape:
            actor_pred = model(state, training=True)
            #print(type(actor_pred), actor_pred.shape)
            loss = Actor_loss(TD_error, actor_pred, action)

        gradients = tape.gradient(loss, model.trainable_weights)
        optimizer.apply_gradients(zip(gradients, model.trainable_weights))

    #actor_pred = model.predict(state)
    #actor_loss = tf.math.reduce_mean(tf.stop_gradient(TD_error) * tf.nn.sparse_softmax_cross_entropy_with_logits(logits = actor_pred, labesl = action))
    #actor_optim = tf.keras.optimizers.Adam(alpha).minimize(actor_loss, model.trainable_weights)


def Action(model, state, epsilon):

    if np.random.random() < epsilon:
        return env.action_space.sample()
        #action_prob = model(state)[0]
        #return np.random.choice(a=np.arange(action_size), p=action_prob)
    action_prob = model(state)[0]
    #action = tf.reshape(action_prob, [-1])
    return np.random.choice(a=np.arange(action_size), p=action_prob)
    #return np.argmax(action_prob)


Model_Critic = Critic_NN(state_size, 1)
Model_Actor = Actor_NN(state_size, action_size)

for episode in range(epochs):
    G = 0
    reward = 0
    done = False
    #epsilon = 1/ (episode+1)
    state = env.reset()
    state = np.reshape(state, (1,-1))

    while not done:
        action = Action(Model_Actor, state, epsilon)
        next_state, reward, done, info = env.step(action)
        next_state = np.reshape(next_state, (1,-1))
        td_error = Critic_Update(Model_Critic, state, reward, next_state, train_epochs)
        Actor_Update(Model_Actor, td_error, state, action, train_epochs)

        G += reward
        state = next_state
        #if len(memory) > batch_size:
         #   replay(Model, memory, batch_size)

    print('Episode: {}/{} had total reward: {}.'.format(episode + 1, epochs, G))

print('Training is done for 500 episodes')

path = 'saved_model/Actor_network3'
os.makedirs(path)
Model_Actor.save(path)

path = 'saved_model/Critic_network3'
os.makedirs(path)
Model_Critic.save(path)

epochs_test = 500
for episode in range(epochs_test):
    obs_test = env.reset()
    G = 0
    reward = 0
    done = False
    while not done:
        env.render()
        obs_test = np.reshape(obs_test, (1, -1))
        action_pred = Model_Actor.predict((obs_test))
        action_test = np.argmax(action_pred)
        obs_test, reward, done, info = env.step(action_test)
        G += reward

    print('Episode: {}/{} had total reward: {}.'.format(episode + 1, epochs_test, G))
env.close()