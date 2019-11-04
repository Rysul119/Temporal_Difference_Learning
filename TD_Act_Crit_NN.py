import numpy as np
import gym
import tensorflow as tf
from scipy.special import softmax
import os
tf.keras.backend.set_floatx('float64')

env = gym.make('CartPole-v1')

alpha = 0.002  # learning rate for actor
beta = 0.01 # learning rate for critic
gamma = 0.95  # discounting factor [0,1]
epsilon = 0.3  # within [0,1]
batch_size = 25
epochs = 200
train_epochs = 1
state_size = env.observation_space.shape[0]
action_size = env.action_space.n

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
        tf.keras.layers.Dense(out_size, activation = None) # no activation = 'softmax' as sparse_softmax_cross_entropy_logit is gonna be used for loss calculation
    ])

    return model

def Critic_loss(TD_error):
    return tf.math.reduce_mean(tf.math.square(TD_error))

def Actor_loss(TD_error, actor_pred, action):
    actor_pred = tf.convert_to_tensor(actor_pred)
    action = tf.convert_to_tensor(action)
    action = tf.reshape(action, [1, ])
    #print(action)
    return tf.math.reduce_mean(tf.stop_gradient(TD_error) * tf.squeeze(tf.nn.sparse_softmax_cross_entropy_with_logits(labels = action, logits = actor_pred)))

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

'''def Critic_Update(model, state, next_state):

    TD_target = reward + gamma * tf.squeeze(model.predict(next_state))
    TD_error = TD_target - tf.squeeze(model.predict(state))
    #critic_loss = tf.math.reduce_mean(tf.math.square(TD_error))
    #critic_optim = tf.compat.v1.train.AdamOptimizer(beta).minimize(Critic_loss(TD_error))
    f_without_any_args = functools.partial(Critic_loss, TD_error = TD_error)
    #optimizer.minimize(f_without_any_args, X)
    critic_optim = tf.keras.optimizers.Adam(beta).minimize(f_without_any_args, var_list = model.trainable_weights)

    return TD_error'''


def Actor_Update(model, TD_error, state, action, train_epochs):
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

    #actor_pred = model.predict(state)
    #action_prob = tf.nn.softmax(actor_pred)

    #sampled_action = tf.squeeze(tf.random.categorical(actor_pred,1))
    #action = tf.keras.utils.to_categorical(sampled_action, num_classes = action_size)
    #return np.argmax(action_prob)

    actor_pred = model(state)
    action_prob = softmax(actor_pred)
    return np.random.choice(a = np.arange(action_prob.shape[1]), p = action_prob.reshape(-1))



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

print('Training for 500 episodes is complete.')

path = 'saved_model/Actor_network1'
os.makedirs(path)
Model_Actor.save(path)

path = 'saved_model/Critic_network1'
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
        action_pred = Model_Actor.predict(obs_test)
        #print(type(action_pred))
        #action_pred = tf.convert_to_tensor(action_pred)
        #action_pred = tf.reshape(action_pred, [1, ])
        action_prob = softmax(action_pred)
        action_test = np.argmax(action_prob)
        obs_test, reward, done, info = env.step(action_test)
        G += reward

    print('Episode: {}/{} had total reward: {}.'.format(episode + 1, epochs_test, G))
env.close()