import tensorflow as tf
import numpy as np
import gym
from scipy.special import softmax

env = gym.make('CartPole-v1')


Actor_NN = tf.keras.models.load_model('saved_model/Actor_network1')
#Critic_NN = tf.keras.models.load_model('saved_model/Critic_network1 ')

epochs_test = 500
for episode in range(epochs_test):
    obs_test = env.reset()
    G = 0
    reward = 0
    done = False
    while not done:
        env.render()
        obs_test = np.reshape(obs_test, (1, -1))
        action_pred = Actor_NN.predict(obs_test)
        #print(type(action_pred))
        #action_pred = tf.convert_to_tensor(action_pred)
        #action_pred = tf.reshape(action_pred, [1, ])

        action_test = np.argmax(action_pred)
        obs_test, reward, done, info = env.step(action_test)
        G += reward

    print('Episode: {}/{} had total reward: {}.'.format(episode + 1, epochs_test, G))
env.close()