import gym
import numpy as np
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt

tf.keras.backend.set_floatx('float64')

gamma = 0.99

#intiallize the parameters -->> create a neural network
#Collect trajectories for a number of episodes
#For each episode trajectories get the log prob distribution, get a derivative to update the parameters of the neural net,and get the discounted reward.

#This policy network returns the probability distribution of all the actions for an input state
class PolicyNetwork(tf.keras.Model):
    def __init__(self, num_states, num_actions, hidden_size, lr = 3e-4):
        super(PolicyNetwork, self).__init__()

        self.num_actions = num_actions
        self.l1 = tf.keras.layers.Dense(hidden_size, input_shape = [num_states], name = 'l1')
        self.out = tf.keras.layers.Dense(num_actions, name = 'out')
        self.optimizer = tf.keras.optimizers.Adam(lr)

    def __call__(self, input):
        state = tf.convert_to_tensor(input, dtype=tf.float64)
        state = tf.expand_dims(state, 0)
        l1_out = tf.nn.relu(self.l1(state))
        out_probs = tf.nn.softmax(self.out(l1_out), axis = 1)

        return out_probs


    def take_action(self, state):
        probs = self.__call__(state)
        #print(probs)
        #choosing an action based on the probability output
        action = np.random.choice(self.num_actions, p = np.squeeze(probs.numpy()))
        log_prob = tf.math.log(tf.squeeze(probs, axis = 0)[action])

        return action, log_prob

def discounting(rewards):
    # calculation of discounted rewards
    disc_rewards = []
    # recursive function?
    for i in range(len(rewards)):
        pow = 0
        g = 0
        for reward in rewards[i:]:
            g += (gamma ** pow) * reward
            pow += 1
        disc_rewards.append(g)

    # converting to array
    disc_rewards = np.array(disc_rewards)
    # normalizing discounted rewards
    norm_disc_rewards = (disc_rewards - disc_rewards.mean()) / disc_rewards.std()
    # converting to tensor
    norm_disc_rewards = tf.convert_to_tensor(norm_disc_rewards, dtype=tf.float64)
    return  norm_disc_rewards

def policy_update(model_nn, rewards, log_probs):

    optimizer = model_nn.optimizer
    with tf.GradientTape() as tape:
        norm_disc_rewards = discounting(rewards)

        loss = [-log_prob * g for log_prob, g in zip(log_probs, norm_disc_rewards)]
        loss = np.array(loss).sum()
        loss = tf.convert_to_tensor(loss, dtype = tf.float64)
        print(loss)
    #print(model_nn.trainable_variables)
    gradients = tape.gradient(loss, model_nn.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model_nn.trainable_variables))


#training function inside update function///
def training(env, episodes, steps):
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n

    policy_net = PolicyNetwork(num_states=state_size, num_actions=action_size, hidden_size=128)
    optimizer = policy_net.optimizer
    num_steps = []
    avg_steps = []
    all_rewards = []

    for episode in range(episodes):
        log_probs = []
        rewards = []
        state = env.reset()

        with tf.GradientTape() as tape:

            for step in range(steps):
                action, log_prob = policy_net.take_action(state)
                new_state, reward, done, _ = env.step(action)
                log_probs.append(log_prob)
                rewards.append(reward)

                if done:
                    #policy_update(policy_net, rewards, log_probs)
                    num_steps.append(step+1)
                    avg_steps.append(np.mean(num_steps[-10:]))
                    all_rewards.append(np.sum(rewards))
                    if (episode+1) % 10 == 0:
                        print("Episode: {}, Total Reward: {}, Average_reward: {}, Length: {}\n".format(episode, np.round(np.sum(rewards), decimals = 3),  np.round(np.mean(all_rewards[-10:]), decimals = 3), steps))
                    break
            norm_disc_rewards = discounting(rewards)
            loss = [-log_prob * g for log_prob, g in zip(log_probs, norm_disc_rewards)]
            loss = np.array(loss).sum()
            loss = tf.convert_to_tensor(loss, dtype=tf.float64)
        gradients = tape.gradient(loss, policy_net.trainable_variables)
        optimizer.apply_gradients(zip(gradients, policy_net.trainable_variables))
    return all_rewards

env = gym.make('CartPole-v1')
episodes = 1000
steps = 500

tot_rewards = training(env, episodes, steps)

plt.plot(tot_rewards)
plt.xlabel(Episdoes)
plt.ylabel(Rewards)
plt.show()