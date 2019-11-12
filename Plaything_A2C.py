import numpy as np
import gym
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd

tf.keras.backend.set_floatx('float64')


#hyperparameters
hidden_size = 256
learning_rate = 0.0003

#constants
gamma = 0.99
steps = 1000
episodes = 5000





#Defining an actor critic neural net with arguments input dimension (state dimenstion), actions available, hidden layer neurons.
#Output value function value and policy distribution for a certain state.
#For critic part no activation function is used. Relu is used for hidden layers
#For actor part softmax activation function is used. Relu is used for hidden layers

class AC_NN(tf.keras.Model):
    def __init__(self, num_states, num_actions, hidden_size, learning_rate = 3e-4):
        super(AC_NN, self).__init__()

        self.num_states = num_states
        self.critic_l1 = tf.keras.layers.Dense(hidden_size, input_shape = [num_states], name = 'critic_l1')
        self.critic_out = tf.keras.layers.Dense(1, name = 'critic_out')

        self.num_actions = num_actions
        #self.actor_l1 = tf.keras.layers.Dense(hidden_size, input_shape = [num_states], name = 'actor_l1')
        self.actor_out = tf.keras.layers.Dense(num_actions, name = 'actor_out')

    def __call__(self, state):
        state = tf.convert_to_tensor(state, dtype = tf.float64)
        state = tf.expand_dims(state, 0)

        critic_val = tf.nn.relu(self.critic_l1(state))
        critic_val = self.critic_out(critic_val)

        #actor_dist = tf.nn.relu(self.actor_l1(state))
        actor_dist = tf.nn.softmax(self.actor_out(critic_val), axis = 1)

        return critic_val, actor_dist

#Defining the function for advantage actor critic policy gradient algorithm
def adv_ac(env, learning_rate):
    num_states = env.observation_space.shape[0]
    num_actions = env.action_space.n

    a2c_nn = AC_NN(num_states, num_actions, hidden_size)

    a2c_optimizer = tf.keras.optimizers.Adam(learning_rate)

    epi_lengths = []
    avg_epi_length = []
    epi_rewards = []
    entropy_term = 0

    for episode in range(episodes):

        log_probs = []
        values = []
        rewards = []

        state = env.reset()
        with tf.GradientTape() as tape:

            for step in range(steps):
                value, prob_dist = a2c_nn.__call__(state)
                #print(value, tf.shape(value)) #Check value and prob_dist dimension..
                #print(prob_dist, tf.shape(prob_dist))  # Check value and prob_dist dimension..
                value = tf.stop_gradient(value).numpy()[0, 0]
                dist = tf.stop_gradient(prob_dist).numpy()
                #print(value, np.shape(value))  # Check value and dist dimension..
                #print(dist, np.shape(dist))

                action = np.random.choice(num_actions, p = np.squeeze(dist))
                log_prob = tf.math.log(tf.squeeze(prob_dist, axis = 0)[action])
                #print(log_prob)
                entropy = -np.sum(np.mean(dist)*np.log(dist))

                new_state, reward, done, _ = env.step(action)
                rewards.append(reward)
                values.append(value)
                log_probs.append(log_prob)
                entropy_term += entropy
                state = new_state

                if done or step == (steps - 1):
                    qval, _ = a2c_nn.__call__(new_state)
                    qval = tf.stop_gradient(qval).numpy()[0, 0]
                    #values.append(qval)
                    epi_rewards.append(np.sum(rewards))
                    epi_lengths.append(step+1)
                    avg_epi_length.append(np.mean(epi_lengths[-10:]))

                    if (episode + 1) % 10 == 0:
                        print('Episode: {}, reward: {}, total length: {}, average length: {} \n'.format(episode+1, np.sum(rewards), step+1, avg_epi_length[-1]))

                    break

            values = np.array(values)
            rewards = np.array(rewards)
            qvals = np.zeros_like(values)
            #print(step+1, values.shape, rewards.shape)
            assert values.shape == rewards.shape
            ##TODO check the dimensions of rewards and qvals
            #assigning end qval equal to reward
            qvals[-1] = rewards[-1] # adding only reward to the last Q_value
            for t in reversed(range(len(rewards)-1)):
                qvals[t] = rewards[t] + gamma * qvals[t+1]

            values = tf.convert_to_tensor(values, dtype = tf.float64)
            qvals = tf.convert_to_tensor(qvals, dtype = tf.float64)
            #log_probs = tf.convert_to_tensor(log_probs, dtype = tf.float64)
            #print(log_probs)
            log_probs = tf.stack(log_probs)
            #print(log_probs)

            advantage = tf.math.subtract(qvals, values)
            #print(advantage)
            actor_loss = tf.reduce_mean(tf.math.multiply(-log_probs, advantage))
            #print(actor_loss)
            critic_loss = tf.reduce_mean(tf.math.squared_difference(qvals, values))
            #print(critic_loss)
            tot_loss = actor_loss + critic_loss + 0.001 * entropy_term
            #print(tot_loss)

            #print(a2c_nn.trainable_variables)
        gradients = tape.gradient(tot_loss, a2c_nn.trainable_variables)
        #a2c_loss, gradients = grad(a2c_nn, qvals, values, log_probs, entropy_term)
        #print(gradients)
        a2c_optimizer.apply_gradients(zip(gradients, a2c_nn.trainable_variables))
        print('Actor loss: {}, Critic loss: {}, Entropy: {}, Total loss: {}'.format(actor_loss, critic_loss, entropy_term, tot_loss))

    return  epi_rewards


def loss(q_vals, values, log_probs, entropy_term):
    advantage = tf.math.subtract(q_vals, values)
    actor_loss = tf.reduce_mean(tf.math.multiply(-log_probs, advantage))
    critic_loss = tf.reduce_mean(tf.math.square(advantage))
    tot_loss = actor_loss + critic_loss + 0.001 * entropy_term

    return tot_loss


def grad(model, q_vals, values, log_probs, entropy_term):
    with tf.GradientTape() as tape:
        loss_out = loss(q_vals, values, log_probs, entropy_term)

    gradients = tape.gradient(loss_out, model.trainable_variables)
    return loss_out, gradients


env = gym.make('CartPole-v1')
env.seed(1)
np.random.seed(2)
score = adv_ac(env, learning_rate)
dict = {'Score': score}
df = pd.DataFrame(dict)
plt.plot(df['Score'].rolling(window = 200).mean())
plt.show()