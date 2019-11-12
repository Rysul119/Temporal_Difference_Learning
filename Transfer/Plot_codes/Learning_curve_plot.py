import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

RUNS = 30
EPISODES = 2e5
data_qlearning = []

for run in range(RUNS):
    path = 'Experiments_ubuntu/Q_learning_200k_'+str(run+1)+'_all.csv'
    df_qlearning = pd.read_csv(path)
    data_qlearning.append(df_qlearning['Reward'])

data_sarsa = []

for run in range(RUNS):
    path = 'Experiments_ubuntu/SARSA_200k_'+str(run+1)+'_all.csv'
    df_sarsa = pd.read_csv(path)
    data_sarsa.append(df_sarsa['Reward'])

x = np.arange(1, EPISODES+1, 1)

data_qlearning = np.array(data_qlearning)
data_qlearning_avg = data_qlearning.mean(axis = 0)
data_qlearning_std = data_qlearning.std(axis = 0)

data_sarsa = np.array(data_sarsa)
data_sarsa_avg = data_sarsa.mean(axis = 0)
data_sarsa_std = data_sarsa.std(axis = 0)


plt.plot(data_qlearning_avg, c = 'red')
plt.fill_between(x, data_qlearning_avg + data_qlearning_std, data_qlearning_avg - data_qlearning_std, facecolor = 'red', alpha = 0.7)

plt.plot(data_sarsa_avg, c = 'blue')
plt.fill_between(x, data_sarsa_avg + data_sarsa_std, data_sarsa_avg - data_sarsa_std, facecolor = 'blue', alpha = 0.7)
plt.legend(['Q learning', 'SARSA'])

plt.xlabel('Episodes')
plt.ylabel('Averaged Rewards')
plt.title('Averaged Rewards for Q_learning, SARSA')

plt.show()