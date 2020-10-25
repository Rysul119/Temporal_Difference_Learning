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
df_qlearning_avg = pd.DataFrame({'Average': data_qlearning_avg})

data_sarsa = np.array(data_sarsa)
data_sarsa_avg = data_sarsa.mean(axis = 0)
df_sarsa_avg = pd.DataFrame({'Average': data_sarsa_avg})

qlearning_roll_avg = df_qlearning_avg['Average'].rolling(window = 200).mean()
qlearning_roll_std = df_qlearning_avg['Average'].rolling(window = 200).std()

sarsa_roll_avg = df_sarsa_avg['Average'].rolling(window = 200).mean()
sarsa_roll_std = df_sarsa_avg['Average'].rolling(window = 200).std()

plt.plot(qlearning_roll_avg, c = 'red')
plt.fill_between(x, qlearning_roll_avg + qlearning_roll_std, qlearning_roll_avg - qlearning_roll_std, facecolor = 'red', alpha = 0.7)

plt.plot(sarsa_roll_avg, c = 'blue')
plt.fill_between(x, sarsa_roll_avg + sarsa_roll_std, sarsa_roll_avg - sarsa_roll_std, facecolor = 'blue', alpha = 0.7)
plt.legend(['Q learning', 'SARSA'])

plt.xlabel('Episodes')
plt.ylabel('Averaged Rewards')
plt.title('Averaged Rewards for Q_learning, SARSA')
plt.show()