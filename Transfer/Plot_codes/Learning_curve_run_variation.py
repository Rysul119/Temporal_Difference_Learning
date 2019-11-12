import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

RUNS = 10
EPISODES = 2e5

for count in range(1,4):
    data_q_learning = []
    for run in range(RUNS * count):
        path = 'Experiments_ubuntu/Q_learning_200k_'+str(run+1)+'_all.csv'
        df_q_learning = pd.read_csv(path)
        data_q_learning.append(df_q_learning['Reward'])

    x = np.arange(1, EPISODES+1, 1)

    data_q_learning = np.array(data_q_learning)
    data_q_learning_avg = data_q_learning.mean(axis = 0)
    data_q_learning_std = data_q_learning.std(axis = 0)


    plt.plot(data_q_learning)
    #plt.fill_between(x, data_sarsa_avg + data_sarsa_std, data_sarsa_avg - data_sarsa_std, facecolor = 'blue', alpha = 0.7)


plt.xlabel('Episodes')
plt.ylabel('Averaged Rewards')
plt.title('Averaged Rewards for Q_learning')
plt.legend(['Run count = 10', 'Run count = 20', 'Run count = 30'])

plt.show()