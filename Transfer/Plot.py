import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = []
for i in range(31):
    name = 'New_experiments/Q_learning_'+str(i+1)+'_all.csv'
    df = pd.read_csv(name)
    avg_roll = df['Reward']
    data.append(avg_roll)

data = np.array(data)
data_avg = data.mean(axis = 0)
data_std = data.std(axis = 0)
dict = {'Average_data': data_avg}
df = pd.DataFrame(dict)
df['roll'] = df['Average_data'].rolling(window = 500).mean()
var = df['Average_data'].rolling(window = 500).std()
x = np.arange(1, 100001, 1)

data = []
for i in range(31):
    name = 'New_experiments/Q_learning_'+str(i+1)+'.csv'
    df = pd.read_csv(name)
    avg_roll = df['Avg_rewards']
    data.append(avg_roll)

data = np.array(data)
data_avg = data.mean(axis = 0)
data_std = data.std(axis = 0)
dict = {'Average_data': data_avg}
df1 = pd.DataFrame(dict)
df1['roll'] = df1['Average_data'].rolling(window = 200).mean()
var1 = df1['Average_data'].rolling(window = 200).std()
x1 = np.arange(1, 100001, 10)

plt.plot(x, df['rolling'], c = 'blue')
plt.fill_between(x, df['rolling']-var, df['rolling']+var, facecolor = 'blue', alpha = 0.2)
plt.plot(x, df1['rolling'], c = 'red')
plt.fill_between(x, df1['rolling']-var1, df1['rolling']+var1, facecolor = 'red', alpha = 0.2)
plt.show()
