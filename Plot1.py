import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('Experiments_ubuntu/SARSA_Lambda_1.csv')
df['rolling'] = df['Avg_rewards'].rolling(window = 500).mean()
var = df['Avg_rewards'].rolling(window = 500).std()
x = np.arange(1,100001,10)

plt.plot(x, df['rolling'], c = 'blue')
plt.fill_between(x, df['rolling']-var, df['rolling']+var, facecolor = 'blue' , alpha = 0.2)
plt.show()