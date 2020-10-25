# Temporal Difference Learning

This repository contains implementations of different model free Temporal difference learning methods--an idea central and novel to Reinforcement Learning.

## Implemented Algorithms
 - SARSA 
   - This algorithm refers to an on-policy TD control which updates the Q-value function while following the $\epsilon$-greedy approach. This way an agent takes an action $A_{t+1}$ where $A_{t+1} = arg max_{a \in A} Q(S_{t+1}, a)$ and update the Q-value function as follows:
$Q(S_{t}, A_{t})\leftarrow Q(S_{t}, A_{t}) + \alpha(R_{t+1} + \gamma Q(S_{t+1}, A_{t+1}) - Q(S_{t}, A_{t}))$.
 - SARSA Lambda
   - This is a more generalized version of SARSA where we perform TD learning by n-steps instead of 1 which was the case for normal SARSA. In this case a weighted sum is applied to all possible n-step TD targets instead of choosing the best one. If the decay factor is $\lambda$, then the return for value updating is defined as $R_{t} = (1 - \lambda)\sum_{n=1}^{\infty} \lambda^{n-1}R_{t}$.
 - Q-learning
   - This algorithm refers to an on-policy TD control which updates the Q-value function while following the $\epsilon$-greedy approach. This way an agent takes an action $A_{t+1}$ where $A_{t+1} = arg max_{a \in A} Q(S_{t+1}, a)$ and update the Q-value function as follows:
$Q(S_{t}, A_{t})\leftarrow Q(S_{t}, A_{t}) + \alpha(R_{t+1} + \gamma max_{a \in A}Q(S_{t+1}, A_{t+1}) - Q(S_{t}, A_{t}))$.
 - Deep Q-Network (DQN)
   - This algorithm uses neural networks to approximate Q-value functions. It contains experience replay which improves data efficiency to update the model parameters. Neural network parameters are updated using gradient descent using MSE error to reduce the difference between the current Q-value with the targeted Q-value of the target Q-network. 

You will need to install [python](https://www.python.org), [numpy](https://numpy.org), [pandas](https://pandas.pydata.org), [matplotlib](https://matplotlib.org), [gym](https://gym.openai.com/), and [tensorflow 2.x](https://www.tensorflow.org/). If you have anaconda installed, run the following:
```bash
conda create -n envName python numpy pandas matplotlib 
```
This will create a conda environment with python, numpy, pandas, and matplotlib installed in it. Run `conda activate envName` to activate or `conda deactivate` to deactivate the environment. Then run the following commands to install [gym](https://gym.openai.com/) and [tensorflow 2.x](https://www.tensorflow.org/) after activating the `conda` environment.
```bash
pip install gym tensorflow
```
If you are not seeing the equations above, please install the [MathJax](https://chrome.google.com/webstore/detail/mathjax-plugin-for-github/ioemnmodlmafdkllaclgeombjnmnbima?hl=en) plugin for your chrome browser.
