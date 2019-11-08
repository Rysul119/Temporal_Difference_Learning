import gym
env = gym.make('Taxi-v2')
state = env.reset()
reward = None
counter = 0

while reward!=20:
    state, reward, done, info = env.step(env.action_space.sample())
    counter+=1

print(counter)
