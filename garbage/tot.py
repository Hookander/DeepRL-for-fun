import gymnasium as gym


env = gym.make('CarRacing-v2')
#env = gym.make('CartPole-v1')

act = env.action_space.shape
obs = env.reset()[0].shape

print(act)
print('\n')
print(obs)