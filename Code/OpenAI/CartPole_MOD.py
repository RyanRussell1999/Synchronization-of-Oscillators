import gym
import numpy as np

T = 0.02

env = gym.make('CartPole-v0')
for i_episode in range(1):
    observation = env.reset()
    for t in range(10000):
        env.render()
        # print(observation[0])
        theta = observation[2]
        theta_dot = observation[3]
        x = observation[0]
        x_dot = observation[1]
        
        K = np.array([-23.5380, -5.1391, -0.7339, -1.2783]) # MATLAB Values
        
        # Calculate Control Input
        x_vec = np.array([[theta], [theta_dot], [x], [x_dot]])
        
        # Calculate Control Input
        if t == 0:
            u = np.array([(23.5380 * theta + 5.1391 * theta_dot + 0.7339 * x + 1.2783 * x_dot) + 0.5*np.sin(0.5*T*t)])
        else:
            u = (23.5380 * theta + 5.1391 * theta_dot + 0.7339 * x + 1.2783 * x_dot) + 0.5*np.sin(0.5*T*t)
            
        action = env.action_space.sample()
        observation, reward, done, info = env.step(u)
        if done:
            print("Episode finished after {} timesteps".format(t+1))
            break
env.close()