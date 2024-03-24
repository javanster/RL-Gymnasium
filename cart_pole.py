import gymnasium as gym
import numpy as np
import time

env = gym.make("CartPole-v1", render_mode="human")

state, _ = env.reset()

print(state)
# State:
# index 0: Cart position
# index 1: Cart velocity
# index 2: Pole angle of rotation
# index 3: Angular velocity

print(env.observation_space.high)
# The max value of each of the states

print(env.observation_space.low)
# The min value of each of the states

# Test simulation of the environment with random steps:
episode_count = 10
time_steps = 100

for episode in range(episode_count):
    env.reset()
    print(f"Episode: {episode}")
    env.render()
    for time_step in range(time_steps):
        print(f"Time step: {time_step}")
        random_action = env.action_space.sample()
        observation, reward, terminated, truncated, info = env.step(
            random_action)
        time.sleep(0.01)
        if terminated:
            time.sleep(0.1)
            break

env.close()
