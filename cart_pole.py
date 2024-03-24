import gymnasium as gym
import numpy as np
import time
import sys


env = gym.make("CartPole-v1", render_mode="human")
alpha = 0.1
gamma = 1
epsilon = 0.2
action_count = env.action_space.n
bin_counts = [30, 30, 30, 30]
upper_bounds = env.observation_space.high
lower_bounds = env.observation_space.low
cartVelocityMin = -3
cartVelocityMax = 3
poleAngleVelocityMin = -10
poleAngleVelocityMax = 10
upper_bounds[1] = cartVelocityMax
upper_bounds[3] = poleAngleVelocityMax
lower_bounds[1] = cartVelocityMin
lower_bounds[3] = poleAngleVelocityMin
reward_sum_per_episode = []
q_matrix = np.random.uniform(low=0, high=1, size=(
    bin_counts[0], bin_counts[1], bin_counts[2], bin_counts[3], action_count))


episode_count = 2000
time_steps = 10000


# Test simulation of the environment with random steps, finding typical velocity range
def find_typical_velocity_range():
    env = gym.make("CartPole-v1")
    cart_velocity_max = sys.float_info.min
    cart_velocity_min = sys.float_info.max
    pole_velocity_max = sys.float_info.min
    pole_velocity_min = sys.float_info.max

    for _ in range(episode_count):
        env.reset()
        env.render()
        for _ in range(time_steps):
            random_action = env.action_space.sample()
            observation, _, terminated, _, _ = env.step(
                random_action)
            cart_velocity_max = max(cart_velocity_max, observation[1])
            cart_velocity_min = min(cart_velocity_min, observation[1])
            pole_velocity_max = max(pole_velocity_max, observation[3])
            pole_velocity_min = min(pole_velocity_min, observation[3])

            if terminated:
                break

    env.close()
    return tuple([cart_velocity_max, cart_velocity_min, pole_velocity_max, pole_velocity_min])


# Q learning:

def state_index(state):
    cart_position = state[0]
    cart_velocity = state[1]
    pole_angle = state[2]
    pole_velocity = state[3]

    cart_position_bins = np.linspace(
        lower_bounds[0], upper_bounds[0], bin_counts[0])
    cart_velocity_bins = np.linspace(
        lower_bounds[1], upper_bounds[1], bin_counts[1])
    pole_angle_bins = np.linspace(
        lower_bounds[2], upper_bounds[2], bin_counts[2])
    pole_velocity_bins = np.linspace(
        lower_bounds[3], upper_bounds[3], bin_counts[3])

    cart_position_index = np.maximum(
        np.digitize(cart_position, cart_position_bins) - 1, 0)
    cart_velocity_index = np.maximum(
        np.digitize(cart_velocity, cart_velocity_bins) - 1, 0)
    pole_angle_index = np.maximum(
        np.digitize(pole_angle, pole_angle_bins) - 1, 0)
    pole_velocity_index = np.maximum(
        np.digitize(pole_velocity, pole_velocity_bins) - 1, 0)

    return tuple([cart_position_index, cart_velocity_index, pole_angle_index, pole_velocity_index])


def select_action(state, episode_index):
    global epsilon

    if episode_index < 500:
        return np.random.choice(action_count)

    random_number = np.random.random()

    if episode_index > 7000:
        epsilon = 0.99999 * epsilon

    if random_number < epsilon:
        return np.random.choice(action_count)
    else:
        indexes = state_index(state)
        return np.random.choice(
            np.where(q_matrix[indexes] == np.max(q_matrix[indexes]))[0])


def train():
    env = gym.make("CartPole-v1")

    for episode in range(episode_count):

        rewards_of_episode = 0

        (state_S, _) = env.reset()
        state_S = list(state_S)

        terminal_state = False

        while not terminal_state:

            state_S_index = state_index(
                state_S)

            action_A = select_action(
                state_S, episode)

            next_state, reward, terminal_state, _, _ = env.step(action_A)

            rewards_of_episode += reward

            next_state = list(next_state)
            next_state_index = state_index(
                next_state)
            q_max_next_state = np.max(q_matrix[next_state_index])

            if not terminal_state:
                error = reward + gamma * q_max_next_state - \
                    q_matrix[state_S_index + (action_A,)]
            else:
                error = reward - q_matrix[state_S_index + (action_A,)]

            q_matrix[state_S_index +
                     (action_A,)] = q_matrix[state_S_index + (action_A,)] + alpha * error

            state_S = next_state

        print(f"Sum of rewards, episode {episode}: {rewards_of_episode}")
        reward_sum_per_episode.append(rewards_of_episode)


def test():
    current_state, _ = env.reset()
    env.render()
    time_steps = 1000
    reward_sum = 0

    for time_step in range(time_steps):
        print(f"Time {time_step}")
        indexes = state_index(current_state)
        action = np.random.choice(
            np.where(q_matrix[indexes] == np.max(q_matrix[indexes]))[0])
        current_state, reward, terminated, _, _ = env.step(action)
        reward_sum += reward
        time.sleep(0.05)
        if (terminated):
            time.sleep(1)
            break
    return reward_sum


train()
test()
