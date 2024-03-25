import gymnasium as gym
import numpy as np
import time
import sys


class CartPoleQLearningAgent:

    def __init__(self, alpha, gamma, epsilon, bin_counts, episode_count, time_steps) -> None:
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.train_env = gym.make("CartPole-v1")
        self.test_env = gym.make("CartPole-v1", render_mode="human")

        upper_bounds = self.train_env.observation_space.high
        lower_bounds = self.train_env.observation_space.low
        cartVelocityMin = -3
        cartVelocityMax = 3
        poleAngleVelocityMin = -10
        poleAngleVelocityMax = 10
        upper_bounds[1] = cartVelocityMax
        upper_bounds[3] = poleAngleVelocityMax
        lower_bounds[1] = cartVelocityMin
        lower_bounds[3] = poleAngleVelocityMin

        # Necessary since state space is continuous
        self.state_bins = self.get_bins(lower_bounds, upper_bounds, bin_counts)
        self.action_count = self.train_env.action_space.n
        self.q_matrix = np.random.uniform(low=0, high=1, size=(
            bin_counts[0], bin_counts[1], bin_counts[2], bin_counts[3], self.action_count))
        self.reward_sum_per_episode = []
        self.episode_count = episode_count
        self.time_steps = time_steps

    def get_bins(self, lower_bounds, upper_bounds, bin_counts):
        cart_position_bins = np.linspace(
            lower_bounds[0], upper_bounds[0], bin_counts[0])
        cart_velocity_bins = np.linspace(
            lower_bounds[1], upper_bounds[1], bin_counts[1])
        pole_angle_bins = np.linspace(
            lower_bounds[2], upper_bounds[2], bin_counts[2])
        pole_velocity_bins = np.linspace(
            lower_bounds[3], upper_bounds[3], bin_counts[3])

        return [cart_position_bins, cart_velocity_bins, pole_angle_bins, pole_velocity_bins]

    # Test simulation of the environment with random steps, finding typical velocity range
    def find_typical_velocity_range(self):
        cart_velocity_max = sys.float_info.min
        cart_velocity_min = sys.float_info.max
        pole_velocity_max = sys.float_info.min
        pole_velocity_min = sys.float_info.max

        for _ in range(self.episode_count):
            self.train_env.reset()
            self.train_env.render()
            for _ in range(self.time_steps):
                random_action = self.train_env.action_space.sample()
                observation, _, terminated, _, _ = self.train_env.step(
                    random_action)
                cart_velocity_max = max(cart_velocity_max, observation[1])
                cart_velocity_min = min(cart_velocity_min, observation[1])
                pole_velocity_max = max(pole_velocity_max, observation[3])
                pole_velocity_min = min(pole_velocity_min, observation[3])

                if terminated:
                    break

        self.train_env.close()
        return tuple([cart_velocity_max, cart_velocity_min, pole_velocity_max, pole_velocity_min])

    def get_state_index(self, state):
        """
        Returns a tuple of the index of the bin (discrete) which matches the state,
        since the state space is continuous (incompatible with the Q-matrix)
        """
        cart_position = state[0]
        cart_velocity = state[1]
        pole_angle = state[2]
        pole_velocity = state[3]

        cart_position_index = np.maximum(
            np.digitize(cart_position, self.state_bins[0]) - 1, 0)
        cart_velocity_index = np.maximum(
            np.digitize(cart_velocity, self.state_bins[1]) - 1, 0)
        pole_angle_index = np.maximum(
            np.digitize(pole_angle, self.state_bins[2]) - 1, 0)
        pole_velocity_index = np.maximum(
            np.digitize(pole_velocity, self.state_bins[3]) - 1, 0)

        return tuple([cart_position_index, cart_velocity_index, pole_angle_index, pole_velocity_index])

    def select_action(self, state, episode_index):
        if episode_index < 500:
            return np.random.choice(self.action_count)

        random_number = np.random.random()

        if episode_index > 7000:
            self.epsilon = 0.99999 * self.epsilon

        if random_number < self.epsilon:
            return np.random.choice(self.action_count)
        else:
            indexes = self.get_state_index(state)
            return np.random.choice(
                np.where(self.q_matrix[indexes] == np.max(self.q_matrix[indexes]))[0])

    def train(self):

        for episode in range(self.episode_count):

            rewards_of_episode = 0

            (state_S, _) = self.train_env.reset()
            state_S = list(state_S)

            terminal_state = False

            while not terminal_state:

                state_S_index = self.get_state_index(
                    state_S)

                action_A = self.select_action(
                    state_S, episode)

                next_state, reward, terminal_state, _, _ = self.train_env.step(
                    action_A)

                rewards_of_episode += reward

                next_state = list(next_state)
                next_state_index = self.get_state_index(
                    next_state)
                q_max_next_state = np.max(self.q_matrix[next_state_index])

                if not terminal_state:
                    error = reward + self.gamma * q_max_next_state - \
                        self.q_matrix[state_S_index + (action_A,)]
                else:
                    error = reward - self.q_matrix[state_S_index + (action_A,)]

                self.q_matrix[state_S_index +
                              (action_A,)] = self.q_matrix[state_S_index + (action_A,)] + self.alpha * error

                state_S = next_state

            print(f"Sum of rewards, episode {episode}: {rewards_of_episode}")
            self.reward_sum_per_episode.append(rewards_of_episode)

    def test(self):
        current_state, _ = self.test_env.reset()
        self.test_env.render()
        time_steps = 1000
        reward_sum = 0

        for time_step in range(time_steps):
            print(f"Time {time_step}")
            indexes = self.get_state_index(current_state)
            action = np.random.choice(
                np.where(self.q_matrix[indexes] == np.max(self.q_matrix[indexes]))[0])
            current_state, reward, terminated, _, _ = self.test_env.step(
                action)
            reward_sum += reward
            time.sleep(0.05)
            if (terminated):
                time.sleep(1)
                break
        return reward_sum


def __main__():
    episode_count = 10000
    time_steps = 10000

    alpha = 0.1
    gamma = 1
    epsilon = 0.2
    bin_counts = [30, 30, 30, 30]

    cart_pole_q_learning_agent = CartPoleQLearningAgent(
        alpha, gamma, epsilon, bin_counts, episode_count, time_steps)

    cart_pole_q_learning_agent.train()
    cart_pole_q_learning_agent.test()


__main__()
