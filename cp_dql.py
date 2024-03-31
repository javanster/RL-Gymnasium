import time
import gymnasium as gym
import numpy as np
import random
import tensorflow as tf
import keras
from collections import deque


class CartPoleDQLAgent:
    """
    A class for an agent to solve the Cart Pole problem using Deep Q-Learning (DQL) with epsilon-greedy strategy.

    Inspired by an implementation by Aleksandar Haber: 
    https://aleksandarhaber.com/deep-q-networks-dqn-in-python-from-scratch-by-using-openai-gym-and-tensorflow-reinforcement-learning-tutorial/

    :param gamma: The discount rate, between 0 (myopic) and 1 (future rewards more important)
    :param epsilon: The exploration / exploitation trade-off. Between 0 (max exploitation, no exploration) and 1 (max exploration, no exploitation)
    :param episode_count: The number of episodes to use for training
    :param time_steps: The max number of time steps to run the testing for
    """

    def __init__(self, gamma, epsilon, episode_count, time_steps) -> None:
        self.train_env = gym.make("CartPole-v1")
        self.test_env = gym.make("CartPole-v1", render_mode="human")
        self.gamma = gamma
        self.epsilon = epsilon
        self.episode_count = episode_count
        self.time_steps = time_steps
        self.state_dimension = 4
        self.action_dimension = 2
        self.replay_buffer_size = 300
        self.batch_size = 100
        self.update_target_network_freq = 100
        self.counter_update_target_network = 0
        self.replay_buffer = deque(maxlen=self.replay_buffer_size)
        self.online_network = self.create_network()
        self.target_network = self.create_network()
        self.target_network.set_weights(self.online_network.get_weights())
        self.selected_actions = []

    def create_network(self):
        """
        Intitializes a neural network model for the agent to use. The model is a simple
        feedforward network with 2 hidden layers. Used both for the online and target
        network.
        """
        model = keras.Sequential()
        model.add(keras.layers.Dense(
            128, input_dim=self.state_dimension, activation='relu'))
        model.add(keras.layers.Dense(56, activation='relu'))
        model.add(keras.layers.Dense(
            self.action_dimension, activation='linear'))
        model.compile(loss=self.loss_function,
                      optimizer=keras.optimizers.RMSprop(), metrics=['accuracy'])
        return model

    def loss_function(self, target_y, predict_y):
        """
        Custom loss function for the neural network. Uses mean squared error.

        :param target_y: The target values
        :param predict_y: The predicted values
        """
        shape_1, shape_2 = target_y.shape

        indices = np.zeros(shape=(shape_1, shape_2))
        indices[:, 0] = np.arange(shape_1)
        indices[:, 1] = self.selected_actions

        return keras.losses.mean_squared_error(tf.gather_nd(target_y, indices.astype(int)), tf.gather_nd(predict_y, indices.astype(int)))

    def select_action_epsilon_greedy(self, state, episode):
        """
        Selects an action based on the epsilon-greedy strategy. After 200 episodes, the epsilon value
        is slowly decreased after each episode, ensuring more exploitation. Chooses a random action for
        the first 30 episodes, to ensure exploration. After that, has a epsilon chance of choosing a random
        action, otherwise chooses the action with the highest Q-value.

        :param state: The current state
        :param episode: The current episode
        """
        if episode > 200:
            self.epsilon = 0.999 * self.epsilon
        if episode < 30 or np.random.random() < self.epsilon:
            return np.random.choice(self.action_dimension)
        else:
            self.select_action_greedy(state)

    def select_action_greedy(self, state):
        """
        Selects an action based on the greedy strategy. Chooses the action with the highest Q-value.

        :param state: The current state
        """
        q_values = self.online_network.predict(state.reshape(1, 4))
        return np.random.choice(np.where(q_values[0, :] == np.max(q_values[0, :]))[0])

    def train_network(self):
        """
        Trains the online network using a batch from the replay buffer, a queue that stores a
        replay_buffer_size amount of tuples, containing the latest state, action, reward, next
        state, and terminal value. Only performs training if the replay buffer has reached a
        sufficient size. The target network is updated every 100 time steps.
        """
        if len(self.replay_buffer) >= self.batch_size:
            random_batch = random.sample(self.replay_buffer, self.batch_size)
            current_state_batch = np.zeros(
                (self.batch_size, self.state_dimension))
            next_state_batch = np.zeros(
                (self.batch_size, self.state_dimension))

            for i, tuple_s in enumerate(random_batch):
                current_state_batch[i, :] = tuple_s[0]
                next_state_batch[i, :] = tuple_s[3]

            q_next_state_target = self.target_network.predict(next_state_batch)
            q_current_state_online = self.online_network.predict(
                current_state_batch)

            input_for_network = current_state_batch
            output_for_network = np.zeros(
                (self.batch_size, self.action_dimension))

            self.selected_actions = []
            for i, (_, action, reward, _, terminal_state) in enumerate(random_batch):
                q_value = reward
                if not terminal_state:
                    q_value = reward + self.gamma * \
                        np.max(q_next_state_target[i])
                self.selected_actions.append(action)
                q_current_state_online[i, action] = q_value
                output_for_network[i] = q_current_state_online[i]
                output_for_network[i, action] = q_value

            self.online_network.fit(
                input_for_network, output_for_network, batch_size=self.batch_size, verbose=0, epochs=100)

            self.counter_update_target_network += 1

            if self.counter_update_target_network % self.update_target_network_freq == 0:
                self.target_network.set_weights(
                    self.online_network.get_weights())
                print("Target network updated")

    def train(self):
        """
        Trains the agent in the environment. The agent performs the action recommended by
        the online network given the current state, and updates the network based on the reward it
        receives after performing the action, using the Deep Q-Learning algorithm. Lastly, saves the
        trained online network to a file.
        """
        for episode in range(self.episode_count):
            state, _ = self.train_env.reset()
            total_reward = 0
            terminal_state = False
            while not terminal_state:
                action = self.select_action_epsilon_greedy(state, episode)
                next_state, reward, terminal_state, _, _ = self.train_env.step(
                    action)
                total_reward += reward
                self.replay_buffer.append(
                    (state, action, reward, next_state, terminal_state))
                self.train_network()
                state = next_state
            print(f"Episode: {episode}, Total Reward: {total_reward}")
        self.online_network.save("cartpole_dql.h5")

    def test(self):
        """
        Tests the agent in the simulated environment, given the trained online network.
        """
        self.online_network = keras.models.load_model(
            "cartpole_dql.h5", custom_objects={'loss_function': self.loss_function})
        state, _ = self.test_env.reset()
        self.test_env.render()
        reward_sum = 0

        for time_step in range(self.time_steps):
            print(f"Time {time_step}")
            action = self.select_action_greedy(state)
            state, reward, terminated, _, _ = self.test_env.step(
                action)
            reward_sum += reward
            time.sleep(0.05)
            if (terminated):
                time.sleep(1)
                break
        return reward_sum


def __main__():
    gamma = 1
    epsilon = 0.1
    episode_count = 1000
    time_steps = 10000
    agent = CartPoleDQLAgent(gamma, epsilon, episode_count, time_steps)
    agent.train()
    agent.online_network.save("cartpole_dql.h5")  # Remove this later


__main__()
