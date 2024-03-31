import gymnasium as gym
import numpy as np
import random
import tensorflow as tf
import keras
from collections import deque


class CartPoleDQLAgent:

    def __init__(self, gamma, epsilon, episode_count) -> None:
        self.train_env = gym.make("CartPole-v1")
        self.test_env = gym.make("CartPole-v1", render_mode="human")
        self.gamma = gamma
        self.epsilon = epsilon
        self.episode_count = episode_count
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
        shape_1, shape_2 = target_y.shape

        indices = np.zeros(shape=(shape_1, shape_2))
        indices[:, 0] = np.arange(shape_1)
        indices[:, 1] = self.selected_actions

        return keras.losses.mean_squared_error(tf.gather_nd(target_y, indices.astype(int)), tf.gather_nd(predict_y, indices.astype(int)))

    def select_action(self, state, episode):
        if episode < 1:
            return np.random.choice(self.action_dimension)

        if episode > 200:
            self.epsilon = 0.999 * self.epsilon

        if np.random.random() < self.epsilon:
            return np.random.choice(self.action_dimension)

        else:
            q_values = self.online_network.predict(state.reshape(1, 4))
            return np.random.choice(np.where(q_values[0, :] == np.max(q_values[0, :]))[0])

    def train_network(self):
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
            for i, (state, action, reward, next_state, terminal_state) in enumerate(random_batch):
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
                print("Target Network Updated")

    def train(self):
        for episode in range(self.episode_count):
            state, _ = self.train_env.reset()
            total_reward = 0
            terminal_state = False
            while not terminal_state:
                action = self.select_action(state, episode)
                next_state, reward, terminal_state, _, _ = self.train_env.step(
                    action)
                total_reward += reward
                self.replay_buffer.append(
                    (state, action, reward, next_state, terminal_state))
                self.train_network()
                state = next_state
            print(f"Episode: {episode}, Total Reward: {total_reward}")


def __main__():
    gamma = 1
    epsilon = 0.1
    episode_count = 1000
    agent = CartPoleDQLAgent(gamma, epsilon, episode_count)
    agent.train()
    agent.online_network.save("cartpole_dql.h5")


__main__()
