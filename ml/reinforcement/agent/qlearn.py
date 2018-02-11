import numpy as np

from collections import deque

from ml.reinforcement.agent.base import Agent


class QLearningAgent(Agent):
    def __init__(self, actions, observation_shape, epsilon, epsilon_min, epsilon_decay, gamma, batch_size,
                 env_history_size, classifier, verbose=True):

        super(QLearningAgent, self).__init__()

        self.env_history = deque(maxlen=env_history_size)
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.gamma = gamma
        self.actions = actions
        self.observation_shape = observation_shape
        self.action_size = len(actions)
        self.classifier = classifier

        self._verbose = verbose
        self.batch_size = batch_size

    def get_action(self, action, current_observation, previous_observation, reward, done, info):
        random_decimal = np.random.rand()

        if random_decimal <= self.epsilon:
            action = self.actions.sample()
        else:
            action = self.predict(current_observation)

        return action

    def predict(self, observation):
        predictions = self.classifier.predict(X=observation)
        prediction = predictions[0]
        return np.argmax(prediction)

    def train(self, env_state_mini_batch):
        for env_state in env_state_mini_batch:
            observation = env_state['observation']
            action = env_state['action']
            reward = env_state['reward']
            next_observation = env_state['next_observation']
            done = env_state['done']

            if done:
                target = reward
            else:
                target = (reward + self.gamma *
                          np.amax(self.classifier.predict(next_observation)[0]))

            future_discounted_reward = self._predict_future_discounted_reward(observation, target, action)
            self.classifier.fit(observation, future_discounted_reward, epochs=1, verbose=0)

        self._update_epsilon()

    def _predict_future_discounted_reward(self, observation, target, action):
        target_future_discounted_rewards = self.classifier.predict(x=observation)
        target_future_discounted_rewards[0][action] = target
        return target_future_discounted_rewards

    def _update_epsilon(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
