import logging
import numpy as np
import random

from ml.reinforcement.agent.base import Agent

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MLAgent(Agent):
    name = 'ml_agent'

    def __init__(self, actions, action_shape, observation_shape, epsilon, epsilon_min, epsilon_decay, gamma, classifier,
                 verbose=True):

        super().__init__()

        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.gamma = gamma
        self.actions = actions
        self.observation_shape = observation_shape
        self.action_shape = action_shape
        self.action_size = len(actions)
        self.classifier = classifier

        self._verbose = verbose

    def get_action(self, action, current_observation, previous_observation, reward, done, info):
        random_decimal = np.random.rand()

        if random_decimal <= self.epsilon:
            action = random.choice(self.actions)
            logger.debug('Random choice action %s', action)
        else:
            action = self.predict(current_observation)
            logging.debug('Predicted action %s', action)

        return action

    def predict(self, observation):
        predictions = self.classifier.predict(X=[observation])
        prediction = predictions[0]
        return np.argmax(prediction)

    def update_epsilon(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
