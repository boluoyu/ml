import logging
import numpy as np

from keras.callbacks import TensorBoard
from ml.reinforcement.agent.qlearn import QLearningAgent


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AutoencoderQlearnAgent(QLearningAgent):
    def __init__(self, actions, action_shape, observation_shape, epsilon, epsilon_min, epsilon_decay, gamma, autoencoder,
                 classifier, verbose=True):
        self.autoencoder = autoencoder

        super().__init__(actions=actions, action_shape=action_shape, epsilon=epsilon, epsilon_min=epsilon_min,
                         epsilon_decay=epsilon_decay, gamma=gamma, classifier=classifier, verbose=verbose, observation_shape=observation_shape)

    def fit(self, env_state_mini_batch):
        X = []
        y = []
        for env_state in env_state_mini_batch:
            observation = env_state['observation']

            action = env_state['action']
            reward = env_state['reward']
            next_observation = env_state['next_observation']
            done = env_state['done']

            if observation is None or next_observation is None:
                continue

            if done:
                target = reward
            else:
                target = (reward + self.gamma *
                          np.amax(self.classifier.predict(X=[next_observation])[0])) # discounted max future reward for the next state (Bellman EQ)

            target_future_discounted_reward = self._get_current_state_mapped_to_future_discounted_reward(observation, target, action)
            X.append(observation)
            y.append(target_future_discounted_reward)
        try:
            self.autoencoder.fit(
                X=X,
                y={'decoder': X , 'classifier': y},
                callbacks=[TensorBoard(log_dir='/tmp/autoencoder')]
            )
        except ValueError:
            logger.warning('Failure X %s y %s', X, y)
            return

        self.update_epsilon()
        logger.info('Finished training, updated epsilon %s threshold %s', self.epsilon, self.epsilon_min)

    def _get_current_state_mapped_to_future_discounted_reward(self, observation, target, action):
        action_ix = self.actions.index(action)
        target_future_discounted_rewards = self.classifier.predict(X=[observation])[0]
        target_future_discounted_rewards[action_ix] = target
        return target_future_discounted_rewards

    def predict(self, observation):
        predictions = self.classifier.predict(X=[observation])
        prediction = predictions[0]
        action_ix = np.argmax(prediction)
        action = self.actions[action_ix]
        return action


