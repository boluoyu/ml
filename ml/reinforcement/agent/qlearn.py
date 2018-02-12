import logging
import numpy as np

from ml.reinforcement.agent.ml import MLAgent

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class QLearningAgent(MLAgent):
    name = 'qlearning_agent'

    def fit(self, env_state_mini_batch):
        X = []
        y = []
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
                          np.amax(self.classifier.predict(X=[next_observation])[0])) # discounted max future reward for the next state (Bellman EQ)

            target_future_discounted_reward = self._get_current_state_mapped_to_future_discounted_reward(observation, target, action)
            X.append(observation)
            y.append(target_future_discounted_reward)

        X = np.array(X)
        y = np.array(y)
        self.classifier.fit(X=X, y=y, epochs=1, verbose=0)
        self.update_epsilon()
        logger.info('Finished training, updated epsilon %s threshold %s', self.epsilon, self.epsilon_min)

    def _get_current_state_mapped_to_future_discounted_reward(self, observation, target, action):
        target_future_discounted_rewards = self.classifier.predict(X=[observation])[0]
        target_future_discounted_rewards[action] = target
        return target_future_discounted_rewards
