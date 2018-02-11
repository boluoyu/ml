import os
import logging

from abc import abstractmethod
from collections import deque

logging.basicConfig(level=os.getenv('LOG_LEVEL', logging.WARNING))
logger = logging.getLogger(__name__)


class EnvironmentRunner:
    def __init__(self, experiment_name, model_dir, model_name, num_epochs, num_steps, agent, env_history_length):
        self.env_history = deque(maxlen=env_history_length)
        self.model_dir = model_dir
        self.experiment_name = experiment_name
        self.model_name = model_name
        self.num_steps = num_steps
        self.num_epochs = num_epochs

        self.agent = agent

    @abstractmethod
    def initialize_environment(self):
        raise NotImplementedError()

    @abstractmethod
    def step(self):
        raise NotImplementedError()

    def run(self):
        results = []

        for epoch in range(self.num_epochs):
            step_num, reward, done = self.run()
            results.append(dict(
                step_num=step_num,
                reward=reward,
                done=done
            ))

        return results

    def _run(self):
        action = None
        previous_observation = None
        info = None
        done = False
        reward = 0
        step_num = 0

        self.agent.initialize()
        observation = self.initialize_environment()

        action = self.agent.get_action(
            action=action,
            previous_observation=previous_observation,
            current_observation=observation,
            reward=reward,
            done=done,
            info=info
        )

        while True:
            step_num += 1
            previous_observation = observation

            observation, reward, done, info = self.step(action)
            self.agent.update_reward(reward, done)

            action = self.agent.get_action(
                action=action,
                previous_observation=previous_observation,
                current_observation=observation,
                reward=reward,
                done=done,
                info=info
            )

            self.store_env_step(
                action=action,
                previous_observation=previous_observation,
                current_observation=observation,
                reward=reward,
                done=done,
                info=info
            )

            if done:
                break

            elif step_num > self.num_steps:
                break

        return step_num, reward, done

    def store_env_step(self, action, current_observation, previous_observation, reward, done, info):
        self.env_history.append(
            {
                'observation': previous_observation,
                'action': action,
                'reward': reward,
                'next_observation': current_observation,
                'done': done
            }
        )

