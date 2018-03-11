import json
import os
import logging
import time
import matplotlib

matplotlib.use("AGG")
from matplotlib import pyplot as plt
import pandas as pd

from abc import abstractmethod
from collections import deque

logging.basicConfig(level=os.getenv('LOG_LEVEL', logging.WARNING))
logger = logging.getLogger(__name__)

RUN_STATISTICS_EPOCH_REWARD_FIGURE_NAME = '{experiment}_{env_name}_{epoch_num}_epoch_reward_{time}.png'
RUN_STATISTICS_EPOCH_STEPS_FIGURE_NAME = '{experiment}_{env_name}_{epoch_num}_epoch_steps_{time}.png'
RUN_STATISTICS_FILE_NAME = '{experiment}_{env_name}_{epoch_num}_{time}.json'


class EnvironmentRunner:
    def __init__(self, experiment_name, env_name, model_dir, num_epochs, num_steps, agent, env_history_length):
        self.env_history = deque(maxlen=env_history_length)
        self.model_dir = model_dir
        self.experiment_name = experiment_name
        self.num_steps = num_steps
        self.num_epochs = num_epochs
        self.env_name = env_name
        self.env = self.initialize_environment()

        self.agent = agent
        self.agent.initialize()
        self.run_statistics = []

    @abstractmethod
    def initialize_environment(self):
        raise NotImplementedError()

    @abstractmethod
    def reset_environment(self):
        raise NotImplementedError()

    @abstractmethod
    def step(self, action):
        raise NotImplementedError()

    def run(self):
        epoch = 0
        for epoch in range(self.num_epochs):
            step_num, reward, done = self._run()
            self.run_statistics.append(dict(
                num_steps=step_num,
                reward=reward,
                done=done
            ))

        self.compute_run_statistics(epoch)

    def compute_run_statistics(self, final_epoch):
        reward_plot = self.get_epoch_vs_reward_plot()
        steps_plot = self.get_epoch_vs_steps_plot()
        self.save_run_stats_figure(reward_plot, final_epoch, RUN_STATISTICS_EPOCH_REWARD_FIGURE_NAME)
        self.save_run_stats_figure(steps_plot, final_epoch, RUN_STATISTICS_EPOCH_STEPS_FIGURE_NAME)
        self.save_run_statistics(self.num_epochs)
        return self.run_statistics

    def _run(self):
        action = None
        previous_observation = None
        info = None
        done = False
        reward = 0
        step_num = 0

        self.agent.reset()
        observation = self.reset_environment()

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
            previous_reward = reward

            observation, reward, done, info = self.step(action)
            self.agent.update_reward(reward, done)
            reward += previous_reward

            previous_action = action
            action = self.agent.get_action(
                action=action,
                previous_observation=previous_observation,
                current_observation=observation,
                reward=reward,
                done=done,
                info=info
            )

            self.store_env_step(
                action=previous_action,
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
        if done:
            reward = -100

        if current_observation is None or previous_observation is None:
            return

        self.env_history.append(
            {
                'observation': previous_observation,
                'action': action,
                'reward': reward,
                'next_observation': current_observation,
                'done': done
            }
        )

    def save_run_statistics(self, epoch_num):
        statistics_file_name = self.get_run_statistics_file_name(epoch_num)

        logger.info('Writing statistics to file %s', statistics_file_name)
        with open(statistics_file_name, 'w') as f:
            f.write(json.dumps(self.run_statistics))

    def save_run_stats_figure(self, fig, epoch, file_template):
        figure_file_name = self.get_run_statistics_figure_name(epoch_num=epoch, file_template=file_template)
        fig.savefig(figure_file_name)

    def get_run_statistics_figure_name(self, epoch_num, file_template):
        epoch_ts = time.time()

        figure_name = file_template.format(
            experiment=self.experiment_name,
            env_name=self.env_name,
            epoch_num=epoch_num,
            time=epoch_ts
        )

        return os.path.join(self.model_dir, figure_name)

    def get_run_statistics_file_name(self, epoch_num):
        epoch_ts = time.time()

        statistics_file_name = RUN_STATISTICS_FILE_NAME.format(
            experiment=self.experiment_name,
            env_name=self.env_name,
            epoch_num=epoch_num,
            time=epoch_ts
        )

        return os.path.join(self.model_dir, statistics_file_name)

    def get_epoch_vs_steps_plot(self):
        run_stats = [(stats['epoch'], stats['num_steps']) for stats in self.run_statistics]
        cf = pd.DataFrame(run_stats, columns=['epoch_num', 'num_steps'])
        fig = cf.plot(x='epoch_num', y='num_steps').get_figure()
        return fig

    def get_epoch_vs_reward_plot(self):
        run_stats = [(stats['epoch'], stats['reward']) for stats in self.run_statistics]
        cf = pd.DataFrame(run_stats, columns=['epoch', 'reward'])
        fig = cf.plot(x='epoch', y='reward').get_figure()
        return fig




