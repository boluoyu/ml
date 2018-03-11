import gym
import universe

from ml.reinforcement.runner.environ.ml import MLEnvironmentRunner


class OpenAIGymMLEnvRunner(MLEnvironmentRunner):
    def initialize_environment(self):
        env = gym.make(self.env_name)
        return env

    def reset_environment(self):
        observation = self.env.reset()
        return observation

    def step(self, action):
        self.env.render()
        observation, reward, done, info = self.env.step(action)
        return observation, reward, done, info
