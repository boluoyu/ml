from abc import abstractmethod
from ml.common.experiment.base import Experiment

from ml.reinforcement.runner.environ.gym_ml import OpenAIGymMLEnvRunner


class ReinforcementLearningExperiment(Experiment):
    name = 'reinforcement_learning_experiment'

    env_history_length = 1000
    env_name = None

    def __init__(self, model_dir, batch_size, num_epochs, checkpoint_step_num, num_steps, training_mode=True, verbose=True):
        super().__init__(
            model_dir=model_dir,
            batch_size=batch_size,
            num_epochs=num_epochs,
            checkpoint_step_num=checkpoint_step_num,
            num_steps=num_steps,
            training_mode=training_mode,
            verbose=verbose
        )

        self.agent = self.get_agent()
        self.training_runner = self.get_training_runner()

    def get_training_runner(self):
        return OpenAIGymMLEnvRunner(
            env_history_length=self.env_history_length,
            checkpoint_step_num=self.checkpoint_step_num,
            env_name=self.env_name,
            experiment_name=self.name,
            model_dir=self.model_dir,
            agent=self.agent,
            num_epochs=self.num_epochs,
            num_steps=self.num_steps,
            batch_size=self.batch_size
        )

    @abstractmethod
    def get_agent(self):
        raise NotImplementedError()

    def run(self):
        return self.training_runner.run()
