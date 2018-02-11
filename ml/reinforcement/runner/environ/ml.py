import os
import logging

from ml.reinforcement.runner.environ.base import EnvironmentRunner

logging.basicConfig(level=os.getenv('LOG_LEVEL', logging.WARNING))
logger = logging.getLogger(__name__)

CHECKPOINT_FILE_TEMPLATE = '{experiment}_{model}_{epoch_num}_chkpt.hdf5'


class MLEnvironmentRunner(EnvironmentRunner):
    def __init__(self, experiment_name, model_dir, model_name, num_epochs, num_steps, agent, env_history_length,
                 batch_size):
        self.batch_size = batch_size

        super(MLEnvironmentRunner, self).__init__(experiment_name, model_dir, model_name, num_epochs, num_steps, agent,
                                                  env_history_length)

    def run(self):
        results = []

        for epoch in range(self.num_epochs):
            step_num, reward, done = self.run()
            results.append(dict(
                step_num=step_num,
                reward=reward,
                done=done
            ))

            if self.has_sufficient_training_data():
                self.agent.classifier.train()
                self.save_checkpoint(epoch)

        return results

    def save_checkpoint(self, epoch_num):
        checkpoint_file_name = self.get_checkpoint_file_name(epoch_num)
        self.agent.classifier.save(checkpoint_file_name)

    def get_checkpoint_file_name(self, epoch_num):
        checkpoint_file_name = CHECKPOINT_FILE_TEMPLATE.format(
            experiment=self.experiment_name,
            model=self.model_name,
            epoch_num=epoch_num,
        )

        return os.path.join(self.model_dir, checkpoint_file_name)

    def has_sufficient_training_data(self):
        return len(self.env_history) > self.batch_size
