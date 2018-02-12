import os
import logging
import random

from ml.reinforcement.runner.environ.base import EnvironmentRunner

logging.basicConfig(level=os.getenv('LOG_LEVEL', logging.WARNING))
logger = logging.getLogger(__name__)

CHECKPOINT_FILE_TEMPLATE = '{experiment}_{model}_{epoch_num}_chkpt.hdf5'


class MLEnvironmentRunner(EnvironmentRunner):
    def __init__(self, checkpoint_step_num, experiment_name, env_name, model_dir, num_epochs, num_steps, agent,
                 env_history_length, batch_size):
        self.checkpoint_step_num = checkpoint_step_num

        super().__init__(
            env_name=env_name,
            experiment_name=experiment_name,
            model_dir=model_dir,
            num_epochs=num_epochs,
            num_steps=num_steps,
            agent=agent,
            env_history_length=env_history_length
        )

        self.model_name = self.agent.classifier.name
        self.batch_size = batch_size

    def run(self):
        epoch = 0
        for epoch in range(self.num_epochs):
            logger.info('Epoch %s', epoch)
            step_num, reward, done = self._run()
            self.run_statistics.append(dict(
                epoch=epoch,
                num_steps=step_num,
                reward=reward,
                done=done
            ))

            if self.has_sufficient_training_data():
                logger.info('Training model')
                self.agent.fit(self.get_minibatch())

            if epoch % self.checkpoint_step_num == 0:
                logger.info('Saving checkpoint')
                self.save_checkpoint(epoch)

        self.compute_run_statistics(epoch)
        return self.run_statistics

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

    def get_minibatch(self):
        return random.sample(self.env_history, k=self.batch_size)
