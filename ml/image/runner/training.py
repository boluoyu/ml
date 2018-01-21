import json
import logging
import os
import time

logging.basicConfig(level=os.getenv('LOG_LEVEL', logging.DEBUG))
logger = logging.getLogger(__name__)

TRAINING_STATISTICS_FILE_NAME = '{model}_{epoch_num}_{time}'
CHECKPOINT_FILE_TEMPLATE = '{model}_{epoch_num}_{step_num}'
VALIDATION_MESSAGE_TEMPLATE = 'Model: %s Epoch: %s Step: %s Training Loss: %s Val Loss: %s'


class TrainingRunner:
    def __init__(self, model_dir, classifier, data_generator, evaluator, batch_size, checkpoint_step_num,
                 validation_step_num, num_epochs, num_steps, verbose=True):

        self.model_dir = model_dir
        self.classifier = classifier
        self.model_name = classifier.name
        self.data_generator = data_generator

        self.batch_size = batch_size
        self.checkpoint_step_num = checkpoint_step_num
        self.validation_step_num = validation_step_num
        self.num_epochs = num_epochs
        self.num_steps = num_steps
        self.verbose = verbose

        self.training_statistics = []

    def run(self):
        for epoch_num in range(self.num_epochs):
            step_num = 0

            training_data_mini_batches = self.data_generator.get_training_data(self.batch_size, self.num_steps)

            while step_num < self.num_steps:
                step_num += 1

                try:
                    X, y = next(training_data_mini_batches)
                except StopIteration:
                    break

                X = list(map(self.classifier.transformer.transform, X))
                X = self.classifier.format_X(X)
                y = self.classifier.format_y(y)

                training_loss = self.classifier.fit(X, y)

                if self.should_compute_validation_loss(step_num):
                    self.training_statistics.append(
                        self.compute_validation_statistics(epoch_num, step_num, training_loss))

                if self.should_save_checkpoint(step_num):
                    self.save_checkpoint(epoch_num, step_num)

            self.save_checkpoint(epoch_num, step_num)

        self.save_training_statistics(self.num_epochs)

    def should_compute_validation_loss(self, step_num):
        return step_num % self.validation_step_num == 0

    def should_save_checkpoint(self, step_num):
        return step_num % self.checkpoint_step_num == 0

    def compute_validation_statistics(self, epoch, step_num, training_loss):
        val_X, val_y = self.data_generator.get_validation_data()
        validation_loss = self.classifier.compute_loss(val_X, val_y)
        logger.info(VALIDATION_MESSAGE_TEMPLATE.format(
            step_num=step_num,
            cv_loss=validation_loss,
            t_loss=training_loss)
        )

        return dict(
            training_loss=training_loss,
            validation_loss=validation_loss,
            epoch=epoch,
            step_num=step_num
        )

    def save_checkpoint(self, epoch_num, step_num):
        checkpoint_file_name = self.get_checkpoint_file_name(epoch_num, step_num)
        self.classifier.save(checkpoint_file_name)

    def save_training_statistics(self, epoch_num):
        training_statistics_file_name = self.get_training_statistics_file_name(epoch_num)

        logger.info('Writing training statistics to file %s', training_statistics_file_name)
        with open(training_statistics_file_name, 'w') as f:
            f.write(json.dumps(self.training_statistics))

    def get_checkpoint_file_name(self, epoch_num, step_num):
        checkpoint_file_name = CHECKPOINT_FILE_TEMPLATE.format(
            model=self.model_name,
            epoch_num=epoch_num,
            step_num=step_num
        )

        return os.path.join(self.model_dir, checkpoint_file_name)

    def get_training_statistics_file_name(self, epoch_num):
        epoch_ts = time.time()

        training_statistics_file_name = TRAINING_STATISTICS_FILE_NAME.format(
            model=self.model_name,
            epoch_num=epoch_num,
            time=epoch_ts
        )

        return os.path.join(self.model_dir, training_statistics_file_name)
