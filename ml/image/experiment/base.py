import logging
import os
import time

from abc import abstractmethod

from ml.image.runner.training import TrainingRunner

logging.basicConfig(level=os.getenv('LOG_LEVEL', logging.INFO))
logger = logging.getLogger(__name__)


class Experiment:
    name = 'experiment'

    height = 64
    width = 64
    num_channels = 3

    def __init__(self, classifier_cls, evaluator_cls, transformer_cls, model_dir, data_generator,
                 batch_size, num_epochs, checkpoint_step_num, validation_step_num, num_steps, verbose=True):

        self.transformer = transformer_cls(width=self.width, height=self.height)
        self.evaluator = evaluator_cls()

        self.classifier = classifier_cls(
            transformer=self.transformer,
            model=self.get_model(),
            width=self.width,
            height=self.height,
            num_channels=self.num_channels
        )

        self.training_runner = TrainingRunner(
            experiment_name=self.name,
            model_dir=model_dir,
            classifier=self.classifier,
            evaluator=self.evaluator,
            data_generator=data_generator,
            batch_size=batch_size,
            checkpoint_step_num=checkpoint_step_num,
            validation_step_num=validation_step_num,
            num_steps=num_steps,
            verbose=verbose,
            num_epochs=num_epochs
        )

    def run_experiment(self):
        start = time.time()
        logger.info('Running experiment %s', self.name)
        self.training_runner.run()
        end = time.time()
        logger.info('Finished running experiment %s in %s seconds', self.name, end - start)

        start = time.time()
        logger.info('Running evaluation %s', self.evaluator.name)
        self.evaluator.evaluate()
        end = time.time()
        logger.info('Finished running evaluation %s in %s seconds', self.evaluator.name, end - start)

    @abstractmethod
    def get_model(self):
        raise NotImplementedError()
