import logging
import os
import time
from abc import abstractmethod

logging.basicConfig(level=os.getenv('LOG_LEVEL', logging.INFO))
logger = logging.getLogger(__name__)


class Experiment:
    name = 'experiment'

    def __init__(self, model_dir, batch_size, num_epochs, checkpoint_step_num, num_steps, training_mode=True, verbose=True):
        self.training_mode = training_mode
        self.model_dir = self.get_model_dir(model_dir)
        self.batch_size = batch_size
        self.checkpoint_step_num = checkpoint_step_num
        self.num_steps = num_steps
        self.verbose = verbose
        self.num_epochs = num_epochs

        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)

    def get_model_dir(self, model_dir):
        return os.path.join(model_dir, self.name + '_' + str(time.time()))

    @abstractmethod
    def get_classifier(self):
        raise NotImplementedError()

    @abstractmethod
    def get_evaluator(self):
        raise NotImplementedError()

    @abstractmethod
    def get_model(self):
        raise NotImplementedError()

    @abstractmethod
    def get_training_runner(self):
        raise NotImplementedError()

    @abstractmethod
    def run(self):
        raise NotImplementedError()

    def run_experiment(self):
        start = time.time()
        logger.info('Running experiment %s', self.name)
        self.run()
        end = time.time()
        logger.info('Finished running experiment %s in %s seconds', self.name, end - start)
