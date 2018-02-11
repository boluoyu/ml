import logging
import os
import time
from abc import abstractmethod

from ml.common.helper.class_map import ClassMapHelper
from ml.common.runner.training import TrainingRunner

logging.basicConfig(level=os.getenv('LOG_LEVEL', logging.INFO))
logger = logging.getLogger(__name__)


class Experiment:
    name = 'experiment'

    def __init__(self, class_map_file_path, classifier_cls, evaluator_cls, transformer_cls, model_dir, data_generator,
                 batch_size, num_epochs, checkpoint_step_num, validation_step_num, num_steps, training_mode=True,
                 verbose=True):
        self.training_mode = training_mode
        self.model_dir = self.get_model_dir(model_dir)
        self.data_generator = data_generator
        self.batch_size = batch_size
        self.checkpoint_step_num = checkpoint_step_num
        self.validation_step_num = validation_step_num
        self.num_steps = num_steps
        self.verbose = verbose
        self.num_epochs = num_epochs

        self.transformer = self.get_transformer(transformer_cls)
        self.classifier = self.get_classifier(classifier_cls, self.transformer)
        self.evaluator = evaluator_cls()

        self.class_map_helper = ClassMapHelper()
        self.class_map = self.class_map_helper.load_class_map(class_map_file_path)

        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)

        self.training_runner = self.get_training_runner()
        self.transformer = self.get_transformer(transformer_cls)

    def get_transformer(self, transformer_cls):
        return transformer_cls()

    def get_model_dir(self, model_dir):
        return os.path.join(model_dir, self.name + '_' + str(time.time()))

    def get_classifier(self, classifier_cls, transformer):
        return classifier_cls(
            transformer=transformer,
            model=self.get_model()
        )

    def get_training_runner(self):
        return TrainingRunner(
            class_map=self.class_map,
            experiment_name=self.name,
            model_dir=self.model_dir,
            classifier=self.classifier,
            evaluator=self.evaluator,
            data_generator=self.data_generator,
            batch_size=self.batch_size,
            checkpoint_step_num=self.checkpoint_step_num,
            validation_step_num=self.validation_step_num,
            num_steps=self.num_steps,
            verbose=self.verbose,
            num_epochs=self.num_epochs
        )

    def run_experiment(self):
        start = time.time()
        logger.info('Running experiment %s', self.name)
        self.training_runner.run()
        end = time.time()
        logger.info('Finished running experiment %s in %s seconds', self.name, end - start)

    @abstractmethod
    def get_model(self):
        raise NotImplementedError()
