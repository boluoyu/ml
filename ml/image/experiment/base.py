import logging
import os

from ml.common.helper.class_map import ClassMapHelper
from ml.common.experiment.base import Experiment
from ml.common.runner.training import TrainingRunner
from ml.image.transformer.image import ImageTransformer

logging.basicConfig(level=os.getenv('LOG_LEVEL', logging.INFO))
logger = logging.getLogger(__name__)


class ImageExperiment(Experiment):
    name = 'image_experiment'

    height = 64
    width = 64
    num_channels = 3

    def __init__(self, model_dir, data_generator, batch_size, num_epochs, checkpoint_step_num, validation_step_num,
                 num_steps, class_map_file_path=None, training_mode=True, verbose=True):
        self.transformer = ImageTransformer(height=self.height, width=self.width, num_channels=self.num_channels)
        self.evaluator = self.get_evaluator()
        self.classifier = self.get_classifier()
        self.training_runner = self.get_training_runner()
        self.class_map = self.get_class_map(class_map_file_path)
        self.data_generator = data_generator
        self.validation_step_num = validation_step_num

        super().__init__(
            model_dir=model_dir,
            batch_size=batch_size,
            num_epochs=num_epochs,
            checkpoint_step_num=checkpoint_step_num,
            num_steps=num_steps,
            training_mode=training_mode,
            verbose=verbose
        )

    def get_class_map(self, class_map_file_path):
        self.class_map_helper = ClassMapHelper()
        self.class_map = self.class_map_helper.load_class_map(class_map_file_path)

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

    def run(self):
        return self.training_runner.run()
