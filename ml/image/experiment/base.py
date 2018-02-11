import logging
import os

from ml.common.experiment.base import Experiment

logging.basicConfig(level=os.getenv('LOG_LEVEL', logging.INFO))
logger = logging.getLogger(__name__)


class ImageExperiment(Experiment):
    name = 'image_experiment'

    height = 64
    width = 64
    num_channels = 3

    def get_transformer(self, transformer_cls):
        return transformer_cls(width=self.width, height=self.height, num_channels=self.num_channels)

    def get_classifier(self, classifier_cls, transformer):
        return classifier_cls(
            transformer=transformer,
            model=self.get_model(),
            width=self.width,
            height=self.height,
            num_channels=self.num_channels
        )
