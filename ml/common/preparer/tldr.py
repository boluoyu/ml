import json
import logging
import os

from abc import abstractmethod
from ds_common.sink.json_file_sink import JSONFileSink
from time import time

from ml.common.preparer.base import Preparer

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


class TLDRPreparer(Preparer):
    name = 'tld_preparer'

    def __init__(self, in_data_file_path, output_dir, class_map):
        self.class_map = class_map
        self.in_data_file_path = in_data_file_path
        self.out_data_file_path = self.get_output_file_name(output_dir)
        self.sink = JSONFileSink(self.out_data_file_path)

    def load_training_data_file(self, data_file_path):
        logger.info('Loading data from file %s', data_file_path)
        with open(data_file_path) as f:
            data = map(json.loads, f)
            for datum in data:
                yield datum

    def prepare(self):
        training_data = self.load_training_data_file(self.in_data_file_path)
        for datum in training_data:
            target = [0 for _ in range(len(self.class_map))]
            media = self.get_media(datum)

            main_category = datum['categories'][0]
            category_ix = self.class_map[main_category]
            target[category_ix] = 1

            datum['y'] = target
            datum['X'] = media

            self.sink.receive(datum)

        self.sink.flush()
        return self.out_data_file_path

    @abstractmethod
    def get_media(self, datum):
        raise NotImplementedError()

    def get_output_file_name(self, output_dir):
        return os.path.join(output_dir, '_'.join([self.name, str(time()) + '.json']))
