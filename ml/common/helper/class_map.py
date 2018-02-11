import json
import logging

logger = logging.getLogger(__name__)


class ClassMapHelper:
    def load_class_map(self, class_map_file_path):
        logger.info('Loading class map from file path %s', class_map_file_path)

        with open(class_map_file_path) as f:
            data = json.load(f)
            return data

    def generate_class_map(self, classes):
        class_map = {cls: ix for ix, cls in enumerate(classes)}
        return class_map

    def save_class_map(self, class_map_file_path, class_map):
        logger.info('Writing class map to file path %s', class_map_file_path)

        with open(class_map_file_path, 'w') as f:
            f.write(json.dumps(class_map))
