import logging
import os
from time import time

from ds_common.sink.json_file_sink import JSONFileSink

from ml.common.extractor.base import Extractor

logger = logging.getLogger(__name__)

CURATION_TYPE_ENUM_POSITIVE = 1
EXTENSION_JSON = '.json'


class TLDRExtractor(Extractor):
    name = 'tldr_extractor'

    def __init__(self, tldr, output_dir, sample_type_enum, limit=10):
        self.limit = limit
        self.tldr = tldr
        self.sample_type_enum = sample_type_enum
        self.out_data_file_path = self.get_output_file_name(output_dir)
        self.sink = JSONFileSink(self.out_data_file_path)

    def extract(self, categories):
        for category in categories:
            num_samples = 0
            logger.info('Querying for category %s', category)
            samples = self.tldr.get_samples(
                category=category,
                type_enum=self.sample_type_enum,
                curation_type_enum=CURATION_TYPE_ENUM_POSITIVE
            )

            for sample in samples:
                self.sink.receive(item=sample)
                num_samples += 1

                if self.limit and num_samples >= self.limit:
                    break

        self.sink.flush()
        return self.out_data_file_path

    def get_output_file_name(self, output_dir):
        return os.path.join(output_dir, '_'.join([self.name, str(time()) + EXTENSION_JSON]))
