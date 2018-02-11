import logging
import os
from argparse import ArgumentParser

from ds_common.interfaces.tldr import BanjoTLDRService
from ds_common.samples import SAMPLE_TYPE_ENUM_IMAGE
from requests import Session

from ml.common.helper.class_map import ClassMapHelper
from ml.common.extractor.tldr import TLDRExtractor
from ml.common.utils.auth import get_banjo_session
from ml.image.preparer.tldr import TLDRImagePreparer
from ml.image.service.download import MediaDownloader

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

CURATION_TYPE_ENUM_POSITIVE = 1


def make_dir(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)


def main(training_data_directory, download_directory, categories):
    make_dir(training_data_directory)
    make_dir(download_directory)

    class_map_helper = ClassMapHelper()
    class_map = class_map_helper.generate_class_map(categories)

    downloader = MediaDownloader(download_directory=download_directory, session=Session())
    tldr = BanjoTLDRService(hostname=os.environ['TLDR_HOSTNAME'], session=get_banjo_session())

    extractor = TLDRExtractor(tldr=tldr, output_dir=training_data_directory, sample_type_enum=SAMPLE_TYPE_ENUM_IMAGE)
    extractor_file_path = extractor.extract(categories)

    preparer = TLDRImagePreparer(
        in_data_file_path=extractor_file_path,
        output_dir=training_data_directory,
        downloader=downloader,
        class_map=class_map
    )
    preparer_file_path = preparer.prepare()

    logger.info('Data can be found at %s', preparer_file_path)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--training_data_directory', required=True)
    parser.add_argument('--download_directory', required=True)
    parser.add_argument('--categories', required=True)

    args = parser.parse_args()
    main(args.training_data_directory, args.download_directory, args.categories.split(','))
