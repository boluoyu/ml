import logging
import os

from argparse import ArgumentParser
from ds_common.interfaces.tldr import BanjoTLDRService
from requests import Session

from ml.image.extractor.tldr import TLDRExtractor
from ml.image.preparer.tldr import TLDRPreparer
from ml.image.service.download import MediaDownloader

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

CURATION_TYPE_ENUM_POSITIVE = 1


def add_x509_headers(headers):
    crt = os.environ['APICERTIFICATE']
    crt_scheme = os.environ['APISCHEME']

    headers.update(
        {
            'X-Banjo-X509-Scheme': crt_scheme,
            'X-SSL-AUTH': crt,
            'Content-Type': 'application/json',
            'Content-Language': 'en'
        }
    )

    return headers


def get_class_map():
    return {'fire': 0, 'flood': 1}


def get_banjo_session():
    session = Session()
    x509_headers = add_x509_headers({})
    session.headers.update(x509_headers)
    return session


def make_dir(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)


def main(training_data_directory, download_directory, categories):
    make_dir(training_data_directory)
    make_dir(download_directory)

    downloader = MediaDownloader(download_directory=download_directory, session=Session())
    tldr = BanjoTLDRService(hostname=os.environ['TLDR_HOSTNAME'], session=get_banjo_session())

    extractor = TLDRExtractor(tldr=tldr, output_dir=training_data_directory)
    extractor_file_path = extractor.extract(categories)

    preparer = TLDRPreparer(
        downloader=downloader,
        in_data_file_path=extractor_file_path,
        output_dir=training_data_directory,
        class_map=get_class_map()
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
