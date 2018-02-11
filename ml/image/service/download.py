import logging
import os

from os.path import basename

logging.basicConfig(level=os.getenv('LOG_LEVEL', logging.INFO))
logger = logging.getLogger(__name__)


class DownloadError(Exception):
    name = 'download_error'


class MediaDownloader:
    def __init__(self, download_directory, session, timeout=10, max_retries=3):
        self._download_directory = download_directory
        self._session = session
        self._timeout = timeout
        self._max_retries = max_retries

    def _get_content(self, media_url):
        while True:
            retries = 0

            while retries <= self._max_retries:
                try:
                    resp = self._session.get(media_url, timeout=self._timeout)
                    resp.raise_for_status()
                    return resp.content
                except:
                    logger.debug('Failed to download url %s', media_url, exec_info=True)
                    retries += 1

            raise DownloadError('Failed to download url %s', media_url)

    def download_data_to_disk(self, media_url):
        content = self._get_content(media_url)
        identifier = basename(media_url)
        file_name = self._get_file_name(identifier=identifier)

        if not os.path.exists(file_name):
            logger.info('Downloading media to file %s', file_name)
            file_name = self.write_data_to_disk(content, identifier=identifier)
        else:
            logger.info('Skipping over already downloaded media at %s', file_name)

        return file_name

    def write_data_to_disk(self, content, identifier):
        if not isinstance(content, bytes):
            content = bytes(content, 'utf-8')

        file_name = self._get_file_name(identifier=identifier)

        with open(file_name, 'wb') as f:
            f.write(content)

        return file_name

    def _get_file_name(self, identifier):
        file_name = os.path.join(self._download_directory, identifier)
        return file_name
