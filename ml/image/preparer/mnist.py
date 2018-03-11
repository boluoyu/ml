from ml.common.preparer.tldr import TLDRPreparer


class TLDRImagePreparer(TLDRPreparer):
    def __init__(self, in_data_file_path, output_dir, class_map, downloader):
        self.downloader = downloader

        super().__init__(in_data_file_path, output_dir, class_map)

    def get_media(self, datum):
        return self.downloader.download_data_to_disk(datum['image_url'])
