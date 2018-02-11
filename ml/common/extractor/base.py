from abc import abstractmethod


class Extractor:
    name = 'extractor'

    @abstractmethod
    def extract(self, categories):
        raise NotImplementedError()
