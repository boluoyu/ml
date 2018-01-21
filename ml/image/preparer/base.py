from abc import abstractmethod


class Preparer:
    name = 'preparer'

    @abstractmethod
    def prepare(self):
        raise NotImplementedError()
