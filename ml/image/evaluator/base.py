from abc import abstractmethod


class Evaluator:
    name = 'evaluator'

    @abstractmethod
    def evaluate(self):
        raise NotImplementedError()
