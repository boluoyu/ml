import numpy as np



class Transformer:
    def __init__(self, max_sequence_length, vocabulary, class_map, tokenizer):
        self._max_sequence_length = max_sequence_length
        self._vocabulary = vocabulary
        self._vocabulary_size = len(self._vocabulary)
        self._class_map = class_map
        self._num_classes = len(class_map)
        self._tokenizer = tokenizer

    def transform(self, x):
        tokens = self.tokenize(x)
        token_vector = np.array(self._vocabulary.encode(tokens))[:self._max_sequence_length]
        document_matrix = np.zeros(shape=(self._max_sequence_length))
        document_matrix[0:len(token_vector)] = token_vector
        return document_matrix

    def preprocess(self, text):
        return text

    def tokenize(self, x):
        return self._tokenizer.tokenize(x)


class TextTransformer(Transformer):
    name = 'text_transformer'

    def transform(self, text):
        text = self.preproccess(text)
        return super().transform(text)

    def preproccess(self, text):
        return text.lower()
