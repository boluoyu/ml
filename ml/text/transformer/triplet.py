import numpy as np
from nltk import TweetTokenizer

class Vocabulary:
    dictionary = None

    def __init__(self):
        self.dictionary = None

    def fit(self, tokens):
        tokens = set(tokens)
        self.dictionary = {token: i for i, token in enumerate(tokens, 1)}

    def encode(self, tokens):
        return [self.dictionary.get(token, 0) for token in tokens]

    def __len__(self):
        return len(self.dictionary) + 1


class TextTokenGenerator:
    def __init__(self, texts):
        self.texts = texts

    def get_tokens(self):
        for text in self.texts:
            tokens = self.tokenize(text)
            for token in tokens:
                yield token

    def tokenize(self, text):
        return [character for character in text]

    def get_max_length(self):
        tokenized_texts = [self.tokenize(text) for text in self.texts]
        document_lengths = sorted([len(tokenized_text) for tokenized_text in tokenized_texts])
        max_document_length = np.percentile(document_lengths, 99)
        return int(max_document_length)



class TripletTransformer:
    def __init__(self, vocabulary):
        self._vocabulary = vocabulary

    def transform_X(self, triplet):
        return tuple(self._transform(sample.text) for sample in triplet)

    def _transform(self, text):
        return np.array(self._vocabulary.encode(text))


class TripletOneHotTransformer(TripletTransformer):
    def _transform(self, text):
        vector = self._vocabulary.encode(text)
        one_hot = np.zeros((len(vector), len(self._vocabulary)))
        for i, code in enumerate(vector):
            one_hot[i, code] = 1
        return one_hot


class HierarchicalTripletTransformer(TripletTransformer):
    def __init__(self, *args, **kwargs):
        super(HierarchicalTripletTransformer, self).__init__(*args, **kwargs)
        self.tokenizer = TweetTokenizer()

    def _transform(self, text):
        tokens = self.tokenizer.tokenize(text)
        token_vectors = [self._vocabulary.encode(token) for token in tokens]
        max_token_length = max(len(vector) for vector in token_vectors)
        document_matrix = np.zeros((len(tokens), max_token_length))
        for v, vector in enumerate(token_vectors):
            document_matrix[v, :len(vector)] = vector
        return document_matrix

