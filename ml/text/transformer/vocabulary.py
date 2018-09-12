import numpy as np


class Vocabulary:
    dictionary = None

    def __init__(self):
        self.dictionary = None

    def fit(self, tokens):
        tokens = set(tokens)
        self.dictionary = {token: i for i, token in enumerate(tokens, 1)}
        self.inverted_dictionary = {i: token for i, token in enumerate(tokens, 1)}

    def encode(self, tokens):
        return [self.dictionary.get(token, 0) for token in tokens]

    def decode(self, encoded_tokens):
        return [self.inverted_dictionary.get(token, " ") for token in encoded_tokens]

    def __len__(self):
        return len(self.dictionary) + 1


class Preprocessor:
    def preprocess(self, text):
        return text.lower()


class Tokenizer:
    def tokenize(self, text):
        return text.split()


class WordTokenizer(Tokenizer):
    def __init__(self, texts, tokenizer=Tokenizer(), preprocessor=Preprocessor()):
        self.texts = texts
        self.tokenizer = tokenizer
        self.preprocessor = preprocessor

    def get_tokens(self):
        for text in self.texts:
            tokens = self.tokenize(text)
            for token in tokens:
                yield token

    def tokenize(self, text):
        text = self.preprocessor.preprocess(text)
        return self.tokenizer.tokenize(text)

    def get_max_length(self, tokenized_samples):
        document_lengths = sorted([len(tokens) for tokens in tokenized_samples])
        max_document_length = np.percentile(document_lengths, 95)
        return int(max_document_length)


class CharacterTokenizer(WordTokenizer):
    def tokenize(self, text):
        return [character for character in text]
