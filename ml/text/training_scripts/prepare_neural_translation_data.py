from abc import abstractmethod, ABCMeta
import re
import os
import string
from argparse import ArgumentParser
from random import shuffle

import joblib
import numpy as np
from unicodedata import normalize

from ml.text.transformer.vocabulary import Vocabulary

import atexit
import json


class Sink(object):
    __metaclass__ = ABCMeta

    def filter(self, _):
        return True

    def transform(self, item):
        return item

    @abstractmethod
    def sink(self, item, **kwargs):
        raise NotImplementedError

    def receive(self, item, **kwargs):
        if self.filter(item):
            item = self.transform(item)
            return self.sink(item, **kwargs)


class JSONFileSink(Sink):
    def __init__(self, file_path):
        self.file_path = file_path
        self.file = open(self.file_path, 'a+')

        atexit.register(self._cleanup)

    def transform(self, item):
        item = {"source": item[0], "target": item[1]}
        return json.dumps(item)

    def sink(self, item, **kwargs):
        self.file.write(item + '\n')

    def flush(self):
        self.file.flush()

    def _cleanup(self):
        self.file.close()


def load_text(training_data_path):
    with open(training_data_path, mode='rt', encoding='utf-8') as f:
        data = f.read()
        return data


def get_pairs(document, reverse=True):
    lines = document.strip().split("\n")
    pairs = [line.split("\t") for line in lines]

    if reverse:
        reversed_pairs = []
        for pair in pairs:
            reversed_pairs.append([pair[1], pair[0]])

        pairs = reversed_pairs

    return pairs


def preprocess_pairs(pairs):
    cleaned_pairs = []

    re_punc = re.compile('[%s]' % re.escape(string.punctuation))
    re_print = re.compile('[^%s]' % re.escape(string.printable))

    for pair in pairs:
        cleaned_pair = []
        for phrase in pair:
            phrase = phrase.lower()
            phrase = normalize('NFD', phrase).encode('ascii', 'ignore')
            phrase = phrase.decode('UTF-8')  # tokenize on white space line = line.split()
            # convert to lowercase
            tokens = phrase.split()
            # remove punctuation from each token
            tokens = [re_punc.sub('', token) for token in tokens]
            # remove non-printable chars form each token
            tokens = [re_print.sub('', token) for token in tokens]
            # remove tokens with numbers in them
            cleaned_pair.append(' '.join(tokens))
        cleaned_pairs.append(cleaned_pair)

    return cleaned_pairs


def get_max_length(texts):
    max_length = 0

    for text in texts:
        tokens = text.split()
        if len(tokens) > max_length:
            max_length = len(tokens)

    return max_length


def get_tokens(texts):
    tokens = set()

    for text in texts:
        for token in text.split():
            tokens.add(token)

    return list(tokens)


def main(training_data_path, path_prefix, split=0.9):
    document = load_text(training_data_path)
    pairs = get_pairs(document)
    shuffle(pairs)
    # pairs = pairs[:100]
    print("before cleaning", pairs[0:10])
    pairs = preprocess_pairs(pairs)
    print("after cleaning", pairs[0:10])
    pairs = np.array(pairs)
    max_source_length = get_max_length(pairs[:, 0])
    max_target_length = get_max_length(pairs[:, 1])

    source_pairs = [pair[0] for pair in pairs]
    target_pairs = [pair[1] for pair in pairs]
    source_tokens = get_tokens(source_pairs)
    target_tokens = get_tokens(target_pairs)
    source_vocabulary = Vocabulary()
    source_vocabulary.fit(source_tokens)
    target_vocabulary = Vocabulary()
    target_vocabulary.fit(target_tokens)

    joblib.dump(source_vocabulary, os.path.join(path_prefix, "source_vocabulary.pkl"))
    joblib.dump(target_vocabulary, os.path.join(path_prefix, "target_vocabulary.pkl"))

    split = int(len(pairs) * split)
    training_pairs = pairs[:split]
    validation_pairs = pairs[:split]

    training_data_sink = JSONFileSink(os.path.join(path_prefix, "training.json"))
    validation_data_sink = JSONFileSink(os.path.join(path_prefix, "validation.json"))

    for pair in training_pairs:
        training_data_sink.receive(pair)

    for pair in validation_pairs:
        validation_data_sink.receive(pair)

    print("max source length ", max_source_length)
    print("max target length ", max_target_length)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--training_data_file_path', required=True)
    parser.add_argument('--path_prefix', required=True)

    args = parser.parse_args()
    main(args.training_data_file_path, args.path_prefix)
