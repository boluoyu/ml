import re
import string
from argparse import ArgumentParser

import numpy as np
from keras.callbacks import TensorBoard, ReduceLROnPlateau, ModelCheckpoint
from keras.utils import to_categorical, plot_model
from keras_preprocessing.sequence import pad_sequences
from unicodedata import normalize

from ml.common.callback.keras import GradientDebugger
from ml.text.transformer.vocabulary import Vocabulary
from ml.text.experiment.neural_translation import NeuralTranslationExperiment


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


class WordTextTransformer:
    def __init__(self, source_vocabulary, target_vocabulary, max_source_length, max_target_length):
        self.source_vocabulary = source_vocabulary
        self.target_vocabulary = target_vocabulary
        self.max_source_length = max_source_length
        self.max_target_length = max_target_length

    def tokenize(self, text):
        return text.split()

    def transform_X(self, texts):
        X = list(map(self.tokenize, texts))
        X = list(map(self.source_vocabulary.encode, X))
        X = pad_sequences(X, maxlen=self.max_source_length, padding="post")
        X = np.array(X)
        return X

    def transform_y(self, texts):
        _y = []

        y = list(map(self.tokenize, texts))
        y = list(map(self.target_vocabulary.encode, y))
        y = pad_sequences(y, maxlen=self.max_target_length, padding="post")

        for sequence in y:
            _y.append(to_categorical(sequence, num_classes=len(self.target_vocabulary)))

        y = np.array(_y)
        y = y.reshape(y.shape[0], y.shape[1], len(self.target_vocabulary))
        return y


def get_tokens(texts):
    tokens = set()

    for text in texts:
        for token in text.split():
            tokens.add(token)

    return list(tokens)


def main(model_name, model_file_path, training_data_path, split_ratio=0.8):
    document = load_text(training_data_path)
    pairs = get_pairs(document)
    #pairs = pairs[:100]
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
    transformer = WordTextTransformer(source_vocabulary=source_vocabulary, max_source_length=max_source_length, max_target_length=max_target_length, target_vocabulary=target_vocabulary)

    split = int(split_ratio * len(pairs))
    train, test = pairs[0:split], pairs[split:]

    train_X = train[:, 0]
    train_y = train[:, 1]
    test_X = test[:, 0]
    test_y = test[:, 1]

    train_X = transformer.transform_X(train_X)
    train_y = transformer.transform_y(train_y)

    test_X = transformer.transform_X(test_X)
    test_y = transformer.transform_y(test_y)

    experiment = NeuralTranslationExperiment()
    model = experiment.get_model(target_vocab_size=len(target_vocabulary),
                                 source_vocab_size=len(source_vocabulary),
                                 source_timesteps=max_source_length,
                                 target_timesteps=max_target_length)

    plot_model(model, to_file='translator.png', show_shapes=True)

    callbacks = [
        GradientDebugger(),
        TensorBoard(log_dir='/tmp/translator'),
        ReduceLROnPlateau(factor=0.1, verbose=1),
        ModelCheckpoint(model_name + '.h5', monitor='val_loss', verbose=1, save_best_only=True, mode='min')
    ]

    model.fit(train_X, train_y, epochs=30, batch_size=64, validation_data=(test_X, test_y),
              callbacks=callbacks, verbose=2)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--model_name', required=True)
    parser.add_argument('--model_file_path', required=True)
    parser.add_argument('--training_data_file_path', required=True)

    args = parser.parse_args()
    main(args.model_name, args.model_file_path, args.training_data_file_path)
