import re
import string
from argparse import ArgumentParser

import numpy as np
from keras.optimizers import Adam
from keras.utils import to_categorical, plot_model
from keras_preprocessing.sequence import pad_sequences

from ml.common.callback.keras import GradientDebugger
from ml.text.transformer.vocabulary import Vocabulary
from ml.text.experiment.text_generation_model import TextGenerationExperiment


def load_text(training_data_path):
    with open(training_data_path) as f:
        data = f.read()
        return data


def get_cleaned_document(document):
    # replace '--' with a space ' '
    doc = document.replace('--', ' ')
    # split into tokens by white space
    tokens = doc.split()
    # prepare regex for char filtering
    re_punc = re.compile('[%s]' % re.escape(string.punctuation))  # remove punctuation from each word
    tokens = [re_punc.sub('', w) for w in tokens]
    # remove remaining tokens that are not alphabetic
    tokens = [word for word in tokens if word.isalpha()]
    # make lower case
    tokens = [word.lower() for word in tokens]
    return tokens


def get_sequences(tokens, max_sequence_length=51):
    sequences = []

    for n in range(max_sequence_length, len(tokens)):
        sequence = tokens[n - max_sequence_length:n]
        sequences.append(sequence)

    return sequences


class WordTextTransformer:
    def __init__(self, vocabulary):
        self.vocabulary = vocabulary

    def transform(self, tokens):
        return self.vocabulary.encode(tokens)


def predict_text(model, seed_tokens, transformer, max_sequence_length, num_tokens):
    in_tokens = seed_tokens
    vocabulary = transformer.vocabulary

    for _ in range(num_tokens):
        encoded = transformer.transform(in_tokens)
        encoded = pad_sequences([encoded], maxlen=max_sequence_length, truncating='pre')
        y_pred = model.predict(encoded, verbose=1)
        y_pred = np.argmax(y_pred, axis=1)[0]
        predicted_token = vocabulary.decode([y_pred])[0]
        print("in %s predicted %s" % (in_tokens, predicted_token))
        in_tokens.append(predicted_token)

    return in_tokens


def main(model_name, model_file_path, training_data_path, epochs=100):
    MAX_SEQUENCE_LENGTH = 51
    document = load_text(training_data_path)
    tokens = get_cleaned_document(document)
    print(tokens[:200])
    print('Total Tokens: %d' % len(tokens))
    print('Unique Tokens: %d' % len(set(tokens)))
    sequences = get_sequences(tokens, max_sequence_length=MAX_SEQUENCE_LENGTH)

    vocabulary = Vocabulary()
    vocabulary.fit(tokens)
    transformer = WordTextTransformer(vocabulary)
    sequences = list(map(transformer.transform, sequences))
    sequences = np.array(sequences)
    X, y = sequences[:, :-1], sequences[:, -1]
    y = to_categorical(y)

    experiment = TextGenerationExperiment()
    model = experiment.get_model(max_document_length=MAX_SEQUENCE_LENGTH - 1, vocabulary_size=len(vocabulary))
    model.compile(optimizer=Adam(), loss="categorical_crossentropy", metrics=["accuracy"])
    plot_model(model, to_file=model_name + '.png')

    callbacks = [
        GradientDebugger(),
    ]

    model.fit(x=X, y=y, epochs=epochs, verbose=1, batch_size=128, callbacks=callbacks)
    predict_text(model=model, transformer=transformer, seed_tokens=sequences[0], num_tokens=50, max_sequence_length=50)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--model_name', required=True)
    parser.add_argument('--model_file_path', required=True)
    parser.add_argument('--training_data_file_path', required=True)

    args = parser.parse_args()
    main(args.model_name, args.model_file_path, args.training_data_file_path)
