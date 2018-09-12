from argparse import ArgumentParser

import numpy as np
from keras.callbacks import TensorBoard
from keras.optimizers import Adam
from keras.utils import to_categorical, plot_model
from keras_preprocessing.sequence import pad_sequences

from ml.common.callback.keras import GradientDebugger
from ml.text.experiment.word_level_language_model import WordLevelLanguageModelExperiment

from ml.text.transformer.vocabulary import Vocabulary


def get_training_text(training_data_file_path):
    f = open(training_data_file_path, 'r')
    data = f.read()
    f.close()
    return data


def preprocess_text(text):
    tokens = text.lower().split(' ')
    return ' '.join(tokens)


def get_sequences(tokens, seq_length=2):
    sequences = []

    for i in range(seq_length, len(tokens)):
        sequence = tokens[i-1:i+seq_length]
        sequences.append(sequence)

    return sequences


class WordTextTransformer:
    def __init__(self, vocabulary):
        self.vocabulary = vocabulary

    def transform(self, tokens):
        tokens = [token for token in tokens]
        encoded = self.vocabulary.encode(tokens)
        return encoded


def predict_text(model, seed_text, transformer, max_sequence_length, num_tokens):
    all_predicted_text = seed_text
    in_text = seed_text
    vocabulary = transformer.vocabulary

    for _ in range(num_tokens):
        encoded = transformer.transform(in_text)
        encoded = pad_sequences([encoded], maxlen=max_sequence_length, truncating='pre')
        y_pred = model.predict(encoded, verbose=1)
        y_pred = np.argmax(y_pred, axis=1)[0]
        predicted_token = vocabulary.decode([y_pred])[0]
        print("in %s predicted %s" % (in_text, predicted_token))
        in_text.append(predicted_token)
        all_predicted_text.append(predicted_token)

    return all_predicted_text


def main(training_data_file_path, model_name, model_file_path):
    MAX_DOCUMENT_LENGTH = 2
    training_data = get_training_text(training_data_file_path)
    training_data = preprocess_text(training_data)

    tokens = sorted(list(set(token for token in training_data.split())))
    vocabulary = Vocabulary()
    vocabulary.fit(tokens)
    tokens = [token for token in training_data.split()]
    sequences = get_sequences(tokens)
    print(len(sequences), sequences[0])

    transformer = WordTextTransformer(vocabulary)
    encoded_sequences = list(map(transformer.transform, sequences))
    encoded_sequences = pad_sequences(encoded_sequences, maxlen=3, padding="post")
    encoded_sequences = np.array(encoded_sequences)
    X, y = encoded_sequences[:,:2], encoded_sequences[:,2]
    print(X.shape, y.shape)
    X = np.array(X)
    y = to_categorical(y, num_classes=len(vocabulary))

    experiment = WordLevelLanguageModelExperiment()
    model = experiment.get_model(max_document_length=MAX_DOCUMENT_LENGTH, vocabulary_size=len(vocabulary))
    model.compile(optimizer=Adam(), loss="categorical_crossentropy",
                  metrics=["accuracy"])
    plot_model(model, show_shapes=True, to_file=model_name + '.png')

    callbacks = [
        GradientDebugger(),
        TensorBoard(log_dir='/tmp/char_level_language_model'),

    ]
    model.fit(x=X, y=y, epochs=800, verbose=1, callbacks=callbacks, shuffle=False)
    predicted_text = predict_text(model, transformer=transformer, max_sequence_length=MAX_DOCUMENT_LENGTH, num_tokens=4,
                                  seed_text=["and", "jill"])
    predicted_text = predict_text(model, transformer=transformer, max_sequence_length=MAX_DOCUMENT_LENGTH, num_tokens=4,
                                  seed_text=["jack", "and"])
    print(predicted_text)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--training_data_file_path', required=True)
    parser.add_argument('--model_name', required=True)
    parser.add_argument('--model_file_path', required=True)

    args = parser.parse_args()
    main(args.training_data_file_path, args.model_name, args.model_file_path)
