from argparse import ArgumentParser

import numpy as np
from keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from keras.optimizers import Adam
from keras.utils import to_categorical, plot_model
from keras_preprocessing.sequence import pad_sequences

from ml.common.callback.keras import GradientDebugger
from ml.text.experiment.char_level_language_model import CharLevelLanguageModelExperiment

from ml.text.transformer.vocabulary import Vocabulary


def get_training_data(training_data_file_path):
    f = open(training_data_file_path, 'r')
    data = f.read()
    f.close()
    return data


def preprocess_text(text):
    tokens = text.split(' ')
    return ' '.join(tokens)


def get_sequences(text, seq_length=10):
    sequences = []

    for i in range(seq_length, len(text)):
        sequence = text[i - seq_length:i + 1]
        sequences.append(sequence)

    return sequences


class CharTextTransformer:
    def __init__(self, vocabulary):
        self.vocabulary = vocabulary

    def transform(self, text):
        tokens = [char for char in text]
        encoded = self.vocabulary.encode(tokens)
        return encoded


def predict_text(model, seed_text, transformer, max_sequence_length, num_chars):
    in_text = seed_text
    vocabulary = transformer.vocabulary

    for _ in range(num_chars):
        encoded = transformer.transform(in_text)
        encoded = pad_sequences([encoded], maxlen=max_sequence_length, truncating='pre')
        encoded = to_categorical(encoded, num_classes=len(vocabulary))
        y_pred = model.predict(encoded, verbose=1)
        y_pred = np.argmax(y_pred, axis=1)[0]
        predicted_char = vocabulary.decode([y_pred])[0]
        in_text += predicted_char

    return in_text



def main(training_data_file_path, model_name, model_file_path):
    MAX_DOCUMENT_LENGTH = 10
    training_data = get_training_data(training_data_file_path)
    training_data = preprocess_text(training_data)
    sequences = get_sequences(training_data)
    print(len(sequences), sequences[0])

    tokens = sorted(list(set(training_data)))
    vocabulary = Vocabulary()
    vocabulary.fit(tokens)

    transformer = CharTextTransformer(vocabulary)
    encoded_sequences = list(map(transformer.transform, sequences))
    encoded_sequences = np.array(encoded_sequences)
    X, y = encoded_sequences[:, :-1], encoded_sequences[:, -1:]
    print(X.shape, y.shape)
    sequences = [to_categorical(x, num_classes=len(vocabulary)) for x in X]
    X = np.array(sequences)
    y = to_categorical(y, num_classes=len(vocabulary))

    experiment = CharLevelLanguageModelExperiment()
    model = experiment.get_model(max_document_length=MAX_DOCUMENT_LENGTH, vocabulary_size=len(vocabulary))
    model.compile(optimizer=Adam(), loss="categorical_crossentropy",
                  metrics=["accuracy"])
    plot_model(model, show_shapes=True, to_file=model_name + '.png')

    callbacks = [
        GradientDebugger(),
        TensorBoard(log_dir='/tmp/char_level_language_model'),
        ReduceLROnPlateau(factor=0.1, verbose=1),

    ]
    model.fit(x=X, y=y, epochs=100, verbose=1, callbacks=callbacks, shuffle=True)
    predicted_text = predict_text(model, transformer=transformer, max_sequence_length=10, num_chars=20, seed_text="sing a so")
    print(predicted_text)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--training_data_file_path', required=True)
    parser.add_argument('--model_name', required=True)
    parser.add_argument('--model_file_path', required=True)

    args = parser.parse_args()
    main(args.training_data_file_path, args.model_name, args.model_file_path)
