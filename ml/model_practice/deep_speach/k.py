import json
import numpy as np
import os

from argparse import ArgumentParser
from keras import backend as K
from keras.layers import Input, Conv1D, Bidirectional, GRU, Dense, Lambda, BatchNormalization, TimeDistributed
from keras.models import Model
from keras.optimizers import sgd
from scipy import signal
from scipy.io import wavfile

from ml.text.transformer.vocabulary import Vocabulary, CharacterTokenizer


def load_json_lines(file_path):
    with open(file_path) as f:
        data = map(json.loads, f)
        for datum in data:
            yield datum


class Transformer:
    def __init__(self, max_frequency, max_time, output_vocabulary):
        self.max_frequency = max_frequency
        self.max_time = max_time
        self.output_vocabulary = output_vocabulary
        self.output_vocab_size = len(output_vocabulary)

    def transform_x(self, audio_file_path):
        x = np.zeros(shape=(self.max_time, self.max_frequency))
        sample_rate, samples = wavfile.read(audio_file_path)
        frequencies, times, spectogram = signal.spectrogram(samples, sample_rate)  # spectogram is freqs * timestamps
        spectogram = np.swapaxes(spectogram, 0, 1)  # timestamps * freq
        x[:spectogram.shape[0], :spectogram.shape[1]] = spectogram[:self.max_time, :self.max_frequency]
        return x

    def transform_y(self, text):
        """
        y = np.zeros(shape=(self.max_time, self.output_vocab_size))
        for i, encoded_token_ix in enumerate(encoded_tokens):
            y[i][encoded_token_ix] = 1
        """
        y = np.zeros(shape=(len(text)))
        encoded_tokens = self.output_vocabulary.encode(self.tokenize(text))
        for i, encoded_token in enumerate(encoded_tokens):
            y[i] = encoded_token

        return y

    def tokenize(self, text):
        return [character for character in text]


class DataGenerator:
    def __init__(self, audio_dir, labels_file):
        self.audio_dir = audio_dir
        self.labels_file = labels_file
        self.joined_data = self.join_data()

    def join_data(self):
        joined_data = []

        labels = load_json_lines(self.labels_file)
        for label in labels:
            audio_file_path = self._get_audio_file_path(label['uuid'])
            joined_data.append(self._make_datum(audio_file_path, label))

        return joined_data

    def _get_audio_file_path(self, audio_id):
        return os.path.join(self.audio_dir, audio_id + '.wav')

    def _make_datum(self, audio_file_path, label):
        return {'x': audio_file_path, 'y': label['text'], 'uuid': label['uuid'], 'payload': label}

    def generate(self):
        for datum in self.joined_data:
            yield datum


def ctc_lambda_func(args):
    y_pred, labels, input_length, label_length = args
    # the 2 is critical here since the first couple outputs of the RNN
    # tend to be garbage:
    #y_pred = y_pred[:, 2:, :]
    return K.ctc_batch_cost(labels, y_pred, input_length, label_length)


def get_model(max_audio_length, max_frequency, output_vocab_size, num_channels=1):
    input_audio = Input(shape=(max_audio_length, max_frequency), name='input')  # Spectogram (like W * H * Channels)
    conv1 = Conv1D(filters=16, kernel_size=(3,), strides=(1,), activation='relu', padding='same')(
        input_audio)  # ablation 1-->3
    conv1 = BatchNormalization()(conv1)
    gru1 = Bidirectional(GRU(64, activation='relu', return_sequences=True))(conv1)  # ablation 1-->7
    gru1 = BatchNormalization()(gru1)
    y_pred = TimeDistributed(Dense(output_vocab_size, activation='softmax'))(
        gru1)  # softmax computing a probability distribution over characters

    # CTC Loss
    labels = Input(name='the_labels', shape=[1], dtype='float32')
    input_length = Input(name='input_length', shape=[1], dtype='int64')
    label_length = Input(name='label_length', shape=[1], dtype='int64')

    ctc_loss = Lambda(ctc_lambda_func, output_shape=(1,), name='ctc')([y_pred, labels, input_length, label_length])
    model = Model(inputs=[input_audio, labels, input_length, label_length], outputs=ctc_loss)
    model.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer='sgd')
    return model


def main(audio_dir, labels_file_path):
    data_generator = DataGenerator(audio_dir, labels_file_path)
    texts = list([datum['y'] for datum in data_generator.generate()])
    token_generator = CharacterTokenizer(texts)
    tokens = token_generator.get_tokens()
    vocabulary = Vocabulary()
    vocabulary.fit(tokens)
    transformer = Transformer(max_frequency=129, max_time=400, output_vocabulary=vocabulary)
    num_training_datums = 10

    X = np.zeros(shape=(num_training_datums, transformer.max_time, transformer.max_frequency))
    y = np.zeros(shape=(num_training_datums, transformer.max_time))
    labels = []
    input_lengths = []
    label_lengths = []

    for i, datum in enumerate(list(data_generator.generate())[0:num_training_datums]):
        xi = transformer.transform_x(datum['x'])
        yi = transformer.transform_y(datum['y'])
        input_lengths.append(xi.shape[0])
        label_lengths.append(yi.shape[0])
        labels.append(yi)
        X[i, :transformer.max_time, :transformer.max_frequency] = xi[:transformer.max_time, :transformer.max_frequency]
        #y[i, :transformer.max_time, :len(vocabulary)] = yi[:transformer.max_time, :len(vocabulary)]
        #y[i, :transformer.max_time] = yi[:transformer.max_time]

    model = get_model(max_audio_length=transformer.max_time, max_frequency=transformer.max_frequency,
                      output_vocab_size=len(vocabulary))
    print(model.summary())
    np.array(labels)
    inputs = {'input':        X,
              'the_labels':   np.array(labels),
              'input_length': np.array(input_lengths),
              'label_length': np.array(label_lengths),
              }

    outputs = {'ctc': np.zeros(shape=[num_training_datums])}

    model.fit(x=inputs, y=outputs)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--audio_dir', required=True)
    parser.add_argument('--labels_file_path', required=True)

    args = parser.parse_args()
    main(args.audio_dir, args.labels_file_path)
