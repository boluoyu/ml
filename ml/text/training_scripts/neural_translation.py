import json
import random
from argparse import ArgumentParser
from collections import namedtuple

import numpy as np
from keras.callbacks import TensorBoard, ReduceLROnPlateau, ModelCheckpoint
from keras.utils import to_categorical, plot_model
from keras_preprocessing.sequence import pad_sequences

import joblib
from ml.common.callback.keras import GradientDebugger
from ml.text.transformer.vocabulary import Vocabulary
from ml.text.experiment.neural_translation import NeuralTranslationExperiment


def load_jsonl(file_path):
    with open(file_path) as f:
        data = map(json.loads, f)
        for datum in data:
            yield datum


def load_vocabulary(file_path):
    return joblib.load(file_path)


class WordTextTransformer:
    def __init__(self, source_vocabulary, target_vocabulary, max_source_length, max_target_length):
        self.source_vocabulary = source_vocabulary
        self.target_vocabulary = target_vocabulary
        self.max_source_length = max_source_length
        self.max_target_length = max_target_length

    def tokenize(self, text):
        return text.split()

    def transform_xi(self, text):
        xi = self.tokenize(text)
        xi = self.source_vocabulary.encode(xi)
        xi = pad_sequences([xi], maxlen=self.max_source_length, padding="post")[0]
        return np.array(xi)

    def transform_yi(self, text):
        yi = self.tokenize(text)
        yi = self.target_vocabulary.encode(yi)
        yi = pad_sequences([yi], maxlen=self.max_target_length, padding="post")
        yi = to_categorical(yi, num_classes=len(self.target_vocabulary))
        return np.array(yi)


Sample = namedtuple("Sample", ("source", "target"))


class SampleProvider:
    def __init__(self, samples, shuffle=False):
        self._samples = samples
        self._shuffle = shuffle

    def __iter__(self):
        if self._shuffle:
            random.shuffle(self._samples)

        while True:
            sample = random.choice(self._samples)
            yield Sample(sample["source"], sample["target"])


class BatchGenerator:
    def __init__(self, provider, max_source_length, max_target_length, target_vocabulary_size, transformer, batch_size):
        self.provider = provider
        self.transformer = transformer
        self.target_vocabulary_size = target_vocabulary_size
        self.batch_size = batch_size
        self.max_source_length = max_source_length
        self.max_target_length = max_target_length

    def __iter__(self):
        while True:
            y = np.zeros(shape=(self.batch_size, self.max_target_length, self.target_vocabulary_size))
            X = np.zeros(shape=(self.batch_size, self.max_source_length))

            for i, sample in enumerate(self.provider):
                idx = i % self.batch_size

                xi = self.transformer.transform_xi(sample.source)
                yi = self.transformer.transform_yi(sample.target)

                X[idx, :self.max_source_length] = xi[:self.max_target_length]
                y[idx, :self.max_target_length, :self.target_vocabulary_size] = yi[:self.max_target_length,
                                                                                :self.target_vocabulary_size]

                if idx == self.batch_size - 1:
                    yield X, y


def main(model_name, model_file_path, training_data_path, validation_data_path, source_vocabulary_path,
         target_vocabulary_path, max_source_length, max_target_length, batch_size=64):
    training_samples = list(load_jsonl(training_data_path))
    validation_samples = list(load_jsonl(validation_data_path))
    source_vocabulary = load_vocabulary(source_vocabulary_path)
    target_vocabulary = load_vocabulary(target_vocabulary_path)

    transformer = WordTextTransformer(source_vocabulary=source_vocabulary, max_source_length=max_source_length,
                                      max_target_length=max_target_length, target_vocabulary=target_vocabulary)

    training_provider = SampleProvider(samples=training_samples)
    validation_provider = SampleProvider(samples=validation_samples)
    training_batch_generator = BatchGenerator(batch_size=batch_size, provider=training_provider,
                                              transformer=transformer, max_source_length=max_source_length, max_target_length=max_target_length, target_vocabulary_size=len(target_vocabulary))
    validation_batch_generator = BatchGenerator(batch_size=batch_size, provider=validation_provider,
                                                transformer=transformer, max_source_length=max_source_length, max_target_length=max_target_length, target_vocabulary_size=len(target_vocabulary))

    experiment = NeuralTranslationExperiment()
    model = experiment.get_model(target_vocab_size=len(target_vocabulary),
                                 source_vocab_size=len(source_vocabulary),
                                 source_timesteps=max_source_length,
                                 target_timesteps=max_target_length)

    plot_model(model, to_file='translator.png', show_shapes=True)

    callbacks = [
        GradientDebugger(),
        TensorBoard(log_dir='/tmp/translator'),
        ReduceLROnPlateau(factor=0.01, verbose=1),
        ModelCheckpoint(model_name + '.h5', monitor='val_loss', verbose=1, save_best_only=True, mode='min')
    ]

    model.fit_generator(
        generator=iter(training_batch_generator),
        epochs=30,
        steps_per_epoch=500,
        validation_steps=50,
        validation_data=iter(validation_batch_generator),
        callbacks=callbacks,
        verbose=1)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--model_name', required=True)
    parser.add_argument('--model_file_path', required=True)
    parser.add_argument('--training_data_file_path', required=True)
    parser.add_argument('--validation_data_file_path', required=True)
    parser.add_argument('--source_vocabulary_path', required=True)
    parser.add_argument('--target_vocabulary_path', required=True)
    parser.add_argument('--max_source_length', required=True, type=int)
    parser.add_argument('--max_target_length', required=True, type=int)

    args = parser.parse_args()
    main(
        model_name=args.model_name,
        model_file_path=args.model_file_path,
        training_data_path=args.training_data_file_path,
        validation_data_path=args.validation_data_file_path,
        source_vocabulary_path=args.source_vocabulary_path,
        target_vocabulary_path=args.target_vocabulary_path,
        max_source_length=args.max_source_length,
        max_target_length=args.max_target_length
    )
