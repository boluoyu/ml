import json
import keras_metrics
import numpy as np

from argparse import ArgumentParser
from collections import namedtuple
from nltk.tokenize.casual import TweetTokenizer
from keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from keras.optimizers import Adam
from keras.utils import plot_model
from imblearn.ensemble import EasyEnsemble
from random import shuffle
from tqdm import tqdm

from ml.common.callback.keras import GradientDebugger
from ml.text.transformer.vocabulary import Vocabulary, WordTokenizer
from ml.text.experiment.ngram_cnn import NgramCNNModel
from ml.text.transformer.text import TextTransformer

Sample = namedtuple("sample", ("text", "label"))
INDEX_NAME_CLASS_MAP = {0: "negative", 1: "positive"}
NAME_INDEX_CLASS_MAP = {"0": 0, "positive": 1}


def get_training_data(text_data_dir):
    with open(text_data_dir) as f:
        return [Sample(sample['text'], sample["curation_type_enum"])
                for sample in map(json.loads, tqdm(f))]


class SampleBatchProvider:
    def __init__(self, X, y, num_labels, max_document_length, max_token_length):
        self._num_labels = num_labels
        self._max_document_length = max_document_length

    def get_batch(self, tokenized_samples, labels):
        e = EasyEnsemble(random_state=0, n_subsets=1)
        e.fit(tokenized_samples, labels)
        X_resampled, y_resampled = e.sample(tokenized_samples, labels)

        X = X_resampled[0]
        y = y_resampled[0]

        targets = np.zeros(shape=(len(X), self._num_labels))
        samples = np.zeros(shape=(len(X), self._max_document_length))

        for sample_ix, sample in enumerate(X):
            label = y[sample_ix]

            targets[sample_ix, label] = 1
            samples[sample_ix, :sample.shape[0]] = \
                sample[:self._max_document_length]
        return samples, targets


def get_samples_labels(training_data_path):
    samples = get_training_data(training_data_path)
    shuffle(samples)
    labels = [sample.label for sample in samples]
    return samples, labels


def get_vocabulary_tokenizer(samples):
    texts = [sample.text for sample in samples]
    vocabulary = Vocabulary()
    tokenizer = WordTokenizer(texts=texts, tokenizer=TweetTokenizer())
    tokenized_samples = [tokenizer.tokenize(sample.text) for sample in samples]
    vocabulary.fit((token for tokens in tokenized_samples for token in tokens))
    print(tokenized_samples[0:1])
    return vocabulary, tokenizer


def main(model_name, model_file_path, training_data_path, epochs=20):
    samples, labels = get_samples_labels(training_data_path)
    vocabulary, tokenizer = get_vocabulary_tokenizer(samples)
    num_classes = len(INDEX_NAME_CLASS_MAP)

    num_tokens = len(vocabulary)
    tokenized_samples = [tokenizer.tokenize(sample.text) for sample in samples]
    max_document_length = tokenizer.get_max_length(tokenized_samples)
    transformer = TextTransformer(class_map=NAME_INDEX_CLASS_MAP, max_sequence_length=max_document_length,
                                  vocabulary=vocabulary, tokenizer=tokenizer)

    X = [transformer.transform(sample.text) for sample in samples]
    sample_batch_provider = SampleBatchProvider(X=X, y=labels, num_labels=num_classes,
                                                max_document_length=max_document_length, max_token_length=num_tokens)
    X, y = sample_batch_provider.get_batch(X, labels)

    experiment = NgramCNNModel()
    model = experiment.get_model(max_document_length=max_document_length, num_classes=num_classes,
                                 vocabulary_size=len(vocabulary))
    model.compile(optimizer=Adam(), loss="binary_crossentropy",
                  metrics=["accuracy", keras_metrics.precision(), keras_metrics.recall()])
    plot_model(model, show_shapes=True, to_file=model_name + '.png')

    callbacks = [
        GradientDebugger(),
        TensorBoard(log_dir='/tmp/shooting_ngram_cnn'),
        ModelCheckpoint(model_file_path, monitor='val_acc', verbose=1, save_best_only=True, mode='max'),
        EarlyStopping(monitor='val_acc', patience=5, mode='max'),
        ReduceLROnPlateau(factor=0.1, verbose=1),

    ]
    model.fit(x=[X, X, X], y=y, batch_size=None, epochs=epochs, verbose=1, callbacks=callbacks, validation_split=0.2,
              shuffle=True, steps_per_epoch=20, validation_steps=100)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--model_name', required=True)
    parser.add_argument('--model_file_path', required=True)
    parser.add_argument('--training_data_path', required=True)

    args = parser.parse_args()
    main(args.model_name, args.model_file_path, args.training_data_path)
