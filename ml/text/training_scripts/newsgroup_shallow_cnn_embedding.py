from argparse import ArgumentParser

from collections import namedtuple

import os
from uuid import uuid4

import joblib
import numpy as np
from keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from keras.optimizers import Adam
from keras.utils import plot_model, to_categorical

from ml.common.callback.keras import GradientDebugger
from ml.embedding.util import get_embedding_index, get_embedding_matrix
from ml.text.transformer.vocabulary import Vocabulary, WordTokenizer
from ml.text.experiment.shallow_cnn_text_embedding import get_model
from ml.text.transformer.text import TextTransformer

Sample = namedtuple("sample", ("text", "label"))


def get_training_data(text_data_dir):
    samples = []  # list of text samples
    target_name_ix_map = {}  # dictionary mapping label name to numeric id

    for name in sorted(os.listdir(text_data_dir)):
        path = os.path.join(text_data_dir, name)
        if os.path.isdir(path):
            label_id = len(target_name_ix_map)
            target_name_ix_map[name] = label_id
            for fname in sorted(os.listdir(path)):
                if fname.isdigit():
                    fpath = os.path.join(path, fname)
                    f = open(fpath, encoding="latin-1")
                    text = f.read()
                    i = text.find('\n\n')  # skip header
                    if 0 < i:
                        text = text[i:]
                        samples.append(Sample(text, label_id))
                    f.close()

    print('Found %s texts.' % len(samples))
    return samples, target_name_ix_map


class SampleBatchProvider:
    def __init__(self, batch_size, num_labels, max_document_length, max_token_length):
        self._batch_size = batch_size
        self._num_labels = num_labels
        self._max_document_length = max_document_length

    def get_batch(self, tokenized_samples, labels):
        targets = np.zeros(shape=(self._batch_size, self._num_labels))
        samples = np.zeros(shape=(self._batch_size, self._max_document_length))

        for sample_ix, sample in enumerate(tokenized_samples):
            label = labels[sample_ix]

            targets[sample_ix, label] = 1
            samples[sample_ix, :sample.shape[0]] = \
                sample[:self._max_document_length]
        return samples, targets


def main(model_name, model_file_path, embedding_path, training_data_dir, batch_size=32, epochs=20):
    uid = str(uuid4())
    samples, labels_name_ix_map = get_training_data(training_data_dir)
    labels = [sample.label for sample in samples]
    num_targets = len(labels_name_ix_map)
    texts = [sample.text for sample in samples]
    vocabulary = Vocabulary()
    tokenizer = WordTokenizer(texts=texts)
    tokenized_samples = [tokenizer.tokenize(sample.text) for sample in samples]
    vocabulary.fit((c for tokens in tokenized_samples for token in tokens for c in token))
    print(tokenized_samples[0:1])

    token_index_map = vocabulary.dictionary

    vocab_path = os.path.join(uid, 'vocab_{}.pkl'.format(uid))
    num_tokens = len(vocabulary)
    max_document_length = tokenizer.get_max_length(tokenized_samples)
    transformer = TextTransformer(class_map=labels_name_ix_map, max_sequence_length=max_document_length,
                                  vocabulary=vocabulary, tokenizer=tokenizer)

    sample_batch_provider = SampleBatchProvider(batch_size=len(samples), num_labels=num_targets,
                                                max_document_length=max_document_length, max_token_length=num_tokens)

    X = [transformer.transform(sample.text) for sample in samples]
    X, y = sample_batch_provider.get_batch(X, labels)

    embedding_index = get_embedding_index(embedding_path)
    embedding_dimensions = len(list(embedding_index.values())[0])

    embedding_matrix = get_embedding_matrix(
        embedding_dimensions=embedding_dimensions,
        embedding_index=embedding_index,
        token_index_mapping=token_index_map
    )

    model = get_model(max_document_length=max_document_length, max_num_tokens=num_tokens,
                      embedding_weights=embedding_matrix, embedding_dims=embedding_dimensions, num_targets=num_targets)
    model.compile(optimizer=Adam(), loss="binary_crossentropy", metrics=["accuracy"])
    plot_model(model, to_file=model_name + '.png')

    callbacks = [
        GradientDebugger(),
        TensorBoard(log_dir='/tmp/reuters_embedding'),
        ModelCheckpoint(model_file_path, monitor='val_acc', verbose=1, save_best_only=True, mode='max'),
        EarlyStopping(monitor='val_acc', patience=5, mode='max'),
        ReduceLROnPlateau(factor=0.1, verbose=1),

    ]
    model.fit(x=X, y=y, batch_size=batch_size, epochs=epochs, verbose=1, callbacks=callbacks, validation_split=0.2,
              shuffle=True)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--model_name')
    parser.add_argument('--model_file_path')
    parser.add_argument('--embedding_path')
    parser.add_argument('--training_data_dir')

    args = parser.parse_args()
    main(args.model_name, args.model_file_path, args.embedding_path, args.training_data_dir)
