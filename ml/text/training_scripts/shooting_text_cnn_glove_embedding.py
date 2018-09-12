import json
from argparse import ArgumentParser

from collections import namedtuple
from random import shuffle

from nltk.tokenize.casual import TweetTokenizer
import numpy as np
from keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from keras.optimizers import Adam
from keras.utils import plot_model
from tqdm import tqdm

from ml.common.callback.keras import GradientDebugger
from ml.embedding.util import get_embedding_index, get_embedding_matrix
from ml.text.transformer.vocabulary import Vocabulary, WordTokenizer
from ml.text.experiment.shallow_cnn_text_embedding import get_model
from ml.text.transformer.text import TextTransformer

Sample = namedtuple("sample", ("text", "label"))
INDEX_NAME_CLASS_MAP = {0: "negative", 1: "positive"}
NAME_INDEX_CLAS_MAP = {"0": 0, "positive": 1}


def get_training_data(text_data_dir):
    with open(text_data_dir) as f:
        return [Sample(sample['text'], sample["curation_type_enum"])
                for sample in map(json.loads, tqdm(f))]


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


def main(model_name, model_file_path, embedding_path, training_data_path, batch_size=32, epochs=20):
    samples = get_training_data(training_data_path)
    shuffle(samples)
    labels = [sample.label for sample in samples]
    num_targets = len(NAME_INDEX_CLAS_MAP)

    texts = [sample.text for sample in samples]
    vocabulary = Vocabulary()
    tokenizer = WordTokenizer(texts=texts, tokenizer=TweetTokenizer())

    tokenized_samples = [tokenizer.tokenize(sample.text) for sample in samples]
    vocabulary.fit((token for tokens in tokenized_samples for token in tokens))
    print(tokenized_samples[0:1])

    token_index_map = vocabulary.dictionary

    num_tokens = len(vocabulary)
    max_document_length = tokenizer.get_max_length(tokenized_samples)
    transformer = TextTransformer(class_map=NAME_INDEX_CLAS_MAP, max_sequence_length=max_document_length,
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
        TensorBoard(log_dir='/tmp/shooting_cnn'),
        ModelCheckpoint(model_file_path, monitor='val_acc', verbose=1, save_best_only=True, mode='max'),
        EarlyStopping(monitor='val_acc', patience=5, mode='max'),
        ReduceLROnPlateau(factor=0.1, verbose=1),

    ]
    model.fit(x=X, y=y, batch_size=None, epochs=epochs, verbose=1, callbacks=callbacks, validation_split=0.2,
              shuffle=True, steps_per_epoch=20, validation_steps=100)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--model_name', required=True)
    parser.add_argument('--model_file_path', required=True)
    parser.add_argument('--embedding_path', required=True)
    parser.add_argument('--training_data_path', required=True)

    args = parser.parse_args()
    main(args.model_name, args.model_file_path, args.embedding_path, args.training_data_path)
