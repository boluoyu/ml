import os
import random
from argparse import ArgumentParser
from functools import partial
from uuid import uuid4

import joblib
import keras.backend as K
import numpy as np
from keras.callbacks import LambdaCallback, Callback, ReduceLROnPlateau, TensorBoard
from keras.optimizers import Adam
from nltk import TweetTokenizer

from ml.visualizations.sentiment_embedding.provider import load_samples, TripletProvider
from ml.visualizations.sentiment_embedding.sentiment import triplet_model, triplet_loss, triplet_accuracy, get_model
from ml.visualizations.sentiment_embedding.transformer import HierarchicalTripletTransformer, Vocabulary

MODEL_SAVER_FORMAT = '{model_name}.{epoch:02d}-{val_loss:.6f}.hdf5'


class TripletBatchGenerator:
    def __init__(self, provider, transformer, max_document_length, max_token_length, vocab_size, batch_size):
        self._provider = provider
        self._transformer = transformer
        self._batch_size = batch_size
        self._max_document_length = max_document_length
        self._max_token_length = max_token_length
        self._vocab_size = vocab_size

    def __iter__(self):
        while True:
            targets = np.ones(shape=(self._batch_size, 2))
            anchor_batch = self._init_batch()
            related_batch = self._init_batch()
            unrelated_batch = self._init_batch()
            for i, triplet in enumerate(self._provider):
                idx = i % self._batch_size
                print('triplet', triplet)
                anchor, related, unrelated = self._transformer.transform(triplet)

                anchor_batch[idx, :anchor.shape[0], :anchor.shape[1]] = \
                    anchor[:self._max_document_length, :self._max_token_length]

                related_batch[idx, :related.shape[0], :related.shape[1]] = \
                    related[:self._max_document_length, :self._max_token_length]

                unrelated_batch[idx, :unrelated.shape[0], :unrelated.shape[1]] = \
                    unrelated[:self._max_document_length, :self._max_token_length]
                if idx == self._batch_size - 1:
                    yield [anchor_batch, related_batch, unrelated_batch], targets
                    anchor_batch = self._init_batch()
                    related_batch = self._init_batch()
                    unrelated_batch = self._init_batch()

    def _init_batch(self):
        return np.zeros((self._batch_size, self._max_document_length, self._max_token_length))




class GradientDebugger(Callback):
    def on_batch_end(self, batch, logs=None):
        self._log_weights(self.model)

    def _log_weights(self, model):
        for layer in model.layers:
            if hasattr(layer, 'layers'):
                self._log_weights(layer)
            else:
                for weight in layer.weights:
                    print(layer.name, weight.name, np.mean(K.eval(weight)))


def train_val_split(samples, split=0.2):
    r = random.Random(1)
    r.shuffle(samples)
    split_num = int(len(samples) * split)
    print(split_num)
    train_samples = samples[:split_num]
    val_samples = samples[split_num:]
    print(len(train_samples), len(val_samples))
    return train_samples, val_samples


def get_max_length(tokenized_samples):
    document_lengths = sorted([len(tokens) for tokens in tokenized_samples])
    token_lengths = sorted([len(token) for tokens in tokenized_samples for token in tokens])
    max_document_length = np.percentile(document_lengths, 95)
    max_token_length = np.percentile(token_lengths, 95)
    return int(max_document_length), int(max_token_length)


def save_models(model_dict, artifact_format, epoch, logs=None):
    logs = logs or {}
    for model_name, model in model_dict.items():
        path = artifact_format.format(epoch=epoch + 1, model_name=model_name, **logs)
        model.save(path, overwrite=True)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--samples_path')
    parser.add_argument('--batch-size', type=int, default=32)
    args = parser.parse_args()

    uid = uuid4().hex
    os.makedirs(uid)
    print('training run: {}'.format(uid))

    samples = load_samples(args.samples_path)
    train_samples, val_samples = train_val_split(samples)
    train_provider = TripletProvider(train_samples, shuffle=True)
    val_provider = TripletProvider(val_samples, shuffle=True)

    tokenizer = TweetTokenizer()
    tokenized_samples = [tokenizer.tokenize(sample.text) for sample in train_samples]

    vocabulary = Vocabulary()
    vocabulary.fit((c for tokens in tokenized_samples for token in tokens for c in token))
    vocab_path = os.path.join(uid, 'vocab_{}.pkl'.format(uid))
    joblib.dump(vocabulary, vocab_path)

    transformer = HierarchicalTripletTransformer(vocabulary)

    max_document_length, max_token_length = get_max_length(tokenized_samples)

    train_generator = TripletBatchGenerator(train_provider, transformer, max_document_length, max_token_length,
                                            len(vocabulary), args.batch_size)

    val_generator = TripletBatchGenerator(val_provider, transformer, max_document_length, max_token_length,
                                          len(vocabulary), args.batch_size)

    encoder = get_model(
        max_document_length=max_document_length,
        max_token_length=max_token_length,
        vocab_size=len(vocabulary)
    )
    encoder.summary()

    triplet = triplet_model(encoder, (max_document_length, max_token_length))
    triplet.compile(optimizer=Adam(), loss=triplet_loss, metrics=[triplet_accuracy])
    triplet.summary()

    models_dict = {'encoder_' + uid: encoder}
    model_path = os.path.join(uid, MODEL_SAVER_FORMAT)
    saver = partial(save_models, models_dict, model_path)

    print('Starting training')

    triplet.fit_generator(
        generator=iter(train_generator),
        steps_per_epoch=10000,
        validation_data=iter(val_generator),
        validation_steps=500,
        verbose=1,
        epochs=20,
        callbacks=[
            LambdaCallback(on_epoch_end=saver),
            ReduceLROnPlateau(factor=0.1, verbose=1),
            TensorBoard(log_dir='/tmp/triplet_v2')

        ]
    )