import logging
import joblib
import numpy as np
import os
from uuid import uuid4

from argparse import ArgumentParser
from keras.models import load_model

from ml.visualizations.sentiment_embedding.transformer import HierarchicalTripletTransformer, TweetTokenizer, Vocabulary
from ml.visualizations.sentiment_embedding.sentiment import AttentionWithContext
from ml.visualizations.sentiment_embedding.provider import TripletProvider
from ml.visualizations.sentiment_embedding.train import load_samples, get_max_length, train_val_split, TripletBatchGenerator

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


def load_keras_model(model_file):
    return load_model(model_file,  custom_objects={'AttentionWithContext': AttentionWithContext})


def main(model_file, out_tsv_file, out_labels_file, data_file_path, vocab_file_path):
    model = load_keras_model(model_file)

    uid = uuid4().hex
    os.makedirs(uid)

    samples = load_samples(data_file_path)
    train_samples, val_samples = train_val_split(samples)
    val_provider = TripletProvider(val_samples, shuffle=True)

    tokenizer = TweetTokenizer()
    tokenized_samples = [tokenizer.tokenize(sample.text) for sample in train_samples]

    vocabulary = joblib.load(vocab_file_path)
    vocabulary.fit((c for tokens in tokenized_samples for token in tokens for c in token))

    transformer = HierarchicalTripletTransformer(vocabulary)

    max_document_length, max_token_length = get_max_length(tokenized_samples)
    val_generator = TripletBatchGenerator(val_provider, transformer, max_document_length, max_token_length,
                                          len(vocabulary), 1)

    vectors = []
    labels = []
    for sample in val_generator:
        X, y, triplet = sample
        for xi in X:
            prediction = model.predict(xi)
            vectors.append(prediction)
            labels.append(sample.text)

    model.predict()
    np.savetxt('vectors_out.tsv', vectors, delimiter='\t')


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--model_file', required=True)
    parser.add_argument('--out_tsv_file', required=True)
    parser.add_argument('--out_labels_file', required=True)
    parser.add_argument('--data_file_path', required=True)
    parser.add_argument('--vocab_file_path', required=True)

    args = parser.parse_args()
    main(args.model_file, args.out_tsv_file, args.out_labels_file, args.data_file_path, args.vocab_file_path)
