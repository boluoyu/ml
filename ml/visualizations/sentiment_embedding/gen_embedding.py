import logging
import joblib
import numpy as np
from uuid import uuid4

from argparse import ArgumentParser
from keras.models import load_model
from sklearn.cluster import DBSCAN
from sklearn.metrics.pairwise import pairwise_distances

from ml.visualizations.sentiment_embedding.transformer import HierarchicalTripletTransformer
from ml.visualizations.sentiment_embedding.sentiment import AttentionWithContext
from ml.visualizations.sentiment_embedding.train import load_samples

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


def load_keras_model(model_file):
    return load_model(model_file, custom_objects={'AttentionWithContext': AttentionWithContext})


def get_vector(encoder, transformer, text):
    max_doc_length = encoder.input.shape[1].value
    max_token_length = encoder.input.shape[2].value
    vector = np.zeros((1, max_doc_length, max_token_length))
    codes = transformer._transform(text.lower())
    vector[0, :codes.shape[0], :codes.shape[1]] = codes[:max_doc_length, :max_token_length]
    return encoder.predict(vector)[0]


def export_vectors(prefix, data_file_path, encoder, transformer):
    samples = load_samples(data_file_path)[0:500]
    update_vectors = np.array([get_vector(encoder, transformer, sample.text) for sample in samples])
    np.savetxt(prefix + '_vectors.tsv', update_vectors, delimiter='\t')
    dbscan = DBSCAN(metric='precomputed', min_samples=2, eps=0.25)

    with open(prefix + '_raw_text.tsv', 'w') as f:
        f.write('\t'.join(['text', 'sentiment']) + '\n')
        for sample in samples:
            sentiment = sample.sentiment
            text = sample.text.replace('\n', ' ')
            f.write('\t'.join([text, sentiment]) + '\n')

    print('Computing distances')
    distances = pairwise_distances(update_vectors, n_jobs=6)
    print('Computing dbscan')
    clusters = dbscan.fit_predict(distances)
    with open(prefix + '_text.tsv', 'w') as f:
        f.write('\t'.join(['text', 'sentiment', 'cluster']) + '\n')
        for sample, cluster in zip(samples, clusters):
            sentiment = sample.sentiment
            text = sample.text.replace('\n', ' ')
            cluster = str(cluster)
            f.write('\t'.join([text, sentiment, cluster]) + '\n')


def main(prefix, model_file, data_file_path, vocab_file_path):
    model = load_keras_model(model_file)
    vocabulary = joblib.load(vocab_file_path)
    transformer = HierarchicalTripletTransformer(vocabulary)
    export_vectors(prefix, data_file_path, model, transformer)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--prefix', required=False, default=uuid4().hex)
    parser.add_argument('--model_file', required=True)
    parser.add_argument('--data_file_path', required=True)
    parser.add_argument('--vocab_file_path', required=True)

    args = parser.parse_args()
    main(args.prefix, args.model_file, args.data_file_path, args.vocab_file_path)
