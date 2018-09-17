import re
import string
from argparse import ArgumentParser

import numpy as np
import os

from keras.engine.saving import load_model
from keras.utils import to_categorical
from keras_preprocessing.sequence import pad_sequences
from nltk.tokenize.casual import TweetTokenizer

import joblib

from ml.text.transformer.vocabulary import Vocabulary

LOCALE_SPANISH = "es"

def get_ingestion_source():
    from ds_common.source.kafka import KafkaSource
    from ds_common.discovery.kafka import get_kafka_hosts, get_zookeeper_client
    zookeeper_hosts = os.environ["ZOOKEEPER_HOSTS"]
    source_topic = os.environ["SOURCE_TOPIC"]
    consumer_group = os.environ["CONSUMER_GROUP"]

    zk_client = get_zookeeper_client(zookeeper_hosts)
    kafka_hosts = get_kafka_hosts(zk_client)

    return KafkaSource(
        kafka_hosts=kafka_hosts, topic=source_topic, consumer_group=consumer_group
    )


def get_text(signal):
    text = signal.get("text", "")
    text = text.strip()
    return text


def load_vocabulary(file_path):
    return joblib.load(file_path)


class WordTextTransformer:
    def __init__(self, source_vocabulary, target_vocabulary, max_source_length, max_target_length):
        self.source_vocabulary = source_vocabulary
        self.target_vocabulary = target_vocabulary
        self.max_source_length = max_source_length
        self.max_target_length = max_target_length
        self.tokenizer =  TweetTokenizer()

    def tokenize(self, text):
        return self.tokenizer.tokenize(text)

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


def preprocess_text(phrase):
    from unicodedata import normalize
    re_punc = re.compile('[%s]' % re.escape(string.punctuation))
    re_print = re.compile('[^%s]' % re.escape(string.printable))

    phrase = phrase.lower()
    phrase = normalize('NFD', phrase).encode('ascii', 'ignore')
    phrase = phrase.decode('UTF-8')  # tokenize on white space line = line.split()
    # convert to lowercase
    tokens = phrase.split()
    # remove punctuation from each token
    tokens = [re_punc.sub('', token) for token in tokens]
    # remove non-printable chars form each token
    tokens = [re_print.sub('', token) for token in tokens]
    # remove tokens with numbers in them
    return " ".join(tokens)


def divide_chunks(l, n):
    # looping till length l
    for i in range(0, len(l), n):
        yield l[i:i + n]



def predict_signal(model, transformer, text):
    tokenizer = TweetTokenizer()
    text = preprocess_text(text)
    tokens = tokenizer.tokenize(text)

    token_chunks = divide_chunks(tokens, 2)
    for chunks in token_chunks:
        text = " ".join(chunks)
        xi = transformer.transform_xi(text)
        X = np.array([xi])
        prediction = model.predict(X, verbose=0)[0]
        integers = [np.argmax(vector) for vector in prediction]
        target = list()
        Vocabulary()

        for ix in integers:
            word = transformer.target_vocabulary.decode([ix])[0]
            if word is None or ix == 0:
                break

            target.append(word)

        out = " ".join(target)
        print("source text %s predicted %s" % (text, out))


def main(model_file_path, source_vocabulary_path, target_vocabulary_path, max_source_length,
         max_target_length):
    source_vocabulary = load_vocabulary(source_vocabulary_path)
    target_vocabulary = load_vocabulary(target_vocabulary_path)

    transformer = WordTextTransformer(source_vocabulary=source_vocabulary, max_source_length=max_source_length,
                                      max_target_length=max_target_length, target_vocabulary=target_vocabulary)

    model = load_model(model_file_path)

    source = get_ingestion_source()
    for signal in source:
        text = get_text(signal)
        locale = signal["locale"]

        if text and locale == LOCALE_SPANISH:
            predict_signal(model, transformer, text)
        else:
            continue


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--model_file_path', required=True)
    parser.add_argument('--source_vocabulary_path', required=True)
    parser.add_argument('--target_vocabulary_path', required=True)
    parser.add_argument('--max_source_length', required=True, type=int)
    parser.add_argument('--max_target_length', required=True, type=int)

    args = parser.parse_args()
    main(
        model_file_path=args.model_file_path,
        source_vocabulary_path=args.source_vocabulary_path,
        target_vocabulary_path=args.target_vocabulary_path,
        max_source_length=args.max_source_length,
        max_target_length=args.max_target_length
    )
