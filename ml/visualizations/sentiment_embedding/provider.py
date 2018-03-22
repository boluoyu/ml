import json
import random
from collections import namedtuple, defaultdict

from tqdm import tqdm

Sample = namedtuple('Sample', ('sample_id', 'text', 'airline', 'sentiment'))
Triplet = namedtuple('Triplet', ('anchor', 'related', 'unrelated'))

SENTIMENT_NEGATIVE = 'negative'
SENTIMENT_NEUTRAL = 'neutral'
SENTIMENT_POSITIVE = 'positive'


def load_samples(path):
    with open(path) as f:
        return [Sample(sample['tweet_id'], sample['text'].lower(), sample['airline'], sample['airline_sentiment'])
                for sample in map(json.loads, tqdm(f))]


class TripletProvider:
    def __init__(self, samples, shuffle=False):
        self._samples = samples
        self._shuffle = shuffle

    def __iter__(self):
        if self._shuffle:
            random.shuffle(self._samples)

        positive_samples = self._get_samples(sentiment=SENTIMENT_POSITIVE)
        negative_samples = self._get_samples(sentiment=SENTIMENT_NEGATIVE)
        neutral_samples = self._get_samples(sentiment=SENTIMENT_NEUTRAL)
        unrelated_positive_samples = negative_samples + neutral_samples
        unrelated_negative_samples = positive_samples + neutral_samples

        while True:
            if random.random() > 0.5:
                yield self._get_triplet(positive_samples, unrelated_positive_samples)
            else:
                yield self._get_triplet(negative_samples, unrelated_negative_samples)

    def _get_triplet(self, samples, unrelated_samples):
        anchor = random.choice(samples)
        unrelated = random.choice(unrelated_samples)

        related = None

        while not related or related == anchor:
            related = random.choice(samples)

        return Triplet(anchor, related, unrelated)

    def _get_samples(self, sentiment):
        return [sample for sample in self._samples if sample.sentiment == sentiment]
