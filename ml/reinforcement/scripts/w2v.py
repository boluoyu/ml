import numpy

from argparse import ArgumentParser
from gensim import matutils
from gensim.models import Word2Vec
from gensim.utils import simple_preprocess


CATEGORY_FIRE = 'fire'
CATEGORY_ACCIDENT = 'accident'
CATEGORY_MEDICAL = 'medical'
CATEGORY_WEATHER = 'weather'
CATEGORY_CRIME = 'crime'
CATEGORY_TRAFFIC = 'traffic'
CATEGORY_HAZARD = 'hazard'
CATEGORY_RESCUE = 'rescue'
CATEGORY_OTHER = 'other'
CATEGORY_POWER_OUTAGE = 'power_outage'

INCIDENT_CATEGORIES = {
    CATEGORY_FIRE:     ['fire', 'smoke', 'burn', 'explosion', 'alarm'],
    CATEGORY_ACCIDENT: ['accident', 'collision', 'crash', 'vehicle'],
    CATEGORY_MEDICAL:  ['alarm', 'medical', 'bleed', 'blood', 'cardiac', 'injury', 'ems', 'hospital', 'aid',
                        'fall', 'seizure', 'seizures', 'allergic'],
    CATEGORY_WEATHER:  ['weather', 'storm', 'tornado', 'rain'],
    CATEGORY_CRIME:    ['crime', 'police', 'arrest', 'shoot', 'stab', 'robbery', 'vandalism', 'suspicious', 'stolen',
                        'fight'],
    CATEGORY_TRAFFIC:  ['traffic'],
    CATEGORY_HAZARD:   ['hazard', 'hazardous', 'odor', 'leak', 'fumes', 'wire'],
    CATEGORY_RESCUE:   ['rescue']
}


class IncidentClassifier(object):
    def __init__(self, w2v, category_keywords):
        self._w2v = w2v

        self._categories = []
        category_vectors = []

        for category, tokens in category_keywords.items():
            self._categories.append(category)
            category_vectors.append(self._compute_vector(tokens))

        self._category_vectors = numpy.array(category_vectors)

    def classify(self, tokens):
        vector = self._compute_vector(tokens)

        if vector is None:
            raise ValueError('Unable to vectorize tokens.')

        similarities = numpy.dot(self._category_vectors, vector)
        return dict(zip(self._categories, similarities))

    def _compute_vector(self, tokens):
        w2v_tokens = [token for token in tokens if token in self._w2v]

        if w2v_tokens:
            vector = matutils.unitvec(numpy.mean(self._w2v[w2v_tokens], axis=0))
            return vector
        else:
            return None




def main(w2v_file_path, text):
    w2v = Word2Vec.load(w2v_file_path)
    w2v = IncidentClassifier(w2v, category_keywords=INCIDENT_CATEGORIES)
    tokens = simple_preprocess(text)

    print('classifying tokens', tokens)
    classification = w2v.classify(tokens)
    print('text', text, 'classification', classification)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--w2v_file_path', required=True)
    parser.add_argument('--text', required=True)

    args = parser.parse_args()
    main(args.w2v_file_path, args.text)
