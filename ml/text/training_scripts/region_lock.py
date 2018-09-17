from argparse import ArgumentParser

import gensim
from nltk.tokenize.casual import TweetTokenizer

ENGLISH_STOPWORDS = {
    "i",
    "me",
    "my",
    "myself",
    "we",
    "our",
    "ours",
    "ourselves",
    "you",
    "your",
    "yours",
    "yourself",
    "yourselves",
    "he",
    "him",
    "his",
    "himself",
    "she",
    "her",
    "hers",
    "herself",
    "it",
    "its",
    "itself",
    "they",
    "them",
    "their",
    "theirs",
    "themselves",
    "what",
    "which",
    "who",
    "whom",
    "this",
    "that",
    "these",
    "those",
    "am",
    "is",
    "are",
    "was",
    "were",
    "be",
    "been",
    "being",
    "have",
    "has",
    "had",
    "having",
    "do",
    "does",
    "did",
    "doing",
    "a",
    "an",
    "the",
    "and",
    "but",
    "if",
    "or",
    "because",
    "as",
    "until",
    "while",
    "of",
    "at",
    "by",
    "for",
    "with",
    "about",
    "against",
    "between",
    "into",
    "through",
    "during",
    "before",
    "after",
    "above",
    "below",
    "to",
    "from",
    "up",
    "down",
    "in",
    "out",
    "on",
    "off",
    "over",
    "under",
    "again",
    "further",
    "then",
    "once",
    "here",
    "there",
    "when",
    "where",
    "why",
    "how",
    "all",
    "any",
    "both",
    "each",
    "few",
    "more",
    "most",
    "other",
    "some",
    "such",
    "no",
    "nor",
    "not",
    "only",
    "own",
    "same",
    "so",
    "than",
    "too",
    "very",
    "s",
    "t",
    "can",
    "will",
    "just",
    "don",
    "should",
    "now",
    "d",
    "ll",
    "m",
    "o",
    "re",
    "ve",
    "y",
    "ain",
    "aren",
    "couldn",
    "didn",
    "doesn",
    "hadn",
    "hasn",
    "haven",
    "isn",
    "ma",
    "mightn",
    "mustn",
    "needn",
    "shan",
    "shouldn",
    "wasn",
    "weren",
    "won",
    "wouldn",
}


def preprocess(text):
    tokenizer = TweetTokenizer()

    # Remove stopwords.
    tokens = tokenizer.tokenize(text)
    tokens = [token for token in tokens if token not in ENGLISH_STOPWORDS and token.isalpha()]
    return tokens


SAMPLE_TEXTS = [
"Praying hard for everyone along the coast as this storm approaches...especially my cousin and her sweet baby who are in Wilmington.",
"Friends, please keep the Carolinas in your prayers as we prepare for hurricane Florence! 🙏🏻"
"#Breaking - Gov. Henry McMaster has issued evacuation orders for the entire S.C. coastline starting at noon Tuesday. https://t.co/wvOnPWxQa1",

]

FLORENCE_TEXTS = [
    "#Breaking - Gov. Henry McMaster has issued evacuation orders for the entire S.C. coastline starting at noon Tuesday. https://t.co/wvOnPWxQa1",
    "Praying for North Carolina and all her peeps! From a friend in Louisiana",
    "My dear friend Tracey @potted_and_planted lives on the coast in North Carolina. Please send all the positive juju you can muster! Her home is in the direct path of the storm."
]


def main():
    embedding_file_path = "glove_w2v.txt"
    model =  gensim.models.KeyedVectors.load_word2vec_format(embedding_file_path)

    for sample_text in SAMPLE_TEXTS:
        sample_tokens = preprocess(sample_text)
        for florence_text in FLORENCE_TEXTS:
            florence_tokens = preprocess(florence_text)
            distance = model.wmdistance(florence_tokens, sample_tokens)
            print("Sample %s Florence %s" % (sample_text, florence_text))
            print("distance", distance)


if __name__ == "__main__":
    main()
