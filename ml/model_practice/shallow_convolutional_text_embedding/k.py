from argparse import ArgumentParser

from collections import namedtuple
from keras.layers import Input, Embedding, Conv1D, MaxPooling1D, Dense, Flatten
from keras.models import Model
from keras.optimizers import Adam
from keras.utils import plot_model

from ml.embedding.util import get_embedding_index, get_embedding_matrix
from ml.text.transformer.vocabulary import Vocabulary, CharacterTokenizer, WordTokenizer

Sample = namedtuple("sample", ("text", "label"))
TRAINING_DATA = [
    Sample("I am the blue text", 0),
    Sample("I am the red text", 1)
]


def get_model(max_document_length, max_num_tokens, embedding_weights, embedding_dims, num_targets):
    document = Input(shape=(max_document_length,))
    x = Embedding(input_dim=max_num_tokens, output_dim=embedding_dims, input_length=max_document_length, trainable=False, weights=[embedding_weights])(document)
    x = Conv1D(filters=128, kernel_size=5, activation="relu")(x)
    x = MaxPooling1D(pool_size=1)(x)
    x = Flatten()(x)
    x = Dense(units=128, activation="relu")(x)
    y = Dense(units=num_targets, activation="softmax")(x)

    model = Model(inputs=document, outputs=y)
    print(model.summary())
    return model


def get_training_data():
    return TRAINING_DATA


def main(model_name, embedding_path):
    num_targets = 20

    training_data = get_training_data()
    vocabulary = Vocabulary()
    texts = [sample.text for sample in training_data]
    text_token_generator = WordTokenizer(texts=texts)
    tokens = list(text_token_generator.get_tokens())

    max_doc_length = text_token_generator.get_max_length()

    vocabulary.fit(tokens=tokens)
    max_num_tokens = len(vocabulary)
    token_index_map = vocabulary.dictionary

    embedding_index = get_embedding_index(embedding_path)
    embedding_dimensions = len(list(embedding_index.values())[0])

    embedding_matrix = get_embedding_matrix(
        embedding_dimensions=embedding_dimensions,
        embedding_index=embedding_index,
        token_index_mapping=token_index_map
    )


    model = get_model(max_document_length=max_doc_length, max_num_tokens=max_num_tokens,
                      embedding_weights=embedding_matrix, embedding_dims=embedding_dimensions, num_targets=num_targets)
    model.compile(optimizer=Adam(), loss="binary_crossentropy")
    plot_model(model, to_file=model_name + '.png')


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--model_name')
    parser.add_argument('--embedding_path')

    args = parser.parse_args()
    main(args.model_name, args.embedding_path)
