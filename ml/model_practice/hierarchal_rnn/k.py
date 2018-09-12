from argparse import ArgumentParser

from keras.models import Model
from keras.layers import Input, Embedding, LSTM, Bidirectional, TimeDistributed, Dense, Dropout


def get_model():
    char_embedding_size = 16
    word_embedding_size = 100
    max_document_length = 100

    max_token_length = 20
    vocab_size = 100
    document = Input(shape=(max_document_length, max_token_length))  # batch size, n words, num_chars
    x = Embedding(vocab_size, char_embedding_size)(document)
    x = TimeDistributed(LSTM(word_embedding_size, return_sequences=False, recurrent_dropout=0.2))(x)
    x = Dropout(0.2)(x)
    x = Dense(128)(x)
    y = Dense(num_classes)(x)
    return Model(input=document, outputs=y) # could add an optional attention mechanism here


def main():
    model = get_model()
    print(model.summary())


if __name__ == '__main__':
    main()
