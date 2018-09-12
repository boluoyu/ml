from keras import Input, Model
from keras.layers import Embedding, Bidirectional, LSTM, Dropout


def get_model(max_document_length, input_vocab_size, max_token_length, vocab_size):
    embedding_dims = 32  # dimensionality of dense embedding
    rnn_dims = 128

    document = Input(shape=(None,))
    embedded_sequence = Embedding(input_vocab_size, embedding_dims, mask_zero=True)(document)
    encoded_sequence = Bidirectional(LSTM(rnn_dims, return_sequences=True))(embedded_sequence)
    encoded_sequence = Dropout(rate=0.3)(encoded_sequence)
    document_embedding = Bidirectional(LSTM(rnn_dims))(encoded_sequence)
    model = Model(document, document_embedding)
    return model