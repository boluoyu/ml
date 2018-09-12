from keras import Input, backend as K, Model
from keras.layers import Embedding, TimeDistributed, LSTM, Bidirectional, Lambda
from keras.regularizers import l2

from ml.text.layer.attention import AttentionWithContext


def get_model(max_document_length,
              max_token_length,
              vocab_size,
              char_embedding=100,
              token_embedding=128,
              document_embedding=156,
              initializer='glorot_uniform',
              regularizer=l2(1e-4)):
    document = Input(shape=(max_document_length, max_token_length))

    char_embeddings = Embedding(vocab_size, char_embedding)(document)

    word_embeddings = TimeDistributed(
        LSTM(
            token_embedding,
            return_sequences=False,
            kernel_initializer=initializer,
            kernel_regularizer=regularizer
        )
    )(char_embeddings)

    document_embedding = Bidirectional(
        LSTM(
            document_embedding // 2,  # divide by two to get document_embedding from Bidirectional concat
            return_sequences=True,
            kernel_initializer=initializer,
            kernel_regularizer=regularizer
        )
    )(word_embeddings)

    context = AttentionWithContext(init=initializer, kernel_regularizer=regularizer)(document_embedding)
    embedding = Lambda(lambda x: K.l2_normalize(x, axis=-1))(context)
    model = Model(document, embedding)
    return model