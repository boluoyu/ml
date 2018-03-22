import keras.backend as K
from keras.engine import Layer
from keras import Input, Model
from keras.layers import CuDNNLSTM, LSTM, Bidirectional, Lambda, Concatenate, Embedding, TimeDistributed
from keras.regularizers import l2
from keras import initializers, regularizers, constraints


class AttentionWithContext(Layer):
    """
        Attention operation, with a context/query vector, for temporal data.
        Supports Masking.
        Follows the work of Yang et al. [https://www.cs.cmu.edu/~diyiy/docs/naacl16.pdf]
        "Hierarchical Attention Networks for Document Classification"
        by using a context vector to assist the attention
        # Input shape
            3D tensor with shape: `(samples, steps, features)`.
        # Output shape
            2D tensor with shape: `(samples, features)`.
        :param kwargs:
        Just put it on top of an RNN Layer (GRU/LSTM/SimpleRNN) with return_sequences=True.
        The dimensions are inferred based on the output shape of the RNN.
        Example:
            model.add(LSTM(64, return_sequences=True))
            model.add(AttentionWithContext())
        """

    def __init__(self, init='glorot_uniform', kernel_regularizer=None, bias_regularizer=None, kernel_constraint=None,
                 bias_constraint=None, **kwargs):
        self.supports_masking = True
        self.init = initializers.get(init)
        self.kernel_initializer = initializers.get('glorot_uniform')

        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)

        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)

        super(AttentionWithContext, self).__init__(**kwargs)

    def build(self, input_shape):
        self.kernel = self.add_weight((input_shape[-1], 1),
                                      initializer=self.kernel_initializer,
                                      name='{}_W'.format(self.name),
                                      regularizer=self.kernel_regularizer,
                                      constraint=self.kernel_constraint)
        self.b = self.add_weight((input_shape[1],),
                                 initializer='zero',
                                 name='{}_b'.format(self.name),
                                 regularizer=self.bias_regularizer,
                                 constraint=self.bias_constraint)

        self.u = self.add_weight((input_shape[1],),
                                 initializer=self.kernel_initializer,
                                 name='{}_u'.format(self.name),
                                 regularizer=self.kernel_regularizer,
                                 constraint=self.kernel_constraint)
        self.built = True

    def compute_mask(self, input, mask):
        return None

    def call(self, x, mask=None):
        # (batch, timesteps, dims) x (dims, 1)
        uit = K.dot(x, self.kernel)  # (batch, timesteps, 1)
        uit = K.squeeze(uit, -1)  # (batch, timesteps)
        uit = uit + self.b  # (batch, timesteps) + (timesteps,)

        uit = K.tanh(uit)  # (batch, timesteps)

        ait = uit * self.u  # (batch, timesteps) * (timesteps, 1) => (x, 1)
        a = K.exp(ait)  # (batch, 1)

        # apply mask after the exp. will be re-normalized next
        if mask is not None:
            mask = K.cast(mask, K.floatx())  # (batch, timesteps)
            a = mask * a  # (batch, timesteps) * (batch, timesteps, )

        # in some cases especially in the early stages of training the sum may be almost zero
        # and this results in NaN's. A workaround is to add a very small positive number ε to the sum.
        # a /= K.cast(K.sum(a, axis=1, keepdims=True), K.floatx())
        a /= K.cast(K.sum(a, axis=1, keepdims=True) + K.epsilon(), K.floatx())
        a = K.expand_dims(a)
        weighted_input = x * a
        return K.sum(weighted_input, axis=1)

    def compute_output_shape(self, input_shape):
        return input_shape[0], input_shape[-1]


def get_model(max_document_length,
              max_token_length,
              vocab_size,
              char_embedding=16,
              token_embedding=300,
              document_embedding=512,
              initializer='glorot_uniform',
              regularizer=l2(1e-4)):
    document = Input(shape=(max_document_length, max_token_length))

    char_embeddings = Embedding(vocab_size, char_embedding)(document)

    word_embeddings = TimeDistributed(
        CuDNNLSTM(
            token_embedding,
            return_sequences=False,
            kernel_initializer=initializer,
            kernel_regularizer=regularizer
        )
    )(char_embeddings)

    document_embedding = Bidirectional(
        CuDNNLSTM(
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


def euclidean_distance(vectors):
    x, y = vectors
    return K.sqrt(K.maximum(K.sum(K.square(x - y), axis=1, keepdims=True), K.epsilon()))


def triplet_loss(_, y_pred):
    margin = K.constant(1)
    return K.mean(K.maximum(K.constant(0), K.square(y_pred[:, 0]) - K.square(y_pred[:, 1]) + margin))


def triplet_accuracy(_, y_pred):
    return K.mean(y_pred[:, 0] < y_pred[:, 1])


def triplet_model(encoder, input_shape):
    x_anchor = Input(shape=input_shape, name='anchor')
    x_related = Input(shape=input_shape, name='related')
    x_unrelated = Input(shape=input_shape, name='unrelated')

    h_anchor = encoder(x_anchor)
    h_related = encoder(x_related)
    h_unrelated = encoder(x_unrelated)

    related_dist = Lambda(euclidean_distance, name='pos_dist')([h_anchor, h_related])
    unrelated_dist = Lambda(euclidean_distance, name='neg_dist')([h_anchor, h_unrelated])

    inputs = [x_anchor, x_related, x_unrelated]
    distances = Concatenate()([related_dist, unrelated_dist])

    model = Model(inputs=inputs, outputs=distances)
    return model


if __name__ == '__main__':
    model = get_model(max_document_length=757, max_token_length=20, vocab_size=10000)
    model.summary()
