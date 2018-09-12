from keras import Model
from keras.layers import Input, Embedding, LSTM, Dropout, Dense
from keras.optimizers import Adam


def get_model(max_sequence_length, input_vocab_size, vocabulary_size, learning_rate, embedding_dims, num_classes, metrics):
    document = Input(shape=(max_sequence_length, input_vocab_size))
    X = Embedding(vocabulary_size, embedding_dims, mask_zero=True)(document)
    X = LSTM(units=128)(X)
    X = LSTM(units=64)(X)
    X = Dropout(rate=0.5)(X)
    y = Dense(units=num_classes, activation="softmax")(X)
    model = Model(document, y)
    optimizer= Adam(lr=learning_rate)
    model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=metrics)
    return model

