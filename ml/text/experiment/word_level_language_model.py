from keras import Input, Model
from keras.layers import LSTM, Dense, Embedding


class WordLevelLanguageModelExperiment:
    def get_model(self, max_document_length, vocabulary_size, embedding_size=10):
        input = Input(shape=(max_document_length,))
        x = Embedding(vocabulary_size, embedding_size)(input)
        x = LSTM(units=50)(x)
        y = Dense(vocabulary_size, activation="softmax")(x)

        model = Model(inputs=input, outputs=y)
        print(model.summary())
        return model
