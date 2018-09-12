from keras import Input, Model
from keras.layers import Embedding, LSTM, Dense


class TextGenerationExperiment:
    def get_model(self, max_document_length, vocabulary_size, embedding_size=50):
        input_document = Input(shape=(max_document_length,))
        x = Embedding(vocabulary_size, embedding_size)(input_document)
        x = LSTM(100, return_sequences=True)(x)
        x = LSTM(100)(x)
        x = Dense(100, activation="relu")(x)
        y = Dense(vocabulary_size, activation="softmax")(x)

        model = Model(inputs=input_document, outputs=y)
        print(model.summary())
        return model
