from keras import Input, Model
from keras.layers import LSTM, Dense


class CharLevelLanguageModelExperiment:
    def get_model(self, max_document_length, vocabulary_size):
        input = Input(shape=(max_document_length, vocabulary_size))
        x = LSTM(units=75)(input)
        y = Dense(vocabulary_size, activation="softmax")(x)

        model = Model(inputs=input, outputs=y)
        print(model.summary())
        return model
