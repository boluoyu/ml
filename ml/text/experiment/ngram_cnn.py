from keras import Input, Model
from keras.layers import Conv1D, MaxPooling1D, Flatten, Dropout, Concatenate, Dense, Embedding, concatenate


class NgramCNNModel:
    def get_model(self, max_document_length, vocabulary_size, num_classes):
        embedding = Embedding(vocabulary_size, 100)

        # channel 1: 4-grams
        input_one = Input(shape=(max_document_length,))
        x_1 = embedding(input_one)
        x_1 = Conv1D(filters=32, kernel_size=4)(x_1)
        x_1 = Dropout(0.5)(x_1)
        x_1 = MaxPooling1D(pool_size=2)(x_1)
        x_1 = Flatten()(x_1)

        # channel 1: 6-grams
        input_two = Input(shape=(max_document_length,))
        x_2 = embedding(input_two)
        x_2 = Conv1D(filters=32, kernel_size=6)(x_2)
        x_2 = Dropout(0.5)(x_2)
        x_2 = MaxPooling1D(pool_size=2)(x_2)
        x_2 = Flatten()(x_2)

        # channel 3: 8-grams
        input_three = Input(shape=(max_document_length,))
        x_3 = embedding(input_three)
        x_3 = Conv1D(filters=32, kernel_size=8)(x_3)
        x_3 = Dropout(0.5)(x_3)
        x_3 = MaxPooling1D(pool_size=2)(x_3)
        x_3 = Flatten()(x_3)

        x = concatenate([x_1, x_2, x_3])
        x = Dense(10, activation="relu")(x)
        x = Dropout(0.5)(x)
        y = Dense(num_classes, activation="softmax")(x)

        model = Model(inputs=[input_one, input_two, input_three], output=y)
        return model
