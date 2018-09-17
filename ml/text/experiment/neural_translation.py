from keras import Input, Model
from keras.layers import Dense, Embedding, LSTM, RepeatVector, TimeDistributed
from keras.optimizers import Adam


class NeuralTranslationExperiment:
    def get_model(self, target_vocab_size, source_vocab_size, source_timesteps, target_timesteps):
        input_layer = Input(shape=(source_timesteps,))

        # encoder
        x = Embedding(source_vocab_size, 256, mask_zero=True)(input_layer)
        x = LSTM(128)(x)

        # repeated encodeed representation
        x = RepeatVector(target_timesteps)(x)

        # decoder
        x = LSTM(256, return_sequences=True)(x)
        y = TimeDistributed(Dense(target_vocab_size, activation='softmax'))(x)

        model = Model(inputs=input_layer, outputs=y)
        model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=["accuracy"])
        print(model.summary())
        return model
