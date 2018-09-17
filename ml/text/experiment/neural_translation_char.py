from keras import Input, Model
from keras.layers import LSTM, Dense, GRU


class NeuralTranslationCharacterExperiment:
    def get_lstm_model(self, source_vocabulary_length, target_vocabulary_length, encoder_latent_dimensions=256,
                       decoder_latent_dimensions=256):
        encoder_inputs = Input(shape=(None, source_vocabulary_length))
        encoder = LSTM(encoder_latent_dimensions, return_state=True)
        encoder_outputs, state_h, state_c = encoder(encoder_inputs)
        # We discard `encoder_outputs` and only keep the states.
        encoder_states = [state_h, state_c]

        decoder_inputs = Input(shape=(None, target_vocabulary_length))
        # We set up decoder to use full output sequences and return internal states
        # We use the returned decoder states for inference
        decoder_lstm = LSTM(decoder_latent_dimensions, return_sequences=True, return_state=True)
        decoder_outputs, _, _ = decoder_lstm(decoder_inputs, initial_state=encoder_states)
        decoder_dense = Dense(target_vocabulary_length, activation="softmax")
        decoder_outputs = decoder_dense(decoder_outputs)

        # Define the model that will turn
        # `encoder_input_data` & `decoder_input_data` into `decoder_target_data`
        model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
        model.compile(optimizer='rmsprop', loss='categorical_crossentropy')
        print("training model", model.summary())

        encoder_model = Model(encoder_inputs, encoder_states)

        decoder_state_input_h = Input(shape=(decoder_latent_dimensions,))
        decoder_state_input_c = Input(shape=(decoder_latent_dimensions,))

        decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
        decoder_outputs, state_h, state_c = decoder_lstm(decoder_inputs, initial_state=decoder_states_inputs)
        decoder_states = [state_h, state_c]
        decoder_outputs = decoder_dense(decoder_outputs)
        decoder_model = Model([decoder_inputs] + decoder_states_inputs, [decoder_outputs] + decoder_states)

        return {
            "training_model": model,
            "encoder": encoder_model,
            "decoder": decoder_model
        }

    def get_gru_model(self, source_vocabulary_length, target_vocabulary_length, encoder_latent_dimensions=256,
                      decoder_latent_dimensions=256):
        encoder_inputs = Input(shape=(None, source_vocabulary_length))
        encoder = GRU(encoder_latent_dimensions, return_state=True)
        encoder_outputs, state_h = encoder(encoder_inputs)

        decoder_inputs = Input(shape=(None, target_vocabulary_length))
        decoder_gru = GRU(decoder_latent_dimensions, return_sequences=True)
        decoder_outputs = decoder_gru(decoder_inputs, initial_state=state_h)
        decoder_dense = Dense(target_vocabulary_length, activation='softmax')
        decoder_outputs = decoder_dense(decoder_outputs)
        model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
        model.compile(optimizer='rmsprop', loss='categorical_crossentropy')
        print(model.summary())

        print("training model", model.summary())

        encoder_model = Model(encoder_inputs, state_h)

        decoder_state_input_h = Input(shape=(decoder_latent_dimensions,))
        decoder_state_input_c = Input(shape=(decoder_latent_dimensions,))

        decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
        decoder_outputs, state_h, state_c = decoder_gru(decoder_inputs, initial_state=decoder_states_inputs)
        decoder_states = [state_h, state_c]
        decoder_outputs = decoder_dense(decoder_outputs)
        decoder_model = Model(
            [decoder_inputs] + decoder_states_inputs,
            [decoder_outputs] + decoder_states)

        return {
            "training_model": model,
            "encoder":        encoder_model,
            "decoder":        decoder_model
        }
