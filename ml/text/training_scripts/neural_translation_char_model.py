from __future__ import print_function

from keras.callbacks import TensorBoard, ReduceLROnPlateau, ModelCheckpoint
from keras.utils import plot_model

from ml.common.callback.keras import GradientDebugger
from ml.text.experiment.neural_translation_char import NeuralTranslationCharacterExperiment
import numpy as np

from ml.text.transformer.vocabulary import Vocabulary


def decode_sequence(encoder_model, decoder_model, max_decoder_seq_length, target_vocabulary, num_decoder_tokens,
                    input_seq):
    # Next: inference mode (sampling).
    # Here's the drill:
    # 1) encode input and retrieve initial decoder state
    # 2) run one step of decoder with this initial state
    # and a "start of sequence" token as target.
    # Output will be the next target token
    # 3) Repeat with the current target token and current states

    # Encode the input as state vectors.
    states_value = encoder_model.predict(input_seq)

    # Generate empty target sequence of length 1.
    target_seq = np.zeros((1, 1, num_decoder_tokens))
    # Populate the first character of target sequence with the start character.
    target_seq[0, 0, target_vocabulary.dictionary.get('\t')] = 1.

    # Sampling loop for a batch of sequences
    # (to simplify, here we assume a batch of size 1).
    stop_condition = False
    decoded_sentence = ''
    while not stop_condition:
        output_tokens, h, c = decoder_model.predict(
            [target_seq] + states_value)

        # Sample a token
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_char = target_vocabulary.inverted_dictionary.get(sampled_token_index)
        decoded_sentence += sampled_char

        # Exit condition: either hit max length
        # or find stop character.
        if (sampled_char == '\n' or
                    len(decoded_sentence) > max_decoder_seq_length):
            stop_condition = True

        # Update the target sequence (of length 1)
        target_seq = np.zeros((1, 1, num_decoder_tokens))
        target_seq[0, 0, sampled_token_index] = 1

        # Update states
        states_value = [h, c]

    return decoded_sentence


def prepare_data(lines, num_samples, switch=True):
    source_texts = []
    target_texts = []
    source_characters = set()
    target_characters = set()

    for line in lines[: min(num_samples, len(lines) - 1)]:
        _source_text, _target_text = line.split('\t')

        if switch:
            source_text = _target_text
            target_text = _source_text
        else:
            source_text = _source_text
            target_text = _target_text

        # We use "tab" as the "start sequence" character
        # for the targets, and "\n" as "end sequence" character.
        target_text = '\t' + target_text + '\n'
        source_texts.append(source_text)
        target_texts.append(target_text)

        for char in source_text:
            if char not in source_characters:
                source_characters.add(char)

        for char in target_text:
            if char not in target_characters:
                target_characters.add(char)

    return source_texts, target_texts, source_characters, target_characters


def load_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.read().split('\n')
        return lines


def main(model_name, model_file_path, num_epochs=100, batch_size=64, num_samples=1000000):
    data_path = '/Users/jnewman/Projects/learning/ai_blog/ml/data/translation/spa-eng/spa.txt'
    experiment = NeuralTranslationCharacterExperiment()
    lines = load_data(data_path)
    source_texts, target_texts, source_characters, target_characters = prepare_data(lines, num_samples)

    source_characters = sorted(list(source_characters))
    target_characters = sorted(list(target_characters))

    source_vocabulary = Vocabulary()
    source_vocabulary.fit(tokens=source_characters)
    target_vocabulary = Vocabulary()
    target_vocabulary.fit(tokens=target_characters)

    source_vocabulary_size = len(source_vocabulary)
    target_vocabulary_size = len(target_vocabulary)
    max_source_seq_length = max([len(txt) for txt in source_texts])
    max_target_seq_length = max([len(txt) for txt in target_texts])

    print('Number of samples:', len(source_texts))
    print('Number of unique input tokens:', source_vocabulary_size)
    print('Number of unique output tokens:', target_vocabulary_size)
    print('Max sequence length for inputs:', max_source_seq_length)
    print('Max sequence length for outputs:', max_target_seq_length)

    encoder_input_data = np.zeros(
        (len(source_texts), max_source_seq_length, source_vocabulary_size),
        dtype='float32')
    decoder_input_data = np.zeros(
        (len(source_texts), max_target_seq_length, target_vocabulary_size),
        dtype='float32')
    decoder_target_data = np.zeros(
        (len(source_texts), max_target_seq_length, target_vocabulary_size),
        dtype='float32')

    for i, (source_text, target_text) in enumerate(zip(source_texts, target_texts)):
        for t, char in enumerate(source_text):
            source_vocab_ix = source_vocabulary.dictionary.get(char)
            encoder_input_data[i, t, source_vocab_ix] = 1.
        for t, char in enumerate(target_text):
            # decoder_target_data is ahead of decoder_input_data by one timestep
            target_vocab_ix = target_vocabulary.dictionary.get(char)
            decoder_input_data[i, t, target_vocab_ix] = 1.
            if t > 0:
                # decoder_target_data will be ahead by one timestep
                # and will not include the start character.
                decoder_target_data[i, t - 1, target_vocabulary.dictionary.get(char)] = 1.

    models = experiment.get_lstm_model(source_vocabulary_length=source_vocabulary_size,
                                       target_vocabulary_length=target_vocabulary_size)
    training_model = models["training_model"]
    decoder_model = models["decoder"]
    encoder_model = models["encoder"]
    # Run training
    training_model.compile(optimizer='rmsprop', loss='categorical_crossentropy')

    plot_model(training_model, to_file='char_translator.png', show_shapes=True)

    callbacks = [
        GradientDebugger(),
        TensorBoard(log_dir='/tmp/char_translator'),
        ReduceLROnPlateau(factor=0.01, verbose=1),
        ModelCheckpoint(model_name + '.h5', monitor='val_loss', verbose=1, save_best_only=True, mode='min')
    ]

    training_model.fit([encoder_input_data, decoder_input_data], decoder_target_data,
                       batch_size=batch_size,
                       epochs=num_epochs,
                       callbacks=callbacks,
                       validation_split=0.2,
                       shuffle=True)
    # Save model
    training_model.save('s2s.h5')

    for seq_index in range(100):
        # Take one sequence (part of the training set)
        # for trying out decoding.
        input_seq = encoder_input_data[seq_index: seq_index + 1]
        decoded_sentence = decode_sequence(
            input_seq=input_seq,
            decoder_model=decoder_model,
            max_decoder_seq_length=max_target_seq_length,
            target_vocabulary=target_vocabulary,
            num_decoder_tokens=target_vocabulary_size,
            encoder_model=encoder_model
        )
        print('-')
        print('Input sentence:', source_texts[seq_index])
        print('Decoded sentence:', decoded_sentence)


if __name__ == "__main__":
    main(model_name="s2s.h5", model_file_path="temp.f")
