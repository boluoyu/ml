import numpy as np
from keras.callbacks import TensorBoard

from ml.text.transformer.vocabulary import Vocabulary

from keras.datasets import reuters
from keras.models import Model
from keras.layers import Input, Conv1D, MaxPooling1D, Flatten, Dense, Dropout
from keras.optimizers import RMSprop

""" (70 features)
The alphabet used in all of our models consists of 70 characters, including 26 english letters, 10
digits, 33 other characters and the new line character. The non-space characters are:
abcdefghijklmnopqrstuvwxyz0123456789
-,;.!?:’’’/\|_@#$%ˆ&*˜‘+-=<>()[]{}
"""

"""
1014 max chars
"""


class Transformer:
    def __init__(self, max_sequence_length, vocabulary_size, class_map):
        self._max_sequence_length = max_sequence_length
        self._vocabulary_size = vocabulary_size
        self._class_map = class_map
        self._num_classes = len(class_map)

    def transform_X(self, X):
        X = list(map(self._transform_xi, X))
        X = np.array(X)
        X = np.reshape(X, (len(X), self._max_sequence_length, self._vocabulary_size))
        return X

    def transform_y(self, y):
        y = list(map(self._transform_yi, y))
        y = np.array(y)
        y = np.reshape, (len(y), self._num_classes)
        return y

    def _transform_xi(self, xi):
        xi = xi[:self._max_sequence_length]
        transformed_xi = np.zeros(shape=(self._max_sequence_length, self._vocabulary_size))

        for ix, encoded_token_ix in enumerate(xi):
            if encoded_token_ix >= self._vocabulary_size:
                encoded_token_ix = 0

            transformed_xi[ix, encoded_token_ix] = 1

        return transformed_xi

    def _transform_yi(self, yi):
        transformed_yi = np.zeros(shape=(self._num_classes,))
        transformed_yi[self._class_map[yi]] = 1
        return yi


class DataGenerator:
    def __init__(self, X, y, transformer):
        self.X = X
        self.y = y
        self.transformer = transformer

    def get_training_data(self):
        return self.transformer.transform_X(self.X), self.transformer.transform_y(self.y)


def get_model(metrics, max_sequence_length, input_vocabulary_size, num_classes,
              filter_sizes=(256, 256, 256, 256, 256, 256),  # 6 conv layers
              kernel_sizes=(7, 7, 3, 3, 3, 3),
              pool_sizes=(3, 3, None, None, 3),
              hidden_sizes=(1024, 1024, 1024),  # 3 FC layers
              droput=0.5,  # Dropout between FC layers
              learning_rate=0.001):
    document = Input(shape=(max_sequence_length, input_vocabulary_size))  # Sequence of encoded characters as input
    features = document

    for i, (filter_size, kernel_size, pool_size) in enumerate(zip(filter_sizes, kernel_sizes, pool_sizes), 1):
        features = Conv1D(
            filters=filter_size,
            kernel_size=(kernel_size,),
            strides=(1,),
            activation='elu',
            name='conv_layer_' + str(i)
        )(features)
        if pool_size:
            features = MaxPooling1D(pool_size)(features)

    features = Flatten()(features)

    for i, hidden_size in enumerate(hidden_sizes):
        features = Dense(hidden_size)(features)
        features = Dropout(droput)(features)

    predictions = Dense(num_classes, activation='softmax', name='predictions')(features)
    optimizer = RMSprop(lr=learning_rate)
    model = Model(document, predictions)
    model.compile(optimizer,
                  loss='categorical_crossentropy',
                  metrics=metrics)
    print(model.summary())
    return model


def get_class_map(labels):
    labels = set(labels)
    return {cls_ix: ix for ix, cls_ix in enumerate(labels)}


def get_tokens():
    tokens = "abcdefghijklmnopqrstuvwxyz0123456789-,;.!?:’’’/\|_@#$%ˆ&*˜‘+-=<>()[]{}"
    return [token for token in tokens]


def main():
    MAX_SEQUENCE_LENGTH = 1014  # from paper
    NUM_CLASSES = 46  # 46 reuters topics
    input_vocabulary = Vocabulary()
    input_vocabulary_size = 1000

    (x_train, y_train), (x_test, y_test) = reuters.load_data(path="reuters.npz",
                                                             num_words=None,
                                                             skip_top=0,
                                                             maxlen=None,
                                                             test_split=0.2,
                                                             seed=113,
                                                             start_char=1,
                                                             oov_char=2,
                                                             index_from=3)
    model = get_model(
        filter_sizes=(256,),
        kernel_sizes=(7,),
        pool_sizes=(3,),
        hidden_sizes=(1056,),
        input_vocabulary_size=input_vocabulary_size,
        num_classes=NUM_CLASSES,
        max_sequence_length=MAX_SEQUENCE_LENGTH,
        metrics=['accuracy']
    )
    print(model.summary())
    class_map = get_class_map(np.concatenate([y_train, y_test]))
    transformer = Transformer(max_sequence_length=MAX_SEQUENCE_LENGTH, vocabulary_size=input_vocabulary_size,
                              class_map=class_map)
    training_generator = DataGenerator(X=x_train, y=y_train, transformer=transformer)
    X, y = training_generator.get_training_data()
    model.fit(x=X, y=y, epochs=1, verbose=1, callbacks=[TensorBoard(log_dir='/tmp/deep_conv')])


if __name__ == '__main__':
    main()
