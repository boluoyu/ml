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
