import numpy as np
from keras.models import Model
from keras.layers import Dense, Dropout, Input


def get_model(max_document_length, num_classes):
    input_layer = Input(shape=(max_document_length,))
    fc_1 = Dense(128, activation='relu')(input_layer)
    fc_1 = Dropout(0.5)(fc_1)
    fc_2 = Dense(64, activation='relu')(fc_1)
    preds = Dense(1, activation='sigmoid')(fc_2)

    model = Model(inputs=input_layer, outputs=preds)
    model.compile(loss='binary_crossentropy', optimizer='rmsprop')
    return model


def main():
    max_document_length = 20
    num_classes = 2
    x_train = np.random.random(size=(1000, max_document_length))
    y_train = np.random.randint(low=0, high=num_classes, size=(1000, 1))
    x_test = np.random.random(size=(500, max_document_length))
    y_test = np.random.randint(low=0, high=num_classes, size=(500, 1))
    model = get_model(max_document_length, num_classes)
    model.fit(x_train, y_train)


if __name__ == '__main__':
    main()
