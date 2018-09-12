from keras import Input, Model
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, UpSampling2D


def get_model(height, width, num_channels):
    input_img = Input(name='input', shape=(height, width, num_channels))
    X = Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same')(input_img)
    X = MaxPooling2D(pool_size=(2, 2), padding='same')(X)
    X = Conv2D(filters=8, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same')(X)
    X = MaxPooling2D(pool_size=(2, 2), padding='same')(X)
    X = Conv2D(filters=8, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same')(X)
    encoded = MaxPooling2D(pool_size=(2, 2), padding='same')(X)

    X = Flatten()(X)
    X = Dense(128, activation='relu')(X)
    y = Dense(10, name='classifier', activation='softmax')(X)

    X = Conv2D(filters=8, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same')(encoded)
    X = UpSampling2D(size=(2, 2))(X)
    X = Conv2D(filters=8, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same')(X)
    X = UpSampling2D(size=(2, 2))(X)
    X = Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), activation='relu')(X)
    X = UpSampling2D(size=(2, 2))(X)
    decoded = Conv2D(name='decoder', filters=1, kernel_size=(3, 3), strides=(1, 1), activation='sigmoid',
                     padding='same')(X)

    autoencoder = Model(inputs=input_img, outputs=[decoded, y])
    autoencoder.compile(loss={'classifier': 'mse', 'decoder': 'binary_crossentropy'}, optimizer='adam')
    classifier = Model(input=input_img, output=y)
    print(autoencoder.summary())
    print(classifier.summary())
    return autoencoder, classifier