import keras
import keras.backend as K
import tensorflow as tf

import numpy as np
from keras import Model, Input
from keras.datasets import mnist
from keras.callbacks import TensorBoard
from keras.layers import Conv2D, MaxPooling2D, UpSampling2D, Dense, Flatten
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve



height = 28
width = 28
num_channels = 1
num_classes = 10

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
decoded = Conv2D(name='decoder', filters=1, kernel_size=(3, 3), strides=(1, 1), activation='sigmoid', padding='same')(X)

autoencoder = Model(inputs=input_img, outputs=[decoded, y])
autoencoder.compile(loss={'classifier': 'mse', 'decoder': 'binary_crossentropy'}, optimizer='adam')
classifier = Model(input=input_img, output=y)
print(autoencoder.summary())
print(classifier.summary())


def display_predictions(x_test, y_test, n=10):
    predictions = autoencoder.predict(x_test)

    plt.figure(figsize=(20, 4))
    for i in range(n):
        print('predictions', np.argmax(predictions[1][i]), 'true', np.argmax(y_test[i]))
        # display original
        ax = plt.subplot(2, n, i + 1)
        plt.imshow(x_test[i].reshape(28, 28))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        # display reconstruction
        ax = plt.subplot(2, n, i + 1 + n)
        plt.imshow(predictions[0][i].reshape(28, 28))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    plt.show()




def main():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)

    x_train = x_train.astype('float32') / 255.
    x_test = x_test.astype('float32') / 255.
    x_train = np.reshape(x_train, (
    len(x_train), height, width, num_channels))  # adapt this if using `channels_first` image data format
    x_test = np.reshape(x_test, (
    len(x_test), height, width, num_channels))  # adapt this if using `channels_first` image data format

    autoencoder.fit(x_train, {'decoder': x_train, 'classifier': y_train},
                    epochs=25,
                    batch_size=128,
                    shuffle=True,
                    validation_data=(x_test, {'decoder': x_test, 'classifier': y_test}),
                    callbacks=[TensorBoard(log_dir='/tmp/autoencoder')])

    autoencoder.save('/Users/jnewman/Projects/Banjo/ml/ml/reinforcement/scripts/models/autoencoder_ckpt.h5')
    classifier.save('/Users/jnewman/Projects/Banjo/ml/ml/reinforcement/scripts/models/classifier_ckpt.h5')
    display_predictions(x_test, y_test)
    predictions = autoencoder.predict(x_test)
    classifier_predictions = predictions[1]
    classifier_predictions = list(map(np.argmax, classifier_predictions))
    classifier_predictions = np.array(classifier_predictions, dtype='float32')
    y_test = list(map(np.argmax, y_test))
    y_test = np.array(y_test, dtype='float32')

    recall, precision, thresholds = precision_recall_curve(y_true=y_test, probas_pred=classifier_predictions)

    plt.step(recall, precision, color='b', alpha=0.2,
             where='post')
    plt.fill_between(recall, precision, step='post', alpha=0.2,
                     color='b')

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title('2-class Precision-Recall curve')



if __name__ == '__main__':
    main()
