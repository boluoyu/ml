from keras import Model
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D
from keras.optimizers import Adam
import keras.backend as K

from ml.common.classifier.keras import KerasClassifier
from ml.common.evaluator.base import Evaluator
from ml.image.experiment.base import ImageExperiment


class AutoencoderImageExperiment(ImageExperiment):
    name = 'autoencoder_keras'

    height = 64
    width = 64
    num_channels = 3
    num_targets = 2
    learning_rate = 0.001
    loss_function = 'mse'
    optimizer = Adam(lr=learning_rate)

    def get_model(self):
        input_img = Input(shape=(self.height, self.width, self.num_channels))
        X = Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same')(input_img)
        X = MaxPooling2D(pool_size=(2, 2), padding='same')(X)
        X = Conv2D(filters=8, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same')(X)
        X = MaxPooling2D(pool_size=(2, 2), padding='same')(X)
        X = Conv2D(filters=8, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same')(X)
        encoded = MaxPooling2D(pool_size=(2, 2), padding='same')(X)

        X = Conv2D(filters=8, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same')(encoded)
        X = UpSampling2D(size=(2, 2))(X)
        X = Conv2D(filters=8, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same')(X)
        X = UpSampling2D(size=(2, 2))(X)
        X = Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), activation='relu')(X)
        X = UpSampling2D(size=(2, 2))(X)
        decoded = Conv2D(filters=1, kernel_size=(3, 3), strides=(1, 1), activation='sigmoid', padding='same')(X)

        autoencoder = Model(inputs=input_img, outputs=decoded)
        autoencoder.compile(optimizer=self.optimizer, loss=self.loss_function)

        print(autoencoder.summary())
        return autoencoder

    def get_classifier(self):
        return KerasClassifier(
            model=self.get_model(),
            transformer=self.transformer
        )

    def get_evaluator(self):
        return Evaluator()
