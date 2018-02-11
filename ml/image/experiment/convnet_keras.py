from keras.layers import Dense, Flatten, Activation, Conv2D
from keras.models import Sequential
from keras.optimizers import Adam

from ml.image.experiment.base import ImageExperiment


class ConvnetKerasImageExperiment(ImageExperiment):
    name = 'convnet_keras_64x64x3_0.0.01_mse_adam'

    height = 64
    width = 64
    num_channels = 3
    num_targets = 2
    learning_rate = 0.001
    loss_function = 'mse'
    optimizer = Adam(lr=learning_rate)

    def get_model(self):
        model = Sequential()
        model.add(Conv2D(filters=32, kernel_size=8, strides=(4, 4), padding='same',
                         input_shape=(self.height, self.width, self.num_channels)))
        model.add(Activation('relu'))
        model.add(Conv2D(filters=64, kernel_size=4, strides=(2, 2), padding='same'))
        model.add(Activation('relu'))
        model.add(Conv2D(filters=64, kernel_size=3, strides=(1, 1), padding='same'))
        model.add(Activation('relu'))
        model.add(Flatten())
        model.add(Dense(512))
        model.add(Activation('relu'))
        model.add(Dense(self.num_targets))
        model.compile(
            loss=self.loss_function,
            optimizer=self.optimizer
        )

        print(model.summary())
        return model
