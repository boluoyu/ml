import tensorflow as tf

from ml.common.classifier.tensorflow import TensorflowModel
from ml.image.experiment.base import ImageExperiment


class ConvnetTensorflowImageExperiment(ImageExperiment):
    name = 'convnet_tensorflow_64x64x3_0.0.01_mse_adam'

    height = 64
    width = 64
    num_channels = 3
    num_targets = 2
    learning_rate = 0.001
    loss_function = 'mse'
    optimizer_cls = tf.train.AdamOptimizer

    def get_model(self):
        self.graph = tf.Graph()

        with self.graph.as_default():
            self.session = tf.Session()

            self.input = tf.placeholder(
                name='input_features',
                dtype=tf.float32,
                shape=(None, self.width, self.height, self.num_channels)
            )

            self.targets = tf.placeholder(
                name='targets',
                dtype=tf.float32,
                shape=(None, self.num_targets)
            )

            self.conv1 = tf.layers.conv2d(
                name='conv1',
                filters=32,
                kernel_size=8,
                strides=(4, 4),
                padding='same',
                inputs=self.input,
                activation=tf.nn.relu
            )

            self.conv2 = tf.layers.conv2d(
                name='conv2',
                filters=64,
                kernel_size=4,
                strides=(2, 2),
                padding='same',
                inputs=self.conv1,
                activation=tf.nn.relu
            )

            self.flatten_1 = tf.layers.flatten(
                name='flatten1',
                inputs=self.conv2
            )

            self.dense_1 = tf.layers.dense(
                name='dense1',
                units=512,
                inputs=self.flatten_1,
                activation=tf.nn.relu
            )

            self.dense_2 = tf.layers.dense(
                name='dense_2',
                inputs=self.dense_1,
                units=self.num_targets
            )

            self.predictions = tf.nn.softmax(self.dense_2)

            self.loss = tf.nn.softmax_cross_entropy_with_logits(logits=self.dense_2, labels=self.targets)
            self.optimizer = self.optimizer_cls(learning_rate=self.learning_rate).minimize(self.loss)
            self.saver = tf.train.Saver()

            return TensorflowModel(
                session=self.session,
                saver=self.saver,
                loss=self.loss,
                optimizer=self.optimizer,
                input=self.input,
                targets=self.targets,
                predictions=self.predictions
            )
