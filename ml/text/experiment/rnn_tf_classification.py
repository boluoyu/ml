import tensorflow as tf

from tensorflow.contrib.rnn import DropoutWrapper, GRUCell, MultiRNNCell
from ml.common.classifier.tensorflow import TensorflowModel
from ml.common.experiment.base import Experiment


class RNNTensorflowTextExperiment(Experiment):
    name = 'rnn_tensorflow_text_experiment'

    num_steps = 100
    dropout_rate = 0.5
    num_tokens = 500000
    num_targets = 2
    num_layers = 10
    learning_rate = 0.001
    num_hidden_layer_neurons = 128
    loss_function = 'mse'
    optimizer_cls = tf.train.AdamOptimizer

    def get_model(self):
        self.graph = tf.Graph()

        with self.graph.as_default():
            self.session = tf.Session()

            self.input = tf.placeholder(
                name='input',
                dtype=tf.float32,
                shape=(None, self.num_steps, self.num_tokens)
            )

            self.targets = tf.placeholder(
                name='targets',
                dtype=tf.float32,
                shape=(None, self.num_targets)
            )

            self.cells = MultiRNNCell([self._get_gru_cell()] * self.num_layers)
            self.outputs, self.states = tf.nn.dynamic_rnn(  # 'outputs' is a tensor of shape [batch_size, max_time, cell_state_size]
                cell=self.cells,
                inputs=self.input,
                dtype=tf.float32
            )

            self.output = tf.transpose(a=self.outputs, perm=[1, 0, 2]) # [max time, batch_size, cell_state_size]
            self.last = tf.gather(params=self.outputs, indices=int(self.outputs.get_shape()[0]) - 1)

            self.weights_out = tf.Variable(tf.truncated_normal(shape=[self.num_hidden_layer_neurons, self.num_targets], stddev=0.1))
            self.biases_out = tf.Variable(tf.constant(0.1, shape=[self.num_targets]))

            # Linear activation using rnn inner loops last output
            self.predictions = tf.nn.softmax(logits=tf.matmul(a=self.last, b=self.weights_out) + self.biases_out)
            self.optimizer = self.optimizer_cls(learning_rate=self.learning_rate)
            self.loss = -tf.reduce_mean(input_tensor=self.num_targets * tf.log(self.predictions))
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

    def _get_gru_cell(self):
        cell = GRUCell(
            num_units=self.num_hidden_layer_neurons,
            reuse=tf.get_variable_scope().reuse
        )

        if self.training_mode:
            cell = DropoutWrapper(cell=cell, output_keep_prob=0.5)
        else:
            cell = DropoutWrapper(cell=cell, output_keep_prob=1.0)

        return cell


