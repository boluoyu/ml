import json
import logging
import os

import joblib
import numpy as np
import tensorflow as tf
from tensorflow.contrib.rnn import GRUCell, DropoutWrapper, MultiRNNCell
from py_zipkin.zipkin import zipkin_span

from automation.common.utils import chunks
from automation.preparer.service.nlp_preparer import NlpTextPreprocessor

logger = logging.getLogger(__name__)

SERVICE_NAME = os.getenv('SERVICE_NAME', 'builder')

INVALID_MODEL_DICT = {'score': -1, 'th': 0, 'vp': 0, 'vr': 0, 'tp': 0, 'tr': 0}


class RnnTextClassifier(object):
    VALIDATION_BATCH_SIZE = 100

    def __init__(self, sess, model_dir, print_progress_every_n=1000, checkpoint_every_n=500, log_every_n=10, n_layers=3,
                 category='unspecified_category', n_classes=2, n_steps=30, n_input=5981, n_hidden=128,
                 batch_size=1, learning_rate=0.01, max_iter=-1, char_based=False, prediction=False, recall=0.30,
                 precision=0.80, test_path=None, val_path=None, checkpoint_name='unspecified_checkpoint_name', **kwargs):

        # n_steps=270, n_input=256 for char
        self.sess = sess
        self.category = category
        self.model_dir = model_dir
        self.print_progress_every_n = print_progress_every_n
        self.checkpoint_every_n = checkpoint_every_n
        self.log_every_n = log_every_n
        self.loaded_checkpoint = ''
        self.n_classes = n_classes
        self.n_layers = n_layers
        self.n_steps = n_steps
        self.n_input = n_input
        self.n_hidden = n_hidden
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.char_based = char_based
        self.prediction = prediction
        self.recall = recall  # minimum required recall
        self.precision = precision  # minimum required precision
        self.test_path = test_path
        self.val_path = val_path
        self.checkpoint_name = checkpoint_name
        self.preprocessor = NlpTextPreprocessor()

    @property
    def checkpoint_path_template(self):
        return os.path.join(self.model_dir, self.checkpoint_name)

    def initialize_saver(self):
        logger.debug('initializing saver')
        self.saver = tf.train.Saver(max_to_keep=20)

    def save(self, global_step):
        ckpt_path = self.saver.save(self.sess, self.checkpoint_path_template, global_step)
        logger.debug('model saved to {}'.format(ckpt_path))
        return ckpt_path

    def load_transformer(self, text_processor_file_path):
        logger.debug('Loading transformer from path %s', text_processor_file_path)
        import os
        logger.debug('Transormer exists %s', os.path.exists(text_processor_file_path))
        self.text_processor = joblib.load(text_processor_file_path)
        self.n_input = len(self.text_processor.vocabulary_)
        logger.debug('Embedding dimensions %s', self.n_input)

    def load_model(self, checkpoint_file_path):
        ''' if you pass an fname, loads from the fname; only need to specify the filename since
        the model_dir is already a member. E.g., checkpoint_fname='fire-v1_ckpt-25000000'
        if you pass an index (int), you can load e.g. the 0th save (usually the init)
        or whatever.  Negative numbers work as well e.g. "-1" for most recent
        '''
        logger.debug('Loading model from path %s', checkpoint_file_path)

        self.initialize_graph()
        self.initialize_saver()
        self.loaded_checkpoint = checkpoint_file_path
        self.saver.restore(self.sess, self.loaded_checkpoint)

    def initialize_graph(self):
        '''
        To classify texts using a recurrent neural network, here we consider every sentence as a sequence of words.
        '''
        # logger.debug('defining input and output variables')
        # define input/output variable
        self.x = tf.placeholder(tf.float32, [None, self.n_steps, self.n_input], name='x')
        self.y = tf.placeholder(tf.float32, [None, self.n_classes], name='y')

        # logger.debug('defining rnn network')
        with tf.variable_scope("rnn"):
            # Prepare data shape to match `rnn` function requirements
            # Current data input shape: (batch_size, n_steps, n_input)
            # Required shape: 'n_steps' tensors list of shape (batch_size, n_input)

            # Define a gru cell with tf.nn.rnn_cell, which is the updated package
            num_neurons = self.n_hidden
            num_layers = self.n_layers

            # to reuse the cell weights, the syntax needs to be changed in following way due to TF change after 27 machine recovers
            def gru_cell():
                cell = GRUCell(num_neurons, reuse=tf.get_variable_scope().reuse)  # Or LSTMCell(num_neurons)
                if self.prediction:
                    cell = DropoutWrapper(cell, output_keep_prob=1)
                else:
                    cell = DropoutWrapper(cell,
                                          output_keep_prob=0.5)  # changed to 0.5 to reduce overfitting for checkpoint_accident5
                return cell

            # self.cell = MultiRNNCell([cell] * num_layers) # this syntax needs to be changed in following way due to TF change after 27 machine recovers
            self.cell = MultiRNNCell([gru_cell() for _ in range(num_layers)])
            # Batch size x time steps x features.
            # data = tf.placeholder(tf.float32, [None, max_length, 28])
            self.outputs, self.states = tf.nn.dynamic_rnn(self.cell, self.x, dtype=tf.float32)
            self.outputs = tf.transpose(self.outputs, [1, 0, 2])
            self.last = tf.gather(self.outputs, int(self.outputs.get_shape()[0]) - 1)

            # define last layer weights after rnn cell: n_hidden * n_classes
            # self.weights_out = tf.Variable(tf.random_normal([self.n_hidden, self.n_classes]))
            self.weights_out = tf.Variable(tf.truncated_normal([self.n_hidden, self.n_classes], stddev=0.1))
            # self.biases_out = tf.Variable(tf.random_normal([self.n_classes]))
            self.biases_out = tf.Variable(tf.constant(0.1, shape=[self.n_classes]))
            # Linear activation, using rnn inner loop last output
            # self.y_hat = tf.matmul(self.last, self.weights_out) + self.biases_out
            self.y_hat = tf.nn.softmax(tf.matmul(self.last, self.weights_out) + self.biases_out)

        # Define loss and optimizer
        # logger.debug('logits shape is', self.y_hat.get_shape)
        # self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.y, logits=self.y_hat))
        self.cost = -tf.reduce_mean(self.y * tf.log(self.y_hat))
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.cost)

        # Evaluate model
        self.correct_pred = tf.equal(tf.argmax(self.y_hat, 1), tf.argmax(self.y, 1))
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_pred, tf.float32))

        # Initializing the variables
        self.sess.run(tf.global_variables_initializer())

    def vectorize(self, X_train_texts, y_train):
        X_train = np.array(list(self.text_processor.transform(X_train_texts)))
        X_train = (np.arange(self.n_input) == X_train[:, :, None]).astype(
            float)  # char id is from 0 to 240; 0 is pad for empty char less than fixed max length; i.e., n_steps
        y_train = np.array(y_train, dtype=int)
        y_train = (np.arange(self.n_classes) == y_train[:, None]).astype(
            float)  # the training data's char id is from 0 to 7; 0 is class id, not padding
        return X_train, y_train

    @zipkin_span(service_name=SERVICE_NAME, span_name='rnn_classifier_fit')
    def fit(self, data_generator, text_processor):
        """

        :param data_generator: iterator producing training generator
        :param text_processor: encodes raw text into numeric format for model
        :return:
        """
        self.initialize_graph()
        self.initialize_saver()

        self.text_processor = text_processor
        from itertools import count

        models = []
        for step in count():
            # Keep training forever or until reach max iterations
            if self.max_iter == -1 or step * self.batch_size < self.max_iter:
                batch_texts, batch_labels = next(data_generator)
                batch_X, batch_y = self.vectorize(batch_texts, batch_labels)
                feed_dict = {self.x: batch_X, self.y: batch_y}
                # Run optimization op (backprop)
                self.sess.run(self.optimizer, feed_dict=feed_dict)
                if step % self.checkpoint_every_n == 0:
                    best_model = None
                    if models:
                        best_model = sorted(models, key=lambda x: x['eval']['score'], reverse=True)[0]

                    eval_result, _ = self.evaluate()
                    if best_model is None or (eval_result['score'] > best_model['eval']['score']):
                        ckpt_path = self.save(step)
                        models.append({'model': ckpt_path, 'eval': eval_result})

                    best_model = sorted(models, key=lambda x: x['eval']['score'], reverse=True)[0]

                    if best_model['eval']['score'] > 0:
                        logger.debug('uptodate best model %s:', best_model)
            else:
                print("Optimization Finished!")
                if len(models) > 0:
                    best_model = sorted(models, key=lambda x: x['eval']['score'], reverse=True)[0]
                    if best_model['eval']['score'] > 0:
                        logger.debug('uptodate best model %s:', best_model)

                # Todo (josh) handle if there is no "best model"
                best_model_path = sorted(models, key=lambda x: x['eval']['score'], reverse=True)[0]
                return best_model_path['model']

    def log_batch_accuracy(self, feed_dict, batch_texts):
        if logger.isEnabledFor(logging.DEBUG):
            pred_y = self.sess.run(self.y_hat, feed_dict=feed_dict)
            match_y = self.sess.run(self.correct_pred, feed_dict=feed_dict)
            logger.debug(batch_texts)
            # logger.debug(np.argmax(batch_X[0][:20], axis=1), np.argmax(batch_X[1][:20], axis=1))
            logger.debug('debugging info: y {}, pred_y {}, match_y {}'.format(feed_dict[self.y], pred_y, match_y))

    def evaluate(self):
        from sklearn.metrics import precision_recall_curve as prc
        val_t_l = []
        for l in open(self.val_path):
            j = json.loads(l)
            val_t_l.append({'t': j['text'], 'y': int(j['curation'] == 'yes')})
        yhat = self.predict_proba([e['t'] for e in val_t_l])[:, 1]
        y = np.array([e['y'] for e in val_t_l])
        p, r, th = prc(y, yhat)  # r will be in descending order
        chosen_th = None
        valid_comb = list(filter(lambda x: x[0] >= self.precision and x[1] >= self.recall, zip(p, r, th)))
        if len(valid_comb) == 0:
            return INVALID_MODEL_DICT, (y, yhat)

        # apply detection and chosen threshold to test data
        test_t_l = []
        for l in open(self.test_path):
            j = json.loads(l)
            test_t_l.append({'t': j['text'], 'y': int(j['curation'] == 'yes')})
        test_yhat = self.predict_proba([e['t'] for e in test_t_l])[:, 1]
        test_y = np.array([e['y'] for e in test_t_l])
        evals = []
        for vp, vr, vth in valid_comb:
            test_p, test_r = self.calc_pr(test_yhat, test_y, vth)
            if test_p >= self.precision and test_r >= self.recall:
                evals.append({'score': vp * test_p, 'th': vth, 'vp': vp, 'vr': vr, 'tp': test_p, 'tr': test_r})
        if len(evals) > 0:
            return sorted(evals, key=lambda x: x['score'], reverse=True)[0], (y, yhat)
        return INVALID_MODEL_DICT, (y, yhat)

    def vectorize_texts(self, texts):
        logger.debug('Vectoring texts n_input %s', self.n_input)
        txt_matrix = np.array(list(self.text_processor.transform(np.array(self.preprocessor.process_data(texts)))))
        text_matrix = (np.arange(self.n_input) == txt_matrix[:, :, None]).astype(float)
        return text_matrix

    def predict_proba(self, texts):
        predictions = []

        for text_chunk in chunks(texts, chunk_size=self.VALIDATION_BATCH_SIZE):
            text_matrix = self.vectorize_texts(texts=text_chunk)
            predictions.extend(self.sess.run(self.y_hat, feed_dict={self.x: text_matrix}))

        predictions = np.array(predictions, dtype=np.float32)
        logger.debug('Predictions shape %s', predictions.shape)
        return predictions

    def calc_pr(self, yhat, y, th):
        tps = list(filter(lambda x: x[1] == 1, zip(yhat, y)))
        r = len(list(filter(lambda x: x[0] >= th, tps))) / float(len(tps))
        detections = list(filter(lambda x: x[0] >= th, zip(yhat, y)))

        if len(detections) == 0:
            p = 0
        else:
            p = len(list(filter(lambda x: x[1] == 1, detections))) / float(len(detections))

        return p, r
