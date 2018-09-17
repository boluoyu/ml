from keras import Sequential, Model, Input
from keras.activations import relu
from keras.layers import BatchNormalization, Conv1D, Bidirectional, SimpleRNN, TimeDistributed, Dense, Lambda, K


def clipped_relu(x):
    return relu(x, max_value=20)


def ctc_lambda_func(args):
    labels, y_pred, input_length, label_length = args
    return K.ctc_batch_cost(labels, y_pred, input_length, label_length)


def ctc(y_true, y_pred):
    return y_pred


class DeepSpeachExperiment:
    def get_model(self):
        model = Sequential()

        # Batch normalize the input
        model.add(BatchNormalization(axis=-1, input_shape=(None, 161), name='BN_1'))

        # 1D Convs
        model.add(Conv1D(512, 5, strides=1, activation=clipped_relu, name='Conv1D_1'))
        model.add(Conv1D(512, 5, strides=1, activation=clipped_relu, name='Conv1D_2'))
        model.add(Conv1D(512, 5, strides=2, activation=clipped_relu, name='Conv1D_3'))

        # Batch Normalization
        model.add(BatchNormalization(axis=-1, name='BN_2'))

        # BiRNNs
        model.add(Bidirectional(SimpleRNN(1280, return_sequences=True, name='BiRNN_1'), merge_mode='sum'))
        model.add(Bidirectional(SimpleRNN(1280, return_sequences=True, name='BiRNN_2'), merge_mode='sum'))
        model.add(Bidirectional(SimpleRNN(1280, return_sequences=True, name='BiRNN_3'), merge_mode='sum'))
        model.add(Bidirectional(SimpleRNN(1280, return_sequences=True, name='BiRNN_4'), merge_mode='sum'))
        model.add(Bidirectional(SimpleRNN(1280, return_sequences=True, name='BiRNN_5'), merge_mode='sum'))
        model.add(Bidirectional(SimpleRNN(1280, return_sequences=True, name='BiRNN_6'), merge_mode='sum'))
        model.add(Bidirectional(SimpleRNN(1280, return_sequences=True, name='BiRNN_7'), merge_mode='sum'))

        # Batch Normalization
        model.add(BatchNormalization(axis=-1, name='BN_3'))

        # FC
        model.add(TimeDistributed(Dense(1024, activation=clipped_relu, name='FC1')))
        model.add(TimeDistributed(Dense(29, activation='softmax', name='y_pred')))

        y_pred = model.outputs[0]
        model_input = model.inputs[0]

        model.summary()

        labels = Input(name='the_labels', shape=[None, ], dtype='int32')
        input_length = Input(name='input_length', shape=[1], dtype='int32')
        label_length = Input(name='label_length', shape=[1], dtype='int32')

        loss_out = Lambda(ctc_lambda_func, name='ctc')([labels, y_pred, input_length, label_length])
        model = Model(inputs=[model_input, labels, input_length, label_length], outputs=loss_out)
        print(model.summary)
        return model
