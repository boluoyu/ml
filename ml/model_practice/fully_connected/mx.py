from __future__ import print_function

import mxnet as mx
import numpy as np

from mxnet import nd, autograd, gluon


def gen_dataset():
    MAX_DOC_LENGTH = 100

    X_train = nd.random_normal(shape=(1000, MAX_DOC_LENGTH))
    y_train = nd.random.rand
    X_test = nd.random_normal(shape=(100, MAX_DOC_LENGTH))
    y_test = 0
    return (X_train, y_train), (X_test, y_test)


def evaluate_accuracy(data_iterator, model, model_ctx):
    acc = mx.metric.Accuracy()
    for ix, (xi, label) in enumerate(data_iterator):
        xi = xi.as_in_context(model_ctx)
        label = label.as_in_context(model_ctx)
        output = model(xi)
        predictions = nd.argmax(output, axis=1)
        acc.update(preds=predictions, labels=label)

    return acc


def get_model(model_ctx, hidden_units=10):
    model = gluon.nn.Sequential()
    with model.name_scope():
        model.add(gluon.nn.Dense(hidden_units, activation='relu'))
        model.add(gluon.nn.Dense(1, activation='sigmoid'))

    model.collect_params().initialize(mx.init.Normal(sigma=0.1), ctx=model_ctx)
    binary_cross_entropy = gluon.loss.SigmoidBinaryCrossEntropyLoss()
    trainer = gluon.Trainer(params=model.collect_params(), optimizer='sgd', optimizer_params={'learning_rate': 0.01})
    return model, trainer, binary_cross_entropy


def train(model, model_ctx, train_data, test_data, trainer, loss_function, epochs, smoothing_constant=0.01):
    for epoch_num in range(1, epochs):
        cumulative_loss = 0
        num_training_datums = 0
        for ix, (xi, label) in enumerate(train_data):
            num_training_datums += 1
            xi = xi.as_in_context(model_ctx)
            label = label.as_in_context(model_ctx)
            with autograd.record():
                output = model(xi)
                loss = loss_function(output, label)

            loss.backward()
            trainer.step()
            cumulative_loss += nd.sum(loss).asscalar()

        test_accuracy = evaluate_accuracy(test_data, model, model_ctx)
        train_accuracy = evaluate_accuracy(train_data, model, model_ctx)
        print("Epoch %s. Loss: %s, Train_acc %s, Test_acc %s" %
              (epoch_num, cumulative_loss / num_training_datums, train_accuracy, test_accuracy))


def transform(xi, yi):
    return xi, yi


def main():
    model_ctx = mx.cpu()
    batch_size = 64

    (X_train, y_train), (X_test, y_test) = gen_dataset()
    train_data = mx.gluon.data.DataLoader(dataset=mx.gluon.data.ArrayDataset(X_train, y_train), batch_size=batch_size, shuffle=True)
    test_data = mx.gluon.data.DataLoader(dataset=mx.gluon.data.ArrayDataset(X_test, y_test), batch_size=batch_size, shuffle=True)

    model, trainer, loss_function = get_model(model_ctx)
    train(
        epochs=2,
        model=model,
        model_ctx=model_ctx,
        train_data=train_data,
        test_data=test_data,
        trainer=trainer,
        loss_function=loss_function
    )


if __name__ == '__main__':
    main()
