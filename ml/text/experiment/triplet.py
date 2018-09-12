from keras import Input, Model, backend as K
from keras.layers import Lambda, Concatenate


def triplet_model(encoder, input_shape):
    x_anchor = Input(shape=input_shape, name='anchor')
    x_related = Input(shape=input_shape, name='related')
    x_unrelated = Input(shape=input_shape, name='unrelated')

    h_anchor = encoder(x_anchor)
    h_related = encoder(x_related)
    h_unrelated = encoder(x_unrelated)

    related_dist = Lambda(euclidean_distance, name='pos_dist')([h_anchor, h_related])
    unrelated_dist = Lambda(euclidean_distance, name='neg_dist')([h_anchor, h_unrelated])

    inputs = [x_anchor, x_related, x_unrelated]
    distances = Concatenate()([related_dist, unrelated_dist])

    model = Model(inputs=inputs, outputs=distances)
    return model


def triplet_loss(_, y_pred):
    margin = K.constant(1)
    return K.mean(K.maximum(K.constant(0), K.square(y_pred[:, 0]) - K.square(y_pred[:, 1]) + margin))


def triplet_accuracy(_, y_pred):
    return K.mean(y_pred[:, 0] < y_pred[:, 1])


def euclidean_distance(vectors):
    x, y = vectors
    return K.sqrt(K.maximum(K.sum(K.square(x - y), axis=1, keepdims=True), K.epsilon()))