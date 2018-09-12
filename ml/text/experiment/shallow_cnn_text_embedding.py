from keras.layers import Input, Embedding, Conv1D, MaxPooling1D, Dense, Flatten, Dropout
from keras.models import Model


def get_model(max_document_length, max_num_tokens, embedding_weights, embedding_dims, num_targets):
    document = Input(shape=(max_document_length,))
    x = Embedding(input_dim=max_num_tokens, output_dim=embedding_dims, input_length=max_document_length,
                  trainable=False, weights=[embedding_weights])(document)
    x = Conv1D(filters=128, kernel_size=5, activation="relu")(x)
    x = Dropout(rate=0.5)(x)
    x = MaxPooling1D(pool_size=1)(x)
    x = Flatten()(x)
    x = Dense(units=128, activation="relu")(x)
    x = Dropout(rate=0.5)(x)
    y = Dense(units=num_targets, activation="softmax")(x)

    model = Model(inputs=document, outputs=y)
    print(model.summary())
    return model
