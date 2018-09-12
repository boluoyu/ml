from keras import Input
from keras.applications import VGG16


class VGGImageCaptioningExperiment:
    def get_feature_extraction_model(self):
        in_layer = Input(shape=(224, 224, 3))
        model = VGG16(include_top=False, input_tensor=in_layer, pooling="avg")
        print(model.summary())
        return model

    def get_model(self, max_document_length, embedding_size, vocabulary_size):
        # feature extractor
        input_1 = Input(shape=(4096,))
        x1 = Dropout(0.5)(input_1)
        x1 = Dense(256, activation="relu")(x1)

        # sequence model
        input_2 = Input(shape=(max_document_length,))
        x2 = Embedding(vocabulary_size, embedding_size, mask_zero=True)(input_2)
        x2 = Dropout(0.5)(x2)
        x2 = LSTM(256)(x2)

        # decoder model
        x = add([x1, x2])
        x = Dense(256, activation="relu")(x)
        y = Dense(vocabulary_size)(x)

        model = Model(inputs=[input_1, input_2], outputs=y)
        model.compile(loss="categorical_crossentropy", optimizer=Adam())
        print(model.summary())

        return model
