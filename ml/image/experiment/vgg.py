from keras.applications.vgg16 import VGG16


class VGGExperiment:
    def get_model(self):
        model = VGG16()
        print(model.summary())
        return model
