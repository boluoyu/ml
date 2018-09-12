from argparse import ArgumentParser

from keras.applications.vgg16 import decode_predictions, preprocess_input
from keras.preprocessing.image import load_img, img_to_array
from keras.utils import plot_model

from ml.image.experiment.vgg import VGGExperiment


def get_image():
    # load an image from file
    image = load_img('/Users/jnewman/Projects/learning/ai_blog/ml/data/misc/images/mug.jpg', target_size=(224, 224))
    # reshape data for the model
    image = img_to_array(image)
    image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
    image = preprocess_input(image)
    return image


def main():
    experiment = VGGExperiment()
    model = experiment.get_model()
    plot_model(model, to_file='vgg.png')
    image = get_image()
    yhat = model.predict(image)
    label = decode_predictions(yhat)
    # retrieve the most likely result, e.g. highest probability
    label = label[0][0]
    # print the classification
    print('%s (%.2f%%)' % (label[1], label[2] * 100))


if __name__ == "__main__":
    parser = ArgumentParser()

    args = parser.parse_args()
    main()
