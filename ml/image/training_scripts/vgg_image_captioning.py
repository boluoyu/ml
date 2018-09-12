import re
import string
from argparse import ArgumentParser

import os
from keras.applications.vgg16 import decode_predictions, preprocess_input
from keras.preprocessing.image import load_img, img_to_array
from keras.utils import plot_model
from pickle import dump

from ml.image.experiment.vgg_img_captioning import VGGImageCaptioningExperiment


def get_training_set(training_dir):
    images = dict()
    for name in os.listdir(training_dir):
        # load an image from file
        filename = os.path.join(training_dir, name)
        image = load_img(filename, target_size=(224, 224))
        # convert the image pixels to a numpy array
        image = img_to_array(image)
        # reshape data for the model
        image = image.reshape(
            (1, image.shape[0], image.shape[1], image.shape[2]))  # prepare the image for the VGG model
        image = preprocess_input(image)
        # get image id
        image_id = name.split('.')[0]
        images[image_id] = image

    return images


def preprocess_caption(caption):
    # prepare regex for char filtering
    re_punc = re.compile('[%s]' % re.escape(string.punctuation))
    # tokenize
    caption = caption.split()
    # convert to lower case
    caption = [word.lower() for word in caption.split()]
    # remove punctuation from each word
    caption = [re_punc.sub('', w) for w in caption]
    caption = [word for word in caption if len(word) > 1]
    return caption


def get_captions(caption_file_path):
    caption_mapping = dict()

    with open(caption_file_path) as f:
        data = f.read()

    for line in data.split("\n"):
        # split line by white space
        tokens = line.split()

        if len(tokens) < 2:
            continue

        # take the first token as the image id, the rest as the description
        image_id, image_desc = tokens[0], tokens[1:]
        image_id = image_id.split('.')[0]
        # convert description tokens back to string image_desc = ' '.join(image_desc)
        # store the first description for each image if image_id not in mapping:
        caption_mapping[image_id] = preprocess_caption(image_desc)

    return caption_mapping


def extract_features(model, training_set):
    features = dict()
    for image_id, image in training_set.items():
        features = model.predict(image)
        features[image_id] = features

    return features


def main(training_dir, caption_file_path, model_name, model_file_path):
    experiment = VGGImageCaptioningExperiment()
    model = experiment.get_feature_extraction_model()
    training_set = get_training_set(training_dir)
    captions = get_captions(caption_file_path)
    print('Loaded Images: %d' % len(training_set))
    extracted_features = extract_features(model=model, training_set=training_set)
    dump(extracted_features, open('features.pkl', 'wb'))


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--training_data_dir')
    parser.add_argument('--caption_file_path')
    parser.add_argument('--model_name')
    parser.add_argument('--model_file_path')

    args = parser.parse_args()
    main(args.training_data_dir, args.caption_file_path, args.model_name, args.model_file_path)
