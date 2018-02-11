from quiver_engine import server
from keras.models import load_model

from argparse import ArgumentParser


def get_model(model_path):
    model = load_model(model_path)
    return model


def main(model_path, temp_folder_path, input_folder_path, classes, num_predictions):
    model = get_model(model_path)

    server.launch(
        model=model,  # a Keras Model
        classes=classes,
        top=num_predictions,  # number of top predictions to show in the gui (default 5)
        temp_folder=temp_folder_path,
        input_folder=input_folder_path,
        # the localhost port the dashboard is to be served on
        port=5000
    )


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--model_path', required=True)
    parser.add_argument('--temp_folder_path', required=True)
    parser.add_argument('--input_folder_path', required=True)
    parser.add_argument('--classes', required=True)
    parser.add_argument('--num_predictions', required=True, type=int)

    args = parser.parse_args()
    main(
        model_path=args.model_path,
        temp_folder_path=args.tmp_folder_path,
        input_folder_path=args.input_folder_path,
        classes=args.classes.split(','),
        num_predictions=args.num_predictions
    )
