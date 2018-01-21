from argparse import ArgumentParser

import json


def load_file(data_file_path):
    with open(data_file_path) as f:
        for line in f:
            yield line


def format_training_datum(_id, url):
    target = _id.split('_')[0]

    return dict(
        y=target,
        X=url
    )


def main(data_file_path, out_file_path):
    data = load_file(data_file_path)

    with open(out_file_path, 'w') as f:
        for datum in data:
            _id, url = datum.split(' ')
            training_datum = format_training_datum(_id, url)
            f.write(json.dumps(training_datum))


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--data_file_path', required=True)
    parser.add_argument('--out_file_path', required=True)

    args = parser.parse_args()
    main(args.data_file_path, args.out_file_path)
