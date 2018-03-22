import logging
import os
import csv
import json

from argparse import ArgumentParser

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


def load_csv_data(in_file):
    with open(in_file) as f:
        reader = csv.DictReader(f)
        for row in reader:
            yield row


def main(in_file, out_file):
    with open(out_file, 'w') as f:
        for datum in load_csv_data(in_file):
            f.write(json.dumps(datum) + '\n')


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--in_file', required=True)
    parser.add_argument('--out_file', required=True)

    args = parser.parse_args()
    main(args.in_file, args.out_file)
