import json


class DataGenerator:
    def __init__(self, training_data_file_path, validation_data_file_path, image_loader):
        self.training_data_file_path = training_data_file_path
        self.validation_data_file_path = validation_data_file_path
        self.image_loader = image_loader

        self.training_data = self.load_data(self.training_data_file_path)

    def get_validation_data(self):
        X = []
        y = []

        validation_data = self.load_data(self.validation_data_file_path)
        for datum in validation_data:
            X.append(self.image_loader.load(datum['X']))
            y.append(datum['y'])

        return (X, y)

    def get_training_data(self, batch_size, num_batches):
        while True:
            training_data = self.load_data(self.training_data_file_path)

            for batch_num in range(num_batches):
                X = []
                y = []

                for i in range(batch_size):
                    training_datum = next(training_data)
                    X.append(self.image_loader.load(training_datum['X']))
                    y.append(training_datum['y'])

                yield (X, y)

    def load_data(self, data_path):
        with open(data_path) as f:
            data = map(json.loads, f)
            for datum in data:
                yield datum
