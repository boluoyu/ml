import json
import logging
import os
import pandas as pd
import time

logging.basicConfig(level=os.getenv('LOG_LEVEL', logging.DEBUG))
logger = logging.getLogger(__name__)

TRAINING_STATISTICS_FIGURE_NAME = '{experiment}_{model}_{epoch_num}_{time}.png'
TRAINING_STATISTICS_FILE_NAME = '{experiment}_{model}_{epoch_num}_{time}.json'
EVALUATION_DATA_FILE_NAME = '{experiment}_{model}_{epoch_num}_{time}_evaluation.json'
CHECKPOINT_FILE_TEMPLATE = '{experiment}_{model}_{epoch_num}_{step_num}_chkpt.hdf5'
VALIDATION_MESSAGE_TEMPLATE = 'Experiment: {experiment} Model: {model} Epoch: {epoch_num} Step: {step_num} Training Loss: {t_loss} Val Loss: {val_loss}'


class TrainingRunner:
    def __init__(self, class_map, experiment_name, model_dir, classifier, data_generator, evaluator, batch_size,
                 checkpoint_step_num, validation_step_num, num_epochs, num_steps, verbose=True):

        self.experiment_name = experiment_name
        self.model_dir = model_dir
        self.classifier = classifier
        self.model_name = classifier.name
        self.data_generator = data_generator
        self.evaluator = evaluator

        self.class_map = class_map
        self.batch_size = batch_size
        self.checkpoint_step_num = checkpoint_step_num
        self.validation_step_num = validation_step_num
        self.num_epochs = num_epochs
        self.num_steps = num_steps
        self.verbose = verbose

        self.training_statistics = []

    def run(self):
        epoch_num = 0

        for epoch_num in range(1, self.num_epochs + 1):
            step_num = 0

            training_data_mini_batches = self.data_generator.get_training_data(self.batch_size, self.num_steps)

            while step_num < self.num_steps:
                step_num += 1

                try:
                    X, y = next(training_data_mini_batches)
                except StopIteration:
                    break

                training_loss = self.classifier.fit(X, y)
                print('training_loss', training_loss)

                if self.should_compute_validation_loss(step_num):
                    self.training_statistics.append(
                        self.compute_validation_statistics(epoch_num, step_num, training_loss))

                if self.should_save_checkpoint(step_num):
                    self.save_checkpoint(epoch_num, step_num)

            self.save_checkpoint(epoch_num, step_num)
        fig = self.get_training_stats_plot(self.training_statistics)
        self.save_training_stats_figure(fig, epoch_num)
        self.save_training_statistics(self.num_epochs)

        val_X, val_y = self.data_generator.get_validation_data()
        y_pred = self.classifier.predict(X=val_X)
        evaluation_result = self.evaluator.evaluate(class_map=self.class_map, y_true=val_y, y_pred=y_pred)
        self.save_evaluation_data(evaluation_result, epoch_num)

    def should_compute_validation_loss(self, step_num):
        return step_num % self.validation_step_num == 0

    def should_save_checkpoint(self, step_num):
        return step_num % self.checkpoint_step_num == 0

    def compute_validation_statistics(self, epoch, step_num, training_loss):
        val_X, val_y = self.data_generator.get_validation_data()

        validation_loss = self.classifier.compute_loss(val_X, val_y)
        logger.info(VALIDATION_MESSAGE_TEMPLATE.format(
            experiment=self.experiment_name,
            epoch_num=epoch,
            model=self.model_name,
            step_num=step_num,
            val_loss=validation_loss,
            t_loss=training_loss)
        )

        return dict(
            training_loss=training_loss,
            validation_loss=validation_loss,
            epoch=epoch,
            step_num=step_num
        )

    def save_checkpoint(self, epoch_num, step_num):
        checkpoint_file_name = self.get_checkpoint_file_name(epoch_num, step_num)
        self.classifier.save(checkpoint_file_name)

    def save_training_statistics(self, epoch_num):
        training_statistics_file_name = self.get_training_statistics_file_name(epoch_num)

        logger.info('Writing training statistics to file %s', training_statistics_file_name)
        with open(training_statistics_file_name, 'w') as f:
            f.write(json.dumps(self.training_statistics))

    def save_evaluation_data(self, evaluation_data, epoch_num):
        evaluation_data_file_name = self.get_evaluation_data_file_name(epoch_num)

        with open(evaluation_data_file_name, 'w') as f:
            f.write(json.dumps(evaluation_data_file_name))

    def get_checkpoint_file_name(self, epoch_num, step_num):
        checkpoint_file_name = CHECKPOINT_FILE_TEMPLATE.format(
            experiment=self.experiment_name,
            model=self.model_name,
            epoch_num=epoch_num,
            step_num=step_num
        )

        return os.path.join(self.model_dir, checkpoint_file_name)

    def get_evaluation_data_file_name(self, epoch_num):
        epoch_ts = time.time()

        evaluation_data_file_name = EVALUATION_DATA_FILE_NAME.format(
            experiment=self.experiment_name,
            model=self.model_name,
            epoch_num=epoch_num,
            time=epoch_ts
        )

        return os.path.join(self.model_dir, evaluation_data_file_name)

    def get_training_statistics_figure_name(self, epoch_num):
        epoch_ts = time.time()

        training_statistics_figure_name = TRAINING_STATISTICS_FIGURE_NAME.format(
            experiment=self.experiment_name,
            model=self.model_name,
            epoch_num=epoch_num,
            time=epoch_ts
        )

        return os.path.join(self.model_dir, training_statistics_figure_name)

    def get_training_statistics_file_name(self, epoch_num):
        epoch_ts = time.time()

        training_statistics_file_name = TRAINING_STATISTICS_FILE_NAME.format(
            experiment=self.experiment_name,
            model=self.model_name,
            epoch_num=epoch_num,
            time=epoch_ts
        )

        return os.path.join(self.model_dir, training_statistics_file_name)

    def get_training_stats_plot(self, training_statistics):
        epoch_training_loss = [(stats['step_num'], stats['training_loss'], stats['validation_loss']) for stats in
                               training_statistics]
        cf = pd.DataFrame(epoch_training_loss, columns=['step', 'training_loss', 'cv_loss'])
        fig = cf.plot(x='step', y=['training_loss', 'cv_loss']).get_figure()
        return fig

    def save_training_stats_figure(self, fig, epoch):
        figure_file_name = self.get_training_statistics_figure_name(epoch_num=epoch)
        fig.savefig(figure_file_name)
