from argparse import ArgumentParser

from ml.common.helper.data_generator import DataGenerator
from ml.image.experiment.registry import EXPERIMENT_REGISTRY
from ml.image.service.load import ImageLoader


def main(experiment_name, class_map_file_path, model_dir, training_data_file_path,
         validation_file_path, batch_size, num_epochs, checkpoint_step_num, validation_step_num, num_steps, verbose):
    experiment_cls = EXPERIMENT_REGISTRY[experiment_name]

    data_generator = DataGenerator(
        training_data_file_path=training_data_file_path,
        validation_data_file_path=validation_file_path,
        image_loader=ImageLoader()
    )

    experiment = experiment_cls(
        class_map_file_path=class_map_file_path,
        data_generator=data_generator,
        model_dir=model_dir,
        batch_size=batch_size,
        num_epochs=num_epochs,
        checkpoint_step_num=checkpoint_step_num,
        validation_step_num=validation_step_num,
        num_steps=num_steps,
        verbose=verbose
    )

    experiment.run_experiment()


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--experiment_name', required=True)
    parser.add_argument('--model_dir', required=True)
    parser.add_argument('--class_map_file_path', required=True)
    parser.add_argument('--training_data_file_path', required=True)
    parser.add_argument('--validation_data_file_path', required=True)
    parser.add_argument('--batch_size', required=True, type=int)
    parser.add_argument('--num_epochs', required=True, type=int)
    parser.add_argument('--checkpoint_step_num', required=True, type=int)
    parser.add_argument('--validation_step_num', required=True, type=int)
    parser.add_argument('--num_steps', required=True, type=int)
    parser.add_argument('--verbose', required=False, default=True, type=bool)

    args = parser.parse_args()

    main(
        class_map_file_path=args.class_map_file_path,
        experiment_name=args.experiment_name,
        model_dir=args.model_dir,
        training_data_file_path=args.training_data_file_path,
        validation_file_path=args.validation_data_file_path,
        num_epochs=args.num_epochs,
        checkpoint_step_num=args.checkpoint_step_num,
        validation_step_num=args.validation_step_num,
        num_steps=args.num_steps,
        verbose=args.verbose,
        batch_size=args.batch_size
    )
