from argparse import ArgumentParser

from ml.image.classifier.registry import CLASSIFIER_REGISTRY
from ml.image.experiment.registry import EXPERIMENT_REGISTRY
from ml.image.evaluator.registry import EVALUATOR_REGISTRY
from ml.image.transform.registry import TRANSFORMER_REGISTRY
from ml.image.helper.data import DataGenerator
from ml.image.service.load import ImageLoader


def main(experiment_name, classifier_name, transformer_name, evaluator_name, model_dir, training_data_file_path,
         validation_file_path, batch_size, num_epochs, checkpoint_step_num, validation_step_num, num_steps, verbose):

    classifier_cls = CLASSIFIER_REGISTRY[classifier_name]
    evaluator_cls = EVALUATOR_REGISTRY[evaluator_name]
    experiment_cls = EXPERIMENT_REGISTRY[experiment_name]
    transformer_cls = TRANSFORMER_REGISTRY[transformer_name]

    data_generator = DataGenerator(
        training_data_file_path=training_data_file_path,
        validation_data_file_path=validation_file_path,
        image_loader=ImageLoader()
    )

    experiment = experiment_cls(
        classifier_cls=classifier_cls,
        transformer_cls=transformer_cls,
        evaluator_cls=evaluator_cls,
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
    parser.add_argument('--classifier_name', required=True)
    parser.add_argument('--transformer_name', required=True)
    parser.add_argument('--evaluator_name', required=True)
    parser.add_argument('--model_dir', required=True)
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
        experiment_name=args.experiment_name,
        classifier_name=args.classifier_name,
        evaluator_name=args.evaluator_name,
        transformer_name=args.transformer_name,
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
