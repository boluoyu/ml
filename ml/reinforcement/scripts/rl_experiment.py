import gym
import universe
from argparse import ArgumentParser

from ml.reinforcement.experiment.registry import EXPERIMENT_REGISTRY


def main(experiment_name, model_dir, batch_size, num_epochs, checkpoint_step_num, num_steps, verbose):
    experiment_cls = EXPERIMENT_REGISTRY[experiment_name]

    experiment = experiment_cls(
        model_dir=model_dir,
        batch_size=batch_size,
        num_epochs=num_epochs,
        checkpoint_step_num=checkpoint_step_num,
        num_steps=num_steps,
        verbose=verbose,
    )

    experiment.run_experiment()


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--experiment_name', required=True)
    parser.add_argument('--model_dir', required=True)
    parser.add_argument('--batch_size', required=True, type=int)
    parser.add_argument('--num_epochs', required=True, type=int)
    parser.add_argument('--checkpoint_step_num', required=True, type=int)
    parser.add_argument('--num_steps', required=True, type=int)
    parser.add_argument('--verbose', required=False, default=True, type=bool)

    args = parser.parse_args()

    main(
        experiment_name=args.experiment_name,
        model_dir=args.model_dir,
        num_epochs=args.num_epochs,
        checkpoint_step_num=args.checkpoint_step_num,
        num_steps=args.num_steps,
        verbose=args.verbose,
        batch_size=args.batch_size
    )
