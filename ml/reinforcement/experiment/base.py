from ml.common.experiment.base import Experiment

from ml.reinforcement.runner.environ.base import EnvironmentRunner


class QlearningReinforcementLearningExperiment(Experiment):
    def __init__(self, agent, classifier_cls, transformer_cls, model_dir, batch_size, num_epochs, checkpoint_step_num,
                 validation_step_num, num_steps, evaluator_cls=None, data_generator=None, class_map=None,
                 training_mode=True, verbose=True):

        self.agent = agent

        super(QlearningReinforcementLearningExperiment, self).__init__(
            classifier_cls=classifier_cls,
            transformer_cls=transformer_cls,
            model_dir=model_dir,
            batch_size=batch_size,
            num_epochs=num_epochs,
            checkpoint_step_num=checkpoint_step_num,
            validation_step_num=validation_step_num,
            class_map=class_map,
            data_generator=data_generator,
            evaluator_cls=evaluator_cls,
            num_steps=num_steps,
            training_mode=training_mode,
            verbose=verbose
        )

    def get_training_runner(self):
        return EnvironmentRunner(
            agent=self.agent,
            num_epochs=self.num_epochs,
            num_steps=self.num_steps
        )
