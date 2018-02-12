from ml.reinforcement.experiment.cartpole.actor_critic import ActorCriticKerasExperiment
from ml.reinforcement.experiment.cartpole.qlearn import CartpoleQlearnKerasExperiment

EXPERIMENT_REGISTRY = {
    ActorCriticKerasExperiment.name:    ActorCriticKerasExperiment,
    CartpoleQlearnKerasExperiment.name: CartpoleQlearnKerasExperiment
}
