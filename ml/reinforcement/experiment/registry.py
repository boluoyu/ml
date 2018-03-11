from ml.reinforcement.experiment.cartpole.actor_critic import ActorCriticKerasExperiment
from ml.reinforcement.experiment.cartpole.qlearn import CartpoleQlearnKerasExperiment
from ml.reinforcement.experiment.neon_racer.qlearn_autoencoder import NeonRacerQlearnAutoencoderKerasExperiment

EXPERIMENT_REGISTRY = {
    ActorCriticKerasExperiment.name:    ActorCriticKerasExperiment,
    CartpoleQlearnKerasExperiment.name: CartpoleQlearnKerasExperiment,
    NeonRacerQlearnAutoencoderKerasExperiment.name: NeonRacerQlearnAutoencoderKerasExperiment
}
