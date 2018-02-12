import tensorflow as tf

from keras import Model, backend
from keras.layers import Add, Dense, Input, Concatenate, Activation
from keras.optimizers import Adam

from ml.common.classifier.keras import KerasClassifier
from ml.reinforcement.agent.actor_critic import ActorCriticAgent
from ml.reinforcement.experiment.base import ReinforcementLearningExperiment
from ml.common.transformer.base import Transformer
from ml.reinforcement.transformer.multi_input import MultiInputTransformer


class ActorCriticKerasExperiment(ReinforcementLearningExperiment):
    name = 'cartpole_actor_critic_experiment'

    env_name = 'CartPole-v0'
    env_history_length = 100000
    gamma = 0.95  # discount rate
    epsilon = 1.0  # exploration rate
    epsilon_min = 0.01
    epsilon_decay = 0.995
    learning_rate = 0.001
    tau = 1.0
    observation_shape = (4,)  # 4 features
    actions = [0, 1]  # left, right
    action_shape = (2,)
    action_size = len(actions)
    loss = 'mse'
    optimizer = Adam(lr=learning_rate)
    verbose = True

    def _get_actor_model(self):
        observation_input = Input(shape=self.observation_shape)
        x = Dense(24, activation='relu')(observation_input)
        x = Dense(48, activation='relu')(x)
        x = Dense(24, activation='relu')(x)
        x = Dense(self.action_size)(x)
        model = Model(inputs=[observation_input], outputs=x)
        model.compile(optimizer=self.optimizer, loss=self.loss)
        print(model.summary())
        return model, observation_input

    def _get_critic_model(self):
        # evaluates based on action and observation
        action_input = Input(shape=self.action_shape)
        action_h1 = Dense(48, activation='linear')(action_input)

        observation_input = Input(shape=self.observation_shape)
        observation_h1 = Dense(24, activation='relu')(observation_input)
        observation_h2 = Dense(48, activation='linear')(observation_h1)

        merged = Add()([action_h1, observation_h2])
        merged_h1 = Dense(24, activation='relu')(merged)
        x = Dense(24, activation='relu')(merged_h1)
        x = Dense(1, activation='linear')(x)  # predicts reward
        model = Model(inputs=[action_input, observation_input], outputs=x)
        model.compile(optimizer=self.optimizer, loss=self.loss)
        print(model.summary())
        return model, action_input, observation_input

    def get_agent(self):
        self.session = tf.Session()
        backend.set_session(self.session)
        # ===================================================================== #
        #                               Actor Model                             #
        # Chain rule: find the gradient of changing the actor network params in #
        # getting closest to the final value network predictions, i.e. de/dA    #
        # Calculate de/dA as = de/dC * dC/dA,                                   #
        # e=error, C=critic, A=actor #                                         #
        # ===================================================================== #
        actor_model, actor_observation_input = self._get_actor_model()
        actor_classifier = KerasClassifier(
            model=actor_model,
            transformer=Transformer()
        )

        target_actor_model, _ = self._get_actor_model()
        target_actor_classifier = KerasClassifier(
            model=actor_model,
            transformer=Transformer()
        )

        critic_model, critic_action_input, critic_observation_input = self._get_critic_model()
        critic_classifier = KerasClassifier(
            model=critic_model,
            transformer=MultiInputTransformer()
        )

        target_critic_model, _, _ = self._get_critic_model()
        target_critic_classifier = KerasClassifier(
            model=target_critic_model,
            transformer=MultiInputTransformer()
        )

        return ActorCriticAgent(
            session=self.session,
            actions=self.actions,
            action_shape=self.action_shape,
            actor_classifier=actor_classifier,
            target_actor_classifier=target_actor_classifier,
            actor_observation_input=actor_observation_input,
            critic_classifier=critic_classifier,
            target_critic_classifier=target_critic_classifier,
            critic_action_input=critic_action_input,
            critic_observation_input=critic_observation_input,
            observation_shape=self.observation_shape,
            epsilon=self.epsilon,
            tau=self.tau,
            epsilon_decay=self.epsilon_decay,
            epsilon_min=self.epsilon_min,
            gamma=self.gamma,
            verbose=self.verbose,
            learning_rate=self.learning_rate
        )
