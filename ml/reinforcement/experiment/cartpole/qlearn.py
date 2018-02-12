from keras import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

from ml.common.transformer.base import Transformer
from ml.common.classifier.keras import KerasClassifier
from ml.reinforcement.agent.qlearn import QLearningAgent
from ml.reinforcement.experiment.base import ReinforcementLearningExperiment


class CartpoleQlearnKerasExperiment(ReinforcementLearningExperiment):
    name = 'cartpole_qlearn_experiment'

    env_name = 'CartPole-v0'
    env_history_length = 100000
    gamma = 0.95  # discount rate
    epsilon = 1.0  # exploration rate
    epsilon_min = 0.01
    epsilon_decay = 0.995
    learning_rate = 0.001
    observation_shape = (4,)  # 4 features
    actions = [0, 1]  # left, right
    action_shape = (2,)
    action_size = len(actions)
    loss = 'mse'
    optimizer = Adam(lr=learning_rate)
    verbose=True
    transformer = Transformer()

    def get_model(self):
        model = Sequential()
        model.add(Dense(24, input_shape=self.observation_shape, activation='relu'))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(
            loss=self.loss,
            optimizer=self.optimizer
        )

        print(model.summary())

        return model

    def get_agent(self):
        classifier = self.get_classifier()

        return QLearningAgent(
            actions=self.actions,
            classifier=classifier,
            action_shape=self.action_shape,
            observation_shape=self.observation_shape,
            epsilon=self.epsilon,
            epsilon_decay=self.epsilon_decay,
            epsilon_min=self.epsilon_min,
            gamma=self.gamma,
            verbose=self.verbose
        )

    def get_classifier(self):
        self.transformer = Transformer()

        return KerasClassifier(
            model=self.get_model(),
            transformer=self.transformer
        )
