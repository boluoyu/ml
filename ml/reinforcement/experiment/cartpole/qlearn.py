import numpy as np

from keras import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

from ml.reinforcement.experiment.base import QlearningReinforcementLearningExperiment


class CartpoleQlearnKerasExperiment(QlearningReinforcementLearningExperiment):
    name = 'cartpole_qlearn_experiment'

    env_history_size = 10000
    gamma = 0.95  # discount rate
    epsilon = 1.0  # exploration rate
    epsilon_min = 0.01
    epsilon_decay = 0.995
    learning_rate = 0.001
    observation_shape = [1, 4]
    actions = [0, 1]
    action_size = len(actions)

    def get_model(self):
        model = Sequential()
        model.add(Dense(24, input_dim=self.observation_shape, activation='relu'))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(
            loss='mse',
            optimizer=Adam(lr=self.learning_rate)
        )

        print(model.summary())

        return model
