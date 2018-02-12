import logging
import numpy as np
import tensorflow as tf

from ml.reinforcement.agent.ml import MLAgent

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ActorCriticAgent(MLAgent):
    name = 'actor_critic_agent'

    def __init__(self, session, actor_classifier, target_actor_classifier, critic_classifier, target_critic_classifier, actions,
                 actor_observation_input, critic_observation_input, critic_action_input, action_shape, observation_shape, tau,
                 epsilon, epsilon_min, epsilon_decay, gamma, learning_rate, verbose=True):

        self.tau = tau
        self.actor = actor_classifier
        self.target_actor = target_actor_classifier
        self.critic = critic_classifier
        self.target_critic = target_critic_classifier
        self.learning_rate = learning_rate

        self.actor_observation_input = actor_observation_input
        self.critic_action_input = critic_action_input
        self.critic_observation_input = critic_observation_input

        self.session = session

        super().__init__(
            actions=actions,
            action_shape=action_shape,
            observation_shape=observation_shape,
            epsilon=epsilon,
            epsilon_min=epsilon_min,
            epsilon_decay=epsilon_decay,
            gamma=gamma,
            classifier=actor_classifier,
            verbose=verbose
        )

        self.actor_critic_gradients = tf.placeholder(tf.float32, shape=[None, self.action_size])  # where we will feed de/dC (from critic)
        self.actor_model_weights = self.actor.model.trainable_weights
        self.actor_gradients = tf.gradients(  # dC/dA (from actor)
            ys=self.actor.model.output,
            xs=self.actor_model_weights,
            grad_ys=-self.actor_critic_gradients
        )

        self.actor_gradients_and_weights = zip(self.actor_gradients, self.actor_model_weights)
        self.optimize = tf.train.AdamOptimizer(self.learning_rate).apply_gradients(grads_and_vars=self.actor_gradients_and_weights)

        self.critic_gradients = tf.gradients(
            ys=self.critic.model.output,
            xs=self.critic_action_input
        )

        self.session.run(tf.global_variables_initializer())


    def fit(self, env_state_mini_batch):
        self.train_critic(env_state_mini_batch)
        self.train_actor(env_state_mini_batch)
        self.update_target_models()
        self.update_epsilon()

    def train_actor(self, env_state_mini_batch):
        for env_state in env_state_mini_batch:
            observation = env_state['observation']

            predicted_action = self.actor.predict(X=[observation])[0]
            predicted_action = np.array(predicted_action)
            predicted_action = np.reshape(predicted_action, [1, 2])

            observation = np.array(observation)
            observation = np.reshape(observation, [1, 4])

            gradients = self.session.run(
                fetches=self.critic_gradients,
                feed_dict={
                    self.critic_observation_input: observation,
                    self.critic_action_input:      predicted_action
                })[0]

            self.session.run(
                fetches=self.optimize,
                feed_dict={
                    self.actor_observation_input: observation,
                    self.actor_critic_gradients: gradients
                })

    def train_critic(self, env_state_mini_batch):
        X_actions = []
        X_observations = []
        y = []
        for env_state in env_state_mini_batch:
            observation = env_state['observation']
            action = env_state['action']
            reward = env_state['reward']
            next_observation = env_state['next_observation']
            done = env_state['done']

            if done:
                future_reward = reward
            else:
                target_action = self.target_actor.predict(X=[next_observation])[0]  # actor picks action
                future_reward = (reward + self.gamma *  # critic computes value of actors predicted action
                                 self.target_critic.predict(X=[[target_action], [next_observation]])[
                                     0])  # discounted max future reward for the next state (Bellman EQ)

            action_arr = [0 for _ in range(self.action_size)]
            action_arr[action] = 1
            action_arr = np.array(action_arr)
            X_actions.append(action_arr)
            X_observations.append(observation)
            y.append([future_reward])  # critic learns to produce better future reward estimates

        self.critic.fit(X=[X_actions, X_observations], y=y, epochs=1)

    def update_actor_target(self):
        actor_model_weights = self.actor.model.get_weights()
        actor_target_weights = self.target_actor.model.get_weights()

        for i in range(len(actor_target_weights)):
            actor_target_weights[i] = actor_model_weights[i]
        self.target_actor.model.set_weights(actor_target_weights)

    def update_critic_target(self):
        critic_model_weights = self.critic.model.get_weights()
        critic_target_weights = self.target_critic.model.get_weights()

        for i in range(len(critic_target_weights)):
            critic_target_weights[i] = critic_model_weights[i]
        self.target_critic.model.set_weights(critic_target_weights)

    def update_target_models(self):
        self.update_actor_target()
        self.update_critic_target()

    def predict(self, observation):
        predictions = self.classifier.predict(X=[observation])
        prediction = predictions[0]
        return np.argmax(prediction)

