import itertools
import numpy as np
import tensorflow as tf
import os

from tensorflow.keras import layers
from parser.nn import bilstm_labeler
from util.nn import nn_utils
from input import embeddor, preprocessor
from parser.nn import layer_utils, label_first_parser

class ActorCritic:
  """An actor critic model."""
  def __init__(self,
               actor: tf.keras.Model,
               critic: tf.keras.Model,
               parser: tf.keras.Model,
               gamma: float = 0.99):
    self.actor = actor
    self.critic = critic
    self.parser = parser
    self.gamma = gamma

  def train(self, dataset):
    """Trains the actor and the critic."""

    # List of states in an episode.
    states_list = []
    # List of state values predicted by the critic.
    state_values_list = []
    # The actions taken by the actor in an episode.
    actions_taken = []
    # The log probs of the actions taken by the actor in an episode.
    action_log_policies = []

    # A dataset is basically a batch of episodes where an episode means a sentence.
    for step, batch in enumerate(dataset):
      # Get label predictions and states from the labeler
      # States are the LSTM output representation for tokens coming from labeler.
      scores, states = self.actor.parse(batch)
      label_scores = scores["labels"]
      batch_size, sequence_length, _ = label_scores.shape
      log_probs, label_preds = self._label_predictions(label_scores)
      predicted_label_probs = self._predicted_label_probs(log_probs)

      # Send label predictions to the parser as input with other features.
      parser_inputs = {"words": batch["words"],
                       "pos": batch["pos"],
                       "dep_labels": label_preds,
                       "morph": batch["morph"]}
      scores = parser.parse(parser_inputs)

      # Remove the first token from label preds and log probs, and predicted label probs
      label_preds = label_preds[:, 1:]
      log_probs = log_probs[:, 1:, :]
      predicted_label_probs = predicted_label_probs[:, 1:]
      # print("log probs ", log_probs)
      # print("predicted label probs ", predicted_label_probs)
      # print("label preds ", label_preds)
      # input("press to cont.")

      # Get head predictions & remove the first token
      head_preds = self._head_predictions(scores["edges"])
      head_preds = head_preds[:, 1:]

      # Get correct heads and remove the first token again.
      correct_heads = batch["heads"]
      correct_heads = correct_heads[:, 1:]

      # Remove the first token from the states too.
      states = states[:, 1:, :]

      # Do a training step.
      c_loss, a_loss, state_values, returns, advantages, mask = self._train_step(
        label_preds, predicted_label_probs, head_preds, correct_heads, states)

  def _train_step(self, label_preds, predicted_label_probs, head_preds, correct_heads, states):
    """Runs one step of training.
    label_preds: The labeler's label predictions.
    predicted_label_probs: The log probabilities of the predicted labels.
    head_preds: The parser's head predictions.
    correct_heads: The gold heads.
    states: The token representations coming from the lstm output.
    """
    # Define the pad mask.
    mask = (correct_heads != -1)
    # Compute the rewards per token.
    rewards = self.compute_rewards(head_preds, correct_heads, mask)
    # Compute the returns per token.
    returns = self.compute_returns(rewards, mask)
    # Get the state_value estimates from the critic.
    state_values = self.compute_state_values(states, mask)
    # Compute advantages based on state values and returns.
    advantages = self.compute_advantages(state_values, returns)

    with tf.GradientTape() as tape_critic:
      # y_true=returns, y_pred=state_values
      critic_loss = self.critic_loss(returns, state_values, mask)
      print("critic loss ", critic_loss)
      input("press to cont.")

    with tf.GradientTape() as tape_actor:
      actor_loss = self.actor_loss(predicted_label_probs, advantages, mask)
      print("actor loss ", actor_loss)
      print("press to cont.")
    # TODO: continue from here to backprop to critic and the label loss to the models.
    """
    grads_critic = tape_critic.gradient(critic_loss, self.critic.trainable_weights)
    # TODO: change this later. we should be able to just call self.actor.trainable_weights.
    grads_actor = tape_actor.gradient(actor_loss, self.actor.model.trainable_weights)

    self.critic.optimizer.apply_gradients(zip(grads_critic, self.critic.trainable_weights))
    # TODO: change this later. we have to redefine the optimizer of the actor.
    self.actor._optimizer.apply_gradients(zip(grads_actor, self.actor.model.trainable_weights))
    """
    return critic_loss, actor_loss, state_values, returns, advantages, mask

  def _head_predictions(self, head_scores):
    """Returns head predictions from raw head scores."""
    return tf.argmax(head_scores, 2)

  def _label_predictions(self, label_scores):
    """Returns label predictions and log probabilities from raw label scores."""
    log_probs = tf.nn.log_softmax(label_scores)
    label_preds = tf.argmax(log_probs, 2)
    return log_probs, label_preds

  def _predicted_label_probs(self, label_probs):
    """Returns the log prob of the predicted label from label log probs"""
    label_probs = tf.reduce_max(label_probs, axis=2)
    return tf.expand_dims(label_probs, -1)

  def compute_rewards(self, head_predictions, correct_heads, mask):
    """Computes rewards per state based on tree path."""
    rewards = np.zeros_like(head_predictions)
    for i, (h, c, m) in enumerate(zip(head_predictions, correct_heads, mask)):
      r = np.array(tf.cast((h == c), tf.float32))
      np.put(r, tf.where(tf.math.logical_not(m)), -1)
      rewards[i, :] = r
    return tf.convert_to_tensor(rewards)

  def compute_returns(self, rewards, mask, standardize=False):
    """Computes returns per state based on rewards."""
    rewards = np.flip(rewards, axis=1)
    returns = np.zeros_like(rewards, dtype=np.float32)
    n_episodes = rewards.shape[0]
    for i in range(n_episodes):
      episode_reward = tf.cast(rewards[i, :], tf.float32)
      episode_mask = mask[i, :]
      episode_return = self.episode_return(episode_reward, episode_mask)
      returns[i, :] = episode_return

    returns = tf.convert_to_tensor(returns)
    # Standardization is supposed to help stabilize training.
    # This is by default False. We need to test how it affects training.
    if standardize:
      eps = np.finfo(np.float32).eps.item()
      returns = ((returns - tf.math.reduce_mean(returns)) /
                 (tf.math.reduce_std(returns) + eps))
    return tf.expand_dims(returns, -1)

  def episode_return(self, episode_reward, episode_mask):
    """Computes returns over a single episode"""
    discounted_sum = 0.0
    episode_return = np.zeros_like(episode_reward, dtype=np.float32)
    for i in range(episode_reward.shape[0]):
      reward = episode_reward[i]
      if reward == -1:
        episode_return[i] = -1
        continue
      discounted_sum = reward + self.gamma * discounted_sum
      episode_return[i] = discounted_sum
    episode_return = np.flip(episode_return)
    return episode_return

  def compute_state_values(self, states, mask):
    state_values = self.critic(states)
    mask = tf.expand_dims(mask, -1)
    state_values = state_values * tf.cast(mask, tf.float32)
    return state_values

  def compute_advantages(self, state_values, returns):
    advantages = returns - state_values
    return advantages

  def actor_loss(self, action_log_probs, advantages, mask):
    print("predicted label probs ", action_log_probs)
    print("advantages ", advantages)
    print("mask ", mask)
    l = tf.boolean_mask(action_log_probs * advantages, mask)
    print("l ", l)
    input("press to cont.")
    loss = -tf.math.reduce_sum(l)
    return loss
    # return -tf.math.reduce_sum(action_log_probs * advantages)

  def critic_loss(self, returns, state_values, mask):
    loss = self.critic.loss(returns, state_values, sample_weight=mask)
    return loss



# Critic.
class CriticNetwork(tf.keras.Model):
  """The critic is a feed forward regression network."""
  def __init__(self, cr_size, discount_factor):
    super(CriticNetwork, self).__init__()
    self.cr_size = cr_size
    self.gamma = discount_factor
    self.loss = tf.keras.losses.Huber(reduction=tf.keras.losses.Reduction.SUM)

    self.layer1 = layers.Dense(self.cr_size,
                               use_bias=True)
    self.layer2 = layers.Dense(self.cr_size,
                               use_bias=True)
    # Outputs a single value
    self.layer3 = layers.Dense(1, use_bias=True)

  def call(self, states):
    h1 = tf.nn.leaky_relu(self.layer1(states))
    h2 = tf.nn.leaky_relu(self.layer2(h1))
    v = self.layer3(h2)
    # print("v is now ", v)
    # from (n, 1), we transpose to (1, n)
    return v


if __name__ == "__main__":
  embeddings = nn_utils.load_embeddings()
  word_embeddings = embeddor.Embeddings(name="word2vec", matrix=embeddings)
  prep = preprocessor.Preprocessor(
    word_embeddings=word_embeddings,
    features=["words", "pos", "morph", "heads", "dep_labels"],
    labels=["heads"]
  )
  label_feature = next(
    (f for f in prep.sequence_features_dict.values() if f.name == "dep_labels"), None)

  # Load the pretrained LSTM labeler.
  print("Loading the labeler.")
  labeler = bilstm_labeler.BiLSTMLabeler(word_embeddings=prep.word_embeddings,
                                         n_output_classes=label_feature.n_values,
                                         predict=["labels"],
                                         features=["words", "pos", "morph"],
                                         model_name="dependency_labeler_test")
  labeler.load_weights(name="dependency_labeler_test")

  print(labeler.model_name)


  # Load the pretrained label first parser.
  print("Loading the parser.")
  parser = label_first_parser.LabelFirstParser(word_embeddings=prep.word_embeddings,
                                               n_output_classes=label_feature.n_values,
                                               predict=["heads"],
                                               features=["words", "pos", "morph", "heads", "dep_labels"],
                                               model_name="label_first_parser")
  parser.load_weights(name="lfp_test")
  print(parser.model_name)

  actor_critic = ActorCritic(actor=labeler,
                             critic=CriticNetwork(cr_size=512, discount_factor=0.99),
                             parser=parser)

  _TEST_DATA_DIR="data/UDv23/Turkish/test"
  test_treebank = "treebank_test_0_10.conllu"

  test_sentences = prep.prepare_sentence_protos(
    path=os.path.join(_TEST_DATA_DIR, test_treebank))
  test_dataset = prep.make_dataset_from_generator(
    sentences=test_sentences, batch_size=5)

  actor_critic.train(dataset=test_dataset)

  ### Testing for transferring weights
  # If you want to keep training your pretrained model, you need to transfer
  # weights to some new model and keep training from there.
  word_embeddings=layer_utils.EmbeddingLayer(pretrained=labeler.word_embeddings)
  pos_embeddings=layers.Embedding(input_dim=35, output_dim=32)
  pos_embeddings.build((None, 32))
  pos_embeddings.set_weights(labeler.model.pos_embeddings.get_weights())
  # print(pos_embeddings.weights)
  for a, b in zip(pos_embeddings.weights, labeler.model.pos_embeddings.weights):
    np.testing.assert_allclose(a.numpy(), b.numpy())
  # pos_em_weights = pos_embeddings.weights
  # print(pos_em_weights[0][1])
  # print(pos_em_weights)

  ### how to access a specific index's embedding vector.
  # print(word_embeddings.stoi(token="tolga"))
  # print(word_embeddings.itov(idx=493047))
  # weights = labeler.model.word_embeddings.get_weights()
  # print("length of weights ", len(weights))
  # print(weights[0][493047])
  # print("press to cont.")