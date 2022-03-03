import collections
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

    self.training_stats = collections.Counter()
    self.test_stats = collections.Counter()

  @staticmethod
  def _n_words_in_batch(words, mask):
    return len(tf.boolean_mask(words, mask))

  def train(self, dataset, epochs: int=10):
    for epoch in range(1, epochs+1):
      for key in self.training_stats:
        self.training_stats[key] = 0.0
      print(f"\n\n{'->' * 12} Training Epoch: {epoch} {'<-' * 12}\n\n")
      self.run_epoch(dataset)
      # Log stats at the end of epoch
      print(f"Training stats: {self.training_stats}")

  def run_epoch(self, dataset):
    """Runs one training epoch."""

    # A dataset is basically a batch of episodes where an episode means a sentence.
    for step, batch in enumerate(dataset):
      correct_labels = batch["dep_labels"]
      correct_labels= correct_labels[:, 1:]
      correct_heads = batch["heads"]
      correct_heads = correct_heads[:, 1:]
      mask = (correct_heads != -1)
      with tf.GradientTape() as tape_actor:
        # Get label predictions (actor predictions) and states from the labeler.
        # States are the LSTM output representation of the tokens coming from labeler.
        # TODO: need to make sure that states are not changing every iteration.
        labeler_scores, states = self.actor.parse(batch)
        # label_scores shape = batch_size, seq_len, n_labels
        # These are raw probabilities rather than softmaxed scores.
        label_scores = labeler_scores["labels"]
        batch_size, sequence_length, _ = label_scores.shape

        # We send the raw probabilities to _label_predictions function to apply
        # tf.random.categorical. label preds shape = batch_size, seq_len
        label_preds = self._label_predictions(label_scores)
        # print("label preds ", label_preds)
        # input("press to cont.")

        # Then we apply softmax
        # softmax_probs shape = batch_size, seq_len, n_labels
        softmax_probs = tf.nn.softmax(label_scores)

        # predicted label probs shape = batch_size, seq_len, 1
        predicted_label_probs = self._predicted_label_probs(softmax_probs)

        # Send label predictions to the parser as input with other features.
        parser_inputs = {"words": batch["words"],
                        "pos": batch["pos"],
                        "dep_labels": label_preds,
                        "morph": batch["morph"]}

        # parser_scores shape: batch_size, seq_len, seq_len
        parser_scores = parser.parse(parser_inputs)

        # Prepare the data to do an actor critic training step.
        # Remove the first token from label preds and log probs, and predicted label probs
        # label preds shape = batch_size, seq_len-1,
        label_preds = label_preds[:, 1:]

        # log_probs shape = batch_size, seq_len-1, n_labels
        softmax_probs = softmax_probs[:, 1:, :]
        # predicted label probs shape = batch_size, seq_len-1, 1
        predicted_label_probs = predicted_label_probs[:, 1:]

        # Get head predictions & remove the first token
        # head preds shape: batch_size, seq_len-1
        head_preds = self._head_predictions(parser_scores["edges"])
        head_preds = head_preds[:, 1:]

        # Remove the first token from the states too.
        # states shape: batch_size, seq_len-1, bilstm_output_size
        states = states[:, 1:, :]
        rewards = self.compute_rewards(head_preds, correct_heads, mask)
        returns = self.compute_returns(rewards, mask)

        with tf.GradientTape() as tape_critic:
          state_values = self.compute_state_values(states, mask)
          critic_loss = self.critic_loss(returns, state_values, mask)
          print("critic_loss ", critic_loss)
          # print([var.name for var in tape_critic.watched_variables()])
          # input("press to cont.")


        advantages =  self.compute_advantages(state_values, returns)

        actor_loss = self.actor_loss(predicted_label_probs, advantages, mask)
        print("actor loss ", actor_loss)
        # print([var.name for var in tape_actor.watched_variables()])
        # input("press to cont.")

      grads_critic = tape_critic.gradient(critic_loss, self.critic.trainable_weights)
      grads_actor = tape_actor.gradient(actor_loss, self.actor.model.trainable_weights)
      self.critic.optimizer.apply_gradients(zip(grads_critic, self.critic.trainable_weights))
      self.actor._optimizer.apply_gradients(zip(grads_actor, self.actor.model.trainable_weights))

      correct_predictions_dict = self._correct_predictions(
        label_predictions=label_preds,
        correct_labels=correct_labels,
        head_predictions=head_preds,
        correct_heads=correct_heads,
        mask=mask,
      )

      n_words_in_batch = self._n_words_in_batch(correct_heads, mask)
      # print("nnumber of words in batch ", n_words_in_batch)
      # input("press to cont.")
      self._update_correct_prediction_stats(correct_predictions_dict, n_words_in_batch)
      # Update stats

    # end for





  def _correct_predictions(self, label_predictions, correct_labels, head_predictions, correct_heads, mask):
    """Returns label accuracy based on a batch."""
    correct_label_preds = tf.boolean_mask(label_predictions == correct_labels, mask)
    n_correct_label_preds = np.sum(correct_label_preds)
    correct_head_preds = tf.boolean_mask(head_predictions == correct_heads, mask)
    n_correct_head_preds = np.sum(correct_head_preds)

    return {
      "chp": correct_head_preds,
      "n_chp": n_correct_head_preds,
      "clp": correct_label_preds,
      "n_clp": n_correct_label_preds,
    }


  def _update_correct_prediction_stats(self,
                                       correct_predictions_dict,
                                       n_words_in_batch,
                                       stats="training"):
    """Updates parsing stats at the end of each training or test step.

    The stats we keep track of are the following:
      n_tokens: total number of tokens in the data.
      n_chp: number of correctly predicted heads.
      n_clp: number of correctly predicted labels.
      n_chlp: number of tokens for which both head and label is correctly predicted.

    These are later used for computing eval metrics like UAS, LS, and LAS.
    """
    if stats == "training":
      stats = self.training_stats
    else:
      stats = self.test_stats

    stats["n_tokens"] += n_words_in_batch
    h, l = None, None

    # Correct head predictions.
    if correct_predictions_dict["n_chp"] is not None:
      stats["n_chp"] += correct_predictions_dict["n_chp"]
      h = correct_predictions_dict["chp"]

    # Correct label predictions.
    if correct_predictions_dict["n_clp"] is not None:
      stats["n_clp"] += correct_predictions_dict["n_clp"]
      l = correct_predictions_dict["clp"]

    # Tokens where both head and label predictions are correct.
    if h is not None and l is not None:
      if not len(h) == len(l):
        raise RuntimeError("Fatal: Mismatch in the number of heads and labels.")
      stats["n_chlp"] += np.sum(
        [1 for tok in zip(h, l) if tok[0] == True and tok[1] == True]
      )

  # TODO potentially change this to tf.function
  def compute_state_values(self, states, mask):
    state_values = self.critic(states)
    mask = tf.expand_dims(mask, -1)
    state_values = state_values * tf.cast(mask, tf.float32)
    return state_values

  def compute_advantages(self, state_values, returns):
    # print("state values ", state_values)
    # print("returns ", returns)
    advantages = returns - state_values
    # print("advantages ", advantages)
    # input("press to cont.")
    return advantages

  def actor_loss(self, action_probs, advantages, mask):
    """"Computes actor loss.
    Args:
        action_probs = softmax_probas. Before computing loss we first convert
          them to log probas.
    """
    action_log_probs = tf.math.log(action_probs)
    l = tf.boolean_mask(action_log_probs * advantages, mask)
    loss = -tf.math.reduce_sum(l)
    return loss

  def critic_loss(self, returns, state_values, mask):
    loss = self.critic.loss(returns, state_values, sample_weight=mask)
    return loss

  def _head_predictions(self, head_scores):
    """Returns head predictions from raw head scores."""
    return tf.argmax(head_scores, 2)

  def _label_predictions(self, label_probs):
    """Returns label predictions and log probabilities from raw label scores.
    Args:
      label_probs: tensor of shape batch_size, seq_len, n_labels.
        Raw probabilities of label predictions (neither softmaxed nor logged).
    """
    label_preds = tf.argmax(label_probs, 2)
    return label_preds

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



# Critic.
class CriticNetwork(tf.keras.Model):
  """The critic is a feed forward regression network."""
  def __init__(self, cr_size, discount_factor):
    super(CriticNetwork, self).__init__()
    self.cr_size = cr_size
    self.gamma = discount_factor
    self.loss = tf.keras.losses.Huber(reduction=tf.keras.losses.Reduction.SUM)
    self.optimizer = tf.keras.optimizers.Adam(lr=0.001)

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




















  """
  def _train_step(self, label_preds, predicted_label_probs,
                        head_preds, correct_heads, states):
    Runs one step of training.
    label_preds: The labeler's label predictions. shape: batch_size, seq_len-1
    predicted_label_probs: The log probabilities of the predicted labels. shape: batch_size, seq_len-1, 1
    head_preds: The parser's head predictions. shape: batch_size, seq_len-1.
    correct_heads: The gold heads. shape: batch_size, seq_len-1.
    states: The token representations coming from the lstm output.
      shape: batch_size, seq_len-1, bilstm_output_size
    
    # Define the pad mask.
    # mask shape = batch_size, seq_len.
    mask = (correct_heads != -1)
    # Compute the rewards per token.
    rewards = self.compute_rewards(head_preds, correct_heads, mask)
    # Compute the returns per token.
    returns = self.compute_returns(rewards, mask)
    # Get the state_value estimates from the critic.

    # Compute advantages based on state values and returns.


    with tf.GradientTape() as tape_critic:
      # y_true=returns, y_pred=state_values
      state_values = self.compute_state_values(states, mask)
      critic_loss = self.critic_loss(returns, state_values, mask)
      print("critic_loss ", critic_loss)
      input("press to cont.")

    with tf.GradientTape() as tape_actor:
      advantages = self.compute_advantages(state_values, returns)
      actor_loss = self.actor_loss(predicted_label_probs, advantages, mask)
      print("actor_loss ", actor_loss)
      input("press to cont.")


    # print("critic trainable weigts ", self.critic.trainable_weights)
    grads_critic = tape_critic.gradient(critic_loss, self.critic.trainable_weights)
    # print("grads critic ", grads_critic)
    # input("press to cont.")

    # print("actor trainable weights ", self.actor.model.trainable_weights)
    grads_actor = tape_actor.gradient(actor_loss, self.actor.model.trainable_weights)

    # See: https://stackoverflow.com/questions/61830841
    print("grads actor ", grads_actor)
    input("press to cont.")
    self.critic.optimizer.apply_gradients(zip(grads_critic, self.critic.trainable_weights))

    self.actor._optimizer.apply_gradients(zip(grads_actor, self.actor.model.trainable_weights))

    return critic_loss, actor_loss
    """