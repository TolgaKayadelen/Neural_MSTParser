import itertools
import numpy as np
import tensorflow as tf
import os

from tensorflow.keras import layers
from parser.nn import bilstm_labeler
from util.nn import nn_utils
from input import embeddor, preprocessor
from parser.nn import layer_utils, label_first_parser

class RLTrain:
  """Applies actor critic training to the output of an RNN."""
  def __init__(self,
               labeler,
               parser,
               bilstm_output_size: int,
               gamma: float = 1):
    self.labeler = labeler
    self.parser = parser
    self.gamma = gamma
    self.critic = CriticNetwork(bilstm_output_size, gamma)

  def v_loss(self, returns, prev_v):
    """Returns least square loss for V.

    L2 regularization is done by the optimizer.
    The v loss is the target regression value for critic V.
    prev_V is the previous estimate of critic v for the returns.
    We want to minimize the MSE between returns and prev_V.
    """
    pass
    # return MSEloss

  def train(self, dataset):
    """Trains the actor critic."""

    # Critic V estimates
    states = []
    actions_taken = []
    action_log_policies = []

    # Start looping through the data coming from the decoder.
    for step, batch in enumerate(dataset):
      scores = labeler.parse(batch)
      label_scores = scores["labels"]
      batch_size, sequence_length, _ = label_scores.shape
      log_probs, label_preds = self._label_predictions(label_scores)
      actions_taken.append(label_preds)
      action_log_policies.append(log_probs)
      parser_inputs = {"words": batch["words"],
                       "pos": batch["pos"],
                       "dep_labels": label_preds,
                       "morph": batch["morph"]}
      scores = parser.parse(parser_inputs)
      head_preds = self._head_predictions(scores["edges"])
      head_preds = head_preds[:, 1:]
      print("head preds ", head_preds)
      correct_heads = batch["heads"]
      correct_heads = correct_heads[:, 1:]
      print("correct heads ", correct_heads)
      input("press to cont.")
      for pred, correct in zip(head_preds, correct_heads):
        episode_rewards = self.compute_rewards(pred, correct)
        print("episode rewards ", episode_rewards)
        input("press to cont.")

  def _head_predictions(self, head_scores):
    """Returns head predictions from raw head scores."""
    return tf.argmax(head_scores, 2)


  def _label_predictions(self, label_scores):
    """Returns label predictions and log probabilities from raw label scores."""
    log_probs = tf.nn.log_softmax(label_scores)
    label_preds = tf.argmax(log_probs, 2)
    return log_probs, label_preds

  # TODO: implement this one.
  def actor_critic(self, taken_actions, action_log_policies, parser_input):
    head_scores = parser.parse(parser_input)
    # predicted_heads = # do something to head scores
    is_true_head = tf.boolean_mask(predicted_heads == heads, pad_mask)
    rewards, tree_structure = tf.cast(is_true_head, tf.float32)
    # state =
    returns = compute_returns(rewards, predicted_heads, parser_input["words"])
    advantages = returns - V_es
    print(advantages)

  def compute_rewards(self, head_predictions, correct_heads):
    """Computes rewards per state based on tree path."""
    print("head preds ", head_predictions)
    print("correect heads ", correct_heads)
    input("press to cont.")
    rewards = np.array(tf.cast((head_predictions == correct_heads), tf.float32))
    np.put(rewards, tf.where(correct_heads==[-1]), [-1])
    return rewards

class CriticNetwork(tf.keras.Model):
  """The critic is a feed forward regression network."""
  def __init__(self, cr_size, discount_factor):
    super(CriticNetwork, self).__init__()
    self.cr_size = cr_size
    self.gamma = discount_factor

    self.layer1 = layers.Dense(self.cr_size,
                               use_bias=True)
    self.layer2 = layers.Dense(self.cr_size,
                               use_bias=True)
    # Outputs a single value
    self.layer3 = layers.Dense(1, use_bias=True)

  def call(self):
    h1 = tf.nn.leaky_relu(self.layer1(states))
    h2 = tf.nn.leaky_relu(self.layer2(h1))
    # h3 is now scalar between 0 and 1
    h3 = tf.nn.sigmoid(self.layer3(h2))

    # v is now scalar between 0 and 1-gamma
    # which are the boundaries for returns w.r.t l and 0/1 rewards.
    v = tf.nn.divide(h3, 1. - self.gamma)

    # TODO: track what this is and set up batch_size
    # and seq_len
    return tf.reshape(v, [self.batch_size, self.seq_len])




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
                                         model_name="dependency_labeler")
  labeler.load_weights(name="dependency_labeler")

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
  input("press to cont.")
  rl_train = RLTrain(labeler=labeler,
                     parser=parser,
                     bilstm_output_size=512
                     )

  _TEST_DATA_DIR="data/UDv23/Turkish/test"
  test_treebank = "treebank_test_0_10.conllu"

  test_sentences = prep.prepare_sentence_protos(
    path=os.path.join(_TEST_DATA_DIR, test_treebank))
  test_dataset = prep.make_dataset_from_generator(
    sentences=test_sentences, batch_size=5)

  rl_train.train(dataset=test_dataset)



  """
  # Testing for transferring weights define a sequential model
  word_inputs = tf.keras.Input(shape=(None, ), name="wordstest")
  pos_inputs = tf.keras.Input(shape=(None, ), name="pos")
  morph_inputs = tf.keras.Input(shape=(None, 66), name="morph")
  word_embeddings=layer_utils.EmbeddingLayer(pretrained=labeler.word_embeddings)
  pos_embeddings=layers.Embedding(input_dim=35, output_dim=32)
  pos_embeddings.build((None, 32))
  pos_embeddings.set_weights(labeler.model.pos_embeddings.get_weights())
  # print(pos_embeddings.weights)
  for a, b in zip(pos_embeddings.weights, labeler.model.pos_embeddings.weights):
    np.testing.assert_allclose(a.numpy(), b.numpy())
  print()
  # how to access a specific index's embedding vector.
  print(word_embeddings.stoi(token="tolga"))
  print(word_embeddings.itov(idx=493047))
  # weights = labeler.model.word_embeddings.get_weights()
  # print("length of weights ", len(weights))
  # print(weights[0][493047])
  # print("press to cont.")
  """