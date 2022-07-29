"""This is the DQN agent used for label predictions based on feedback from a dependency parser."""

import os
import dataclasses
import logging

import tensorflow as tf
import numpy as np

from parser.nn.combined_parser_memory import Memory
from parser.nn import base_parser
from parser.nn import sequential_parser_exp
from parser.nn import seq_lstm_attn
from parser.nn import base_parser, architectures, layer_utils
from parser.nn import load_models

from util.nn import nn_utils
from input import embeddor, preprocessor
from tensorflow.keras import layers, metrics, losses, optimizers

Embeddings = embeddor.Embeddings
Dataset = tf.data.Dataset

_DATA_DIR="data/UDv29/train/tr"
_TEST_DATA_DIR="data/UDv29/test/tr"

@dataclasses.dataclass
class Experience:
  state: tf.Tensor
  action: int
  action_qs : tf.Tensor
  reward: float
  action_hot: tf.Tensor
  action_name: str
  true_label: str
  edge_correct: bool
  label_correct: bool

class Agent():
  def __init__(self, parser_path, network=None, epsilon_decay=0.1):
    """Initilize the agent.
    Args:
      parser: a pretrained dependency parser loaded from memory.
      network: a dqn network. This is a keras model.
      memory: the dqn memory.
      epsilon_decay: param to use for exploit/explore decision.
      action_size: number of dependendy labels.
    """
    self.memory = Memory(memory_size=50, batch_size=20)
    self.epsilon_decay = epsilon_decay
    self.action_size = 43 # but do something to take label size.
    self.embeddings = nn_utils.load_embeddings()
    self.word_embeddings = embeddor.Embeddings(name="word2vec", matrix=self.embeddings)
    self.prep = load_models.load_preprocessor(self.word_embeddings,
                                              one_hot_features=["dep_labels"])
    self.parser = self._load_parser(path=parser_path, parser_name="sequential_parser_exp_saved")
    self.labeler = self._label_network()

  def collect_data(self, dataset):
    """Gathers data and updates the agent memory.

    Inputs here doesn't contain correct labels.
    """
    for step, batch in enumerate(dataset):
      labeler_inputs = {"words": batch["words"],
                        "pos": batch["pos"],
                        "morph": batch["morph"],
                        "dep_labels": batch["dep_labels"],
                        "heads": batch["heads"],
                        "sent_id": batch["sent_id"]}

      states, label_scores, sentence_repr = self.labeler.model(labeler_inputs)
      label_indexes = tf.argmax(label_scores, axis=2)

      # Convert indexes to one-hot-represenation.
      label_hot = tf.one_hot(label_indexes, self.action_size)
      parser_inputs = {
        "labels": label_hot,
        "words": batch["words"],
        "sentence_repr": sentence_repr,
        "heads": batch["heads"],
        "sent_id": batch["sent_id"][:, :1]
      }

      _, parent_prob_dict = self.parser.model(parser_inputs,
                                              encode_sentence=False,
                                              training=False)
      print("parent table ", parent_prob_dict)
      input("press")
      parent_prob_table = list(parent_prob_dict.values())
      _preds = [tf.math.top_k(prob_table).indices for prob_table in parent_prob_table]
      print("_preds ", _preds)
      head_preds = tf.cast(tf.concat(_preds, 0), tf.int64)
      print("head_preds ", head_preds)
      true_heads = batch["heads"]
      print("true heads ", true_heads)
      true_heads = self._flatten(batch["heads"][:, 1:]) # slice out 0th token
      print("true heads ", true_heads)
      input("press to cont.")

      # Remove the 0the token from states and labels before creating experiences too.
      print("label scores ", label_scores)
      label_scores = label_scores[:, 1:, :]
      print("label scores now", label_scores)
      print("-------------------------")
      print("states ", states)
      print("states==================")
      input("press to cont")
      states = states[:, 1:, :]
      print("states now ", states)
      print("states==================")
      input("press to cont")
      print("label preds ", label_indexes)
      label_preds = label_indexes[:, 1:]
      print("lable preds now ", label_preds)
      input("press ")
      print("label hot ", label_hot)
      label_hot = label_hot[:, 1:, :]
      print("label hot now ", label_hot)
      input("press to cont.")
      experiences = self._make_experience(states, head_preds, true_heads, label_preds,
                                          label_scores, label_hot, true_labels)
      self.memory.update(experiences)
      if self.memory.filled:
        self.learn(self.memory.random_sample())


  def _make_experience(self, states, label_indexes, label_scores, label_hot, parent_table):
    """Creates an experience for memory replay buffer."""
    experiences = []
    predicted_heads = tf.expand_dims(tf.argmax(head_probs, 1), 1)
    true_labels = tf.argmax(true_labels, 1)
    predicted_label_names = [dep_label_tags.Tag.Name(val.numpy()) for val in predicted_labels]
    true_label_names = [dep_label_tags.Tag.Name(val.numpy()) for val in true_labels]



  def _compute_reward(self):
    pass


  def _load_parser(self, path, parser_name):
    # Finally load the parser.
    parser = sequential_parser_exp.SequentialParser(
      word_embeddings=self.prep.word_embeddings,
      predict=["heads"],
      features=["words", "pos", "morph", "dep_labels", "sent_id"],
      test_every=10,
      model_name=parser_name
    )
    parser.load_weights(name=parser_name, path=path)
    logging.info("parser loaded ")
    return parser

  def _label_network(self):
    label_feature = next(
      (f for f in self.prep.sequence_features_dict.values() if f.name == "dep_labels"), None)
    labeler = seq_lstm_attn.SeqLSTMAttnLabeler(
      word_embeddings=self.word_embeddings,
      n_output_classes=label_feature.n_values,
      predict=["labels"],
      features=["words", "pos", "morph"],
      model_name="labeler_agent"
    )
    shared_layers = ["word_embeddings"]
    for layer in self.parser.model.layers:
      if layer.name == "pos_embeddings":
        shared_layers.append(layer.name)
        labeler.model.pos_embeddings.set_weights(layer.get_weights())
        labeler.model.pos_embeddings.trainable=False
        for a, b in zip(self.parser.model.pos_embeddings.weights, labeler.model.pos_embeddings.weights):
          np.testing.assert_allclose(a.numpy(), b.numpy())
      if layer.name == "lstm_encoder":
        shared_layers.append(layer.name)
        labeler.model.pre_attn_lstm.lstm1.set_weights(layer.lstm1.get_weights())
        labeler.model.pre_attn_lstm.lstm1.trainable = False
        labeler.model.pre_attn_lstm.lstm2.set_weights(layer.lstm2.get_weights())
        labeler.model.pre_attn_lstm.lstm2.trainable = False
        for a, b in zip(self.parser.model.encoder.lstm1.weights,
                        labeler.model.pre_attn_lstm.lstm1.weights):
          np.testing.assert_allclose(a.numpy(), b.numpy())
        for a, b in zip(self.parser.model.encoder.lstm2.weights,
                        labeler.model.pre_attn_lstm.lstm2.weights):
          np.testing.assert_allclose(a.numpy(), b.numpy())
    logging.info(f"Weights transferred between parser and labeler. Sharing weights for {shared_layers}.")
    return labeler


  def train(self):
    sample = self.memory.random_sample(20)
    states = sample.states

    self.network.compile(loss="mse")
    self.network()

  @staticmethod
  def _flatten(_tensor, outer_dim=1):
    """Flattens a 3D tensor into a 2D one.

    A tensor of [batch_size, seq_len, outer_dim] is converted to ([batch_size*seq_len], outer_dim)
    ie. into a two-dim tensor
    """
    batch_size, seq_len = _tensor.shape[0], _tensor.shape[1]
    return tf.reshape(_tensor, shape=(batch_size*seq_len, outer_dim))

if __name__ == "__main__":
  agent = Agent(parser_path="./model/nn/pretrained")

  train_treebank="tr_boun-ud-train-random500.pbtxt"
  test_treebank = "tr_boun-ud-test-random50.pbtxt"

  train_sentences = agent.prep.prepare_sentence_protos(
    path=os.path.join(_DATA_DIR, train_treebank))
  test_sentences = agent.prep.prepare_sentence_protos(
    path=os.path.join(_TEST_DATA_DIR, test_treebank)
  )
  dataset = agent.prep.make_dataset_from_generator(
    sentences=train_sentences,
    batch_size=2)
  test_dataset = agent.prep.make_dataset_from_generator(
    sentences=test_sentences,
    batch_size=20
  )
  agent.collect_data(dataset=dataset)
