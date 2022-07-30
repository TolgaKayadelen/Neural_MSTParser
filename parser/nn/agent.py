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
from tagset.dep_labels import dep_label_enum_pb2 as dep_label_tags
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
  true_label: int
  true_label_name: str
  word: int
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
    self.memory = Memory(memory_size=50, batch_size=2)
    self.epsilon_decay = epsilon_decay
    self.action_size = 43 # but do something to take label size.
    self.embeddings = nn_utils.load_embeddings()
    self.word_embeddings = embeddor.Embeddings(name="word2vec", matrix=self.embeddings)
    self.prep = load_models.load_preprocessor(self.word_embeddings,
                                              one_hot_features=["dep_labels"])
    self.parser = self._load_parser(path=parser_path, parser_name="sequential_parser_exp_saved")
    self.labeler = self._label_network()
    self.rl_optimizer = tf.keras.optimizers.RMSprop(0.1)
    self.rl_loss = tf.keras.losses.MeanSquaredError()

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


  def learn(self, dataset):
    """Runs backpropagation based on a sample of data.

    This method is used if we just want to compute backpropagation based on an existing sample which contains
    results from forward prop (predicted-labels). We only compute loss on send gradients back to the network.

    This method is only used for Reinforcement Learning Based training scenarios and the loss function and
    optimization method is different that the ones used in supervised training.
    """
    # unpack the sample
    for step, batch in enumerate(dataset):
      words = batch["words"]
      labeler_inputs = {"words": words,
                        "pos": batch["pos"],
                        "morph": batch["morph"],
                        "dep_labels": batch["dep_labels"],
                        "heads": batch["heads"],
                        "sent_id": batch["sent_id"]}
      words = self._flatten(words[:, 1:])
      pad_mask = (words == 0)

      with tf.GradientTape() as tape:
        states, label_scores, sentence_repr = self.labeler.model(labeler_inputs)
        label_indexes = tf.argmax(label_scores, axis=2)

        # Convert indexes to one-hot-represenation.
        pred_label_hot = tf.one_hot(label_indexes, self.action_size)
        parser_inputs = {
          "labels": pred_label_hot,
          "words": batch["words"],
          "sentence_repr": sentence_repr,
          "heads": batch["heads"],
          "sent_id": batch["sent_id"][:, :1]
        }
        _, parent_prob_dict = self.parser.model(parser_inputs,
                                                encode_sentence=False,
                                                training=False)
        parent_prob_table = list(parent_prob_dict.values())
        _preds = [tf.math.top_k(prob_table).indices for prob_table in parent_prob_table]
        head_preds = tf.cast(tf.concat(_preds, axis=0), tf.int64)

        true_heads = batch["heads"]
        true_heads = self._flatten(batch["heads"][:, 1:]) # slice out 0th token

        # Remove the 0the token from states and labels before creating experiences too.
        # print("label scores ", label_scores)
        label_scores = label_scores[:, 1:, :]
        label_scores = self._flatten(label_scores, outer_dim=label_scores.shape[2])

        states = states[:, 1:, :]
        states = self._flatten(states, outer_dim=states.shape[2])

        pred_label_indexes = label_indexes[:, 1:]
        pred_label_indexes = self._flatten(pred_label_indexes)

        pred_label_hot = pred_label_hot[:, 1:, :]
        pred_label_hot = self._flatten(pred_label_hot, outer_dim=pred_label_hot.shape[2])

        true_labels = batch["dep_labels"][:, 1:]
        true_labels = self._flatten(tf.argmax(true_labels, 2))

        experiences = self._make_experience(states, head_preds, true_heads, pred_label_indexes,
                                            label_scores, pred_label_hot, true_labels, pad_mask,
                                            words)
        self.memory.update(experiences)
        print(f"length of memory: {len(self.memory)}")
        print("memory filled ", self.memory.filled())
        if not self.memory.filled():
          logging.info("Memory is not filled, sampling more data.")
          input()
          continue
        else:
          logging.info("Memory is filled, learning..")
          states, actions, action_q_scores, rewards, actions_hot, action_names = self.memory.random_sample()
          # The action_q_scores is the model's prediction of actions.
          print("action qs ", action_q_scores)
          print("actions hot ", actions_hot)
          input("#########################")
          # the target q scores are rewards. This is similar to Bellman equation but we don't compute target q_scores
          # based on the expected outcome of the next state, as the next state doesn't depend on the action I take in
          # the current state, it's just the next token in seqeuence. Therefore, the target_q_score is just the immediate
          # reward, which is computed based on whether the label and the parser predictions where correct or not.
          target_q_scores = tf.expand_dims(rewards, 1)
          print("target_q_scores ", target_q_scores)
          # Here we put the reward values as the target_q_values for the output scores of the relevatn nodes and
          # convert all other nodes to zero. What we want to teach the system is to predict this target value
          # for the relevant node in the given state, without teaching anything about the other values.
          target_q_scores_masked = tf.multiply(target_q_scores, tf.cast(actions_hot, tf.int32))
          action_q_scores_masked = tf.multiply(action_q_scores, actions_hot)
          print("target_qs_masked ", target_q_scores_masked)
          print("action_q_scores ", action_q_scores)
          print("action_q_scores masked ", action_q_scores_masked)
          input("##########################")
          loss = self.rl_loss(target_q_scores_masked, action_q_scores_masked)
          print("loss is ", loss)
      # end tape.
          # input("press: printintg trainable weigths of the mdoel ")
          # print(self.labeler.model.trainable_weights)
      # input("prss: printed traianble weights")
      grads = tape.gradient(loss, self.labeler.model.trainable_weights)
      print("grads are ", grads)
      input("-----------------")
    # In this setting, action_qs are label scores, rewards are target_qs.
    # loss = self.rl_loss(target_q_scores_masked, action_q_scores)

  def collect_data(self, dataset):
    """Gathers data and updates the agent memory.

    Inputs here doesn't contain correct labels.
    """
    for step, batch in enumerate(dataset):
      words = batch["words"]
      labeler_inputs = {"words": words,
                        "pos": batch["pos"],
                        "morph": batch["morph"],
                        "dep_labels": batch["dep_labels"],
                        "heads": batch["heads"],
                        "sent_id": batch["sent_id"]}
      words = self._flatten(words[:, 1:])
      pad_mask = (words == 0)
      states, label_scores, sentence_repr = self.labeler.model(labeler_inputs)
      label_indexes = tf.argmax(label_scores, axis=2)

      # Convert indexes to one-hot-represenation.
      pred_label_hot = tf.one_hot(label_indexes, self.action_size)
      parser_inputs = {
        "labels": pred_label_hot,
        "words": batch["words"],
        "sentence_repr": sentence_repr,
        "heads": batch["heads"],
        "sent_id": batch["sent_id"][:, :1]
      }
      _, parent_prob_dict = self.parser.model(parser_inputs,
                                              encode_sentence=False,
                                              training=False)
      parent_prob_table = list(parent_prob_dict.values())
      _preds = [tf.math.top_k(prob_table).indices for prob_table in parent_prob_table]
      head_preds = tf.cast(tf.concat(_preds, axis=0), tf.int64)

      true_heads = batch["heads"]
      true_heads = self._flatten(batch["heads"][:, 1:]) # slice out 0th token

      # Remove the 0the token from states and labels before creating experiences too.
      # print("label scores ", label_scores)
      label_scores = label_scores[:, 1:, :]
      label_scores = self._flatten(label_scores, outer_dim=label_scores.shape[2])

      states = states[:, 1:, :]
      states = self._flatten(states, outer_dim=states.shape[2])

      pred_label_indexes = label_indexes[:, 1:]
      pred_label_indexes = self._flatten(pred_label_indexes)

      pred_label_hot = pred_label_hot[:, 1:, :]
      pred_label_hot = self._flatten(pred_label_hot, outer_dim=pred_label_hot.shape[2])

      true_labels = batch["dep_labels"][:, 1:]
      true_labels = self._flatten(tf.argmax(true_labels, 2))

      experiences = self._make_experience(states, head_preds, true_heads, pred_label_indexes,
                                          label_scores, pred_label_hot, true_labels, pad_mask,
                                          words)
      self.memory.update(experiences)
      print(f"length of memory: {len(self.memory)}")
      print("memory filled ", self.memory.filled())
      if self.memory.filled():
        logging.info("Memory is filled, running a learning loop.")
        self.learn()

  def _make_experience(self, states,
                       pred_heads,  # shape: batch_size*seq_len, 1
                       true_heads,  # shape: batch_size*seq_len, 1
                       pred_label_indexes, # shape: batch_size*seq_len, 1
                       pred_label_scores, # shape: batch_size*seq_len, 43
                       pred_label_hot, # shape: batch_size*seq_len, 43
                       true_labels, # shape: batch_size*seq_len, 1
                       pad_mask, # shape: batch_size*seq_len, 1
                       words): # shape: batch_size*seq_len, 1
    """Creates an experience for memory replay buffer."""
    experiences = []
    seq_len = states.shape[0]
    for i in range(seq_len):
      if tf.keras.backend.get_value(pad_mask[i]) == True:
        continue
      pred_label_index = tf.keras.backend.get_value(pred_label_indexes[i])
      true_label_index = tf.keras.backend.get_value(true_labels[i])
      pred_label_name = self._label_index_to_name(pred_label_index)
      true_label_name = self._label_index_to_name(true_label_index)
      # print(f"""pred heads: {pred_heads[i]}, "pred_heads": {pred_heads},
      #           true_heads: {true_heads[i]}, "true_heads": {true_heads},
      #           label_hot: {pred_label_hot[i]}, "label_hots": {pred_label_hot}
      #           label_preds: {pred_label_indexes[i]}, "label_preds: {pred_label_indexes},
      #           true_labels: {true_labels[i]}, "true_labels: {true_labels}
      #           pred_label_name: {pred_label_name},
      #           true_label_name: {true_label_name}""")
      # print("i: ", i)
      # input()
      reward, edge_correct, label_correct = self._compute_reward(
        pred_heads[i], true_heads[i], pred_label_index, true_label_index,
        pred_label_name
      )
      experience = Experience(
        state=states[i],
        action=pred_label_index,
        action_qs=pred_label_scores[i],
        reward=reward,
        action_hot=pred_label_hot[i],
        action_name=pred_label_name,
        true_label=true_labels[i].numpy()[0],
        true_label_name=true_label_name,
        word=self.word_embeddings.itos(idx=words[i].numpy()[0]),
        edge_correct=edge_correct,
        label_correct=label_correct
      )
      experiences.append(experience)
    return experiences

  def _compute_reward(self, predicted_head,
                      true_head, predicted_label, true_label,
                      predicted_label_name):
    edge_correct, label_correct = False, False
    reward = 0
    if predicted_head == true_head:
      # print("edge correct")
      edge_correct = True
    if predicted_label == true_label:
      # print("label correct")
      label_correct = True
    if edge_correct:
      reward += 1
      if label_correct:
        reward += 1 # when both edge and label is correct, reward is 2.
      else:
        reward -= 1 # when edge is correct but label is not, reward equals 0.
    else:
      if label_correct:
        reward += 1 # when edge is false but label correct, reward is 1.
      else:
        reward -= 1 # if both edge and label is false, reward -1.
    return reward, edge_correct, label_correct

  @staticmethod
  def _flatten(_tensor, outer_dim=1):
    """Flattens a 3D tensor into a 2D one.

    A tensor of [batch_size, seq_len, outer_dim] is converted to ([batch_size*seq_len], outer_dim)
    ie. into a two-dim tensor
    """
    batch_size, seq_len = _tensor.shape[0], _tensor.shape[1]
    return tf.reshape(_tensor, shape=(batch_size*seq_len, outer_dim))

  @staticmethod
  def _label_index_to_name(label_index):
    return dep_label_tags.Tag.Name(label_index[0])

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
  agent.learn(dataset=dataset)
