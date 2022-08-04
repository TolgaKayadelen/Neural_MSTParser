"""This is the DQN agent used for label predictions based on feedback from a dependency parser."""

import os
import dataclasses
import logging
import random

import tensorflow as tf
import numpy as np

from parser.nn.combined_parser_memory import Memory
from parser.nn import base_parser
from parser.nn import sequential_parser_exp
from parser.nn import seq_lstm_attn
from parser.nn import rl_env
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
  reward: float
  action_hot: tf.Tensor
  action_name: str
  true_label: int
  true_label_name: str
  word: int
  edge_correct: bool
  label_correct: bool
  action_provenance: str

class Agent():
  def __init__(self, parser_path, network=None):
    """Initilize the agent.
    Args:
      parser: a pretrained dependency parser loaded from memory.
      network: a dqn network. This is a keras model.
      memory: the dqn memory.
      epsilon_decay: param to use for exploit/explore decision.
      action_size: number of dependendy labels.
    """
    self.memory = Memory(memory_size=50, batch_size=30)
    self.embeddings = nn_utils.load_embeddings()
    self.word_embeddings = embeddor.Embeddings(name="word2vec", matrix=self.embeddings)
    self.prep = load_models.load_preprocessor(self.word_embeddings,
                                              one_hot_features=["dep_labels"])
    label_feature = next(
      (f for f in self.prep.sequence_features_dict.values() if f.name == "dep_labels"), None)
    self.action_size = label_feature.n_values
    self.parser = self._load_parser(path=parser_path, parser_name="sequential_parser_exp_real_test")
    self.labeler = self._label_network()
    self.env = self._set_up_env()
    self.epsilon = 0.95
    self.epsilon_decay = 0.95

  def _load_parser(self, path, parser_name):
    # Finally load the parser.
    parser = sequential_parser_exp.SequentialParser(
      word_embeddings=self.prep.word_embeddings,
      predict=["heads"],
      features=["words", "pos", "morph", "dep_labels", "sent_id"],
      model_name=parser_name
    )
    parser.load_weights(name=parser_name, path=path)
    logging.info("parser loaded ")
    return parser

  def _set_up_env(self):
    label_feature = next(
      (f for f in self.prep.sequence_features_dict.values() if f.name == "dep_labels"), None)
    env = rl_env.State(
      word_embeddings=self.prep.word_embeddings,
      use_morph=True,
      use_pos=True,
    )
    env.word_embeddings.trainable=False
    for layer in self.parser.model.layers:
      if layer.name == "pos_embeddings":
        env.pos_embeddings.set_weights(layer.get_weights())
        env.pos_embeddings.trainable=False
        for a, b in zip(self.parser.model.pos_embeddings.weights, env.pos_embeddings.weights):
          np.testing.assert_allclose(a.numpy(), b.numpy())
      if layer.name == "lstm_encoder":
        env.encoder.trainable=False
        env.encoder.lstm1.set_weights(layer.lstm1.get_weights())
        env.encoder.lstm1.trainable = False
        env.encoder.lstm2.set_weights(layer.lstm2.get_weights())
        env.encoder.lstm2.trainable = False
        for a, b in zip(self.parser.model.encoder.lstm1.weights,
                        env.encoder.lstm1.weights):
          np.testing.assert_allclose(a.numpy(), b.numpy())
        for a, b in zip(self.parser.model.encoder.lstm2.weights,
                        env.encoder.lstm2.weights):
          np.testing.assert_allclose(a.numpy(), b.numpy())
    logging.info("Weights set up for environment.")
    for layer in env.layers:
      # print("layer name ", layer.name)
      # print("layer trainable ", layer.trainable)
      if layer.name == "lstm_encoder":
        print(env.encoder.lstm1.name, env.encoder.lstm1.trainable)
        print(env.encoder.lstm2.name, env.encoder.lstm2.trainable)
      # input("press to cont.")
    return env

  def _label_network(self):
    model = tf.keras.Sequential()
    model.add(tf.keras.Input(shape=(None, 512)))
    # model.add(layers.Dense(128, activation="relu", name="dense1"))
    # model.add(layers.Dense(64, activation="relu", name="dense2"))
    model.add(layers.Dense(self.action_size, name="output"))
    model.compile(optimizer=optimizers.Adam(0.001), loss="mse")
    model.summary()
    return model


  def _episode_reward(self, total_tokens, total_rewards):
    """Given total tokens and totak rewards collected in an episode, returns the episode reward.

    Args:
      total_tokens: int. Number of tokens in a batch.
      total_reward: int. Amount of rewards collected in a batch.

    Returns:
      float: ratio of rewards to the tokens.

    The episode reward is the ratio of rewards to the tokens in a batch normalized to a 0-1 scale.
    """
    # At most, we can get 2 reward points per token (edge_correct, label_correct).
    max_reward_for_episode = total_tokens*200
    min_reward_for_episode = -total_tokens*100

    episode_reward = (total_rewards - min_reward_for_episode) / (max_reward_for_episode - min_reward_for_episode)
    return episode_reward

  def epsilon_update(self, performance_on_epoch):
    # We do an epsilon update at the end of each epoch based on agent's performance on the epoch.
    # As the agent's performance increases, the likelihood of choosing random actions decreases.
    self.epsilon = performance_on_epoch

  def act(self, states):
    """Chooses action based on epsilon-greedy policy."""
    label_indexes = []
    action_provenances = []
    batch_size, seq_len = states.shape[0], states.shape[1]
    # print("batch_size ", batch_size, "seq_len ", seq_len)
    # input("press to cont.")
    # epsilon is a measure of agent's performance on an epoch.
    for i in range(seq_len):
      state_slice = states[:, i, :]
      # print("state slice shape ", state_slice.shape)
      # print("epsilon ", self.epsilon)
      if np.random.rand() > self.epsilon:
        # print("Exploring")
        label_slice = np.random.randint(1, self.action_size, size=(state_slice.shape[0], 1))
        # print("label ", label_slice)
        action_provenance = [["explore"]] * batch_size
      else:
        # print("Exploiting!")
        label_scores = self.labeler.predict(tf.expand_dims(state_slice, 1))
        label_slice = tf.argmax(label_scores, axis=2)
        # print("label from model ", label_slice)
        action_provenance = [["exploit"]] * batch_size
      label_indexes.append(label_slice)
      action_provenances.append(action_provenance)
      # input()
    label_indexes = tf.convert_to_tensor(tf.concat(label_indexes, 1))
    action_provenances = tf.convert_to_tensor(tf.concat(action_provenances, 1))
    # print("label indexes ", label_indexes)
    # print("action provenances ", action_provenances)
    # input()
    return label_indexes, action_provenances

  def learn(self, dataset, epoch):
    """Runs backpropagation based on a sample of data.

    This method is used if we just want to compute backpropagation based on an existing sample which contains
    results from forward prop (predicted-labels). We only compute loss on send gradients back to the network.

    This method is only used for Reinforcement Learning Based training scenarios and the loss function and
    optimization method is different that the ones used in supervised training.
    """
    # unpack the sample
    epoch_reward = 0
    epoch_tokens = 0
    for step, batch in enumerate(dataset):
      words = batch["words"]
      inputs = {"words": words,
                "pos": batch["pos"],
                "morph": batch["morph"],
                "dep_labels": batch["dep_labels"],
                "heads": batch["heads"],
                "sent_id": batch["sent_id"]}
      words = self._flatten(words[:, 1:])
      pad_mask = (words == 0)

      states = self.env(inputs)
      # print("states coming from inputs ", states.shape)
      # input()
      label_scores = self.labeler.predict(states)
      # print("label indexes from network ", tf.argmax(label_scores, axis=2))
      # input()
      label_indexes, action_provenances = self.act(states)
      # print("label indexes from exploration ", label_indexes)
      # input()

      # Convert indexes to one-hot-represenation.
      pred_label_hot = tf.one_hot(label_indexes, self.action_size)
      # print("label hot ", pred_label_hot)
      parser_inputs = {
        "labels": pred_label_hot,
        "words": batch["words"],
        "sentence_repr": states,
        "heads": batch["heads"],
        "sent_id": batch["sent_id"][:, :1]
      }
      _, parent_prob_dict = self.parser.model(parser_inputs,
                                              encode_sentence=False,
                                              training=False)
      parent_prob_table = list(parent_prob_dict.values())
      _preds = [tf.math.top_k(prob_table).indices for prob_table in parent_prob_table]
      head_preds = tf.cast(tf.concat(_preds, axis=0), tf.int64)

      states = states[:, 1:, :]
      states = self._flatten(states, outer_dim=states.shape[2])

      true_heads = batch["heads"]
      true_heads = self._flatten(batch["heads"][:, 1:]) # slice out 0th token

      # Remove the 0the token from states and labels before creating experiences too.
      # print("label scores ", label_scores)
      # label_scores = label_scores[:, 1:, :]
      # label_scores = self._flatten(label_scores, outer_dim=label_scores.shape[2])

      action_provenances = action_provenances[:, 1:]
      action_provenances = self._flatten(action_provenances)

      pred_label_indexes = label_indexes[:, 1:]
      pred_label_indexes = self._flatten(pred_label_indexes)
      # print("pred label indexes ", pred_label_indexes)
      # print("action provenances ", action_provenances)
      # input("press to cont.")

      pred_label_hot = pred_label_hot[:, 1:, :]
      pred_label_hot = self._flatten(pred_label_hot, outer_dim=pred_label_hot.shape[2])

      true_labels = batch["dep_labels"][:, 1:]
      true_labels = self._flatten(tf.argmax(true_labels, 2))

      experiences, total_tokens, total_reward = self._make_experience(states, head_preds, true_heads,
                                                                      pred_label_indexes, pred_label_hot,
                                                                      true_labels, pad_mask, words,
                                                                      action_provenances)
      episode_reward = self._episode_reward(total_tokens, total_reward)
      # print(f"episode reward={episode_reward}")
      epoch_tokens += total_tokens
      epoch_reward += total_reward
      self.memory.update(experiences)
      # print(f"length of memory: {len(self.memory)}")
      if not self.memory.filled():
        # logging.info("Memory is not filled, sampling more data.")
        continue
      else:
        # logging.info("Memory is filled, running replay..")
        self._replay(epoch)
    if epoch % 200 == 0:
      print("number of examples with correct label prediction in memory: ")
      print(len([experience for experience in self.memory.buffer if experience.label_correct]))
      print("number of examples with correct edge prediction in memory: ")
      print(len([experience for experience in self.memory.buffer if experience.edge_correct]))
    return epoch_tokens, epoch_reward



  def _replay(self, epoch):
    """Executes memory replay.

    During memory replay, we sample states, actions and rewards from memory. We re-run the states over
    the model to get the label scores. Then  we replace the argmax prediction value with the reward for
    that node."""
    states, target_q_scores, action_indexes, action_provenances = self._sample(strategy="targeted")
    enumerated_action_indexes = self._enumerated_tensor(action_indexes)
    updates = tf.squeeze(tf.transpose(target_q_scores))

    # action q scores are just label scores for a token.
    # target q scores are environment reward for the argmax of the label scores.
    # In action_q_scores (label_scores), write the value for target_q_scores to the index specified by
    # action_indexes.
    action_q_scores = tf.Variable(self.labeler.predict(states))
    action_q_scores.scatter_nd_update(indices=enumerated_action_indexes, updates=updates)
    target_q_scores = tf.convert_to_tensor(action_q_scores)
    # if epoch % 200 == 0:
    #   print("enumerated action indexes ", enumerated_action_indexes)
    #   print("updates ", updates)
    #   print("action indexes from sample", action_indexes)
    #   print("action provenances ", action_provenances)
    #   print("argmaxes ", tf.argmax(action_q_scores, axis=1))
    #   print("action q scores ", action_q_scores)
    #   print("target_q_scores ", target_q_scores)
    # input()
    # print("states shape ", states.shape)
    states = tf.expand_dims(states, 1)
    # print("states shape now ", states.shape)
    self.labeler.fit(states, target_q_scores, epochs=1)
    # input("press to cont.")


  def _sample(self, strategy: str):
    """Gathers data and updates the agent memory.

    Inputs here doesn't contain correct labels.
    """
    if strategy == "random":
      states, actions, rewards, actions_hot, action_names, action_provenances = self.memory.random_sample()
    elif strategy == "targeted":
      states, actions, rewards, actions_hot, action_names, action_provenances = self.memory.targeted_sample()
    # the target q scores are rewards. This is similar to Bellman equation but we don't compute target q_scores
    # based on the expected outcome of the next state, as the next state doesn't depend on the action I take in
    # the current state, it's just the next token in seqeuence. Therefore, the target_q_score is just the immediate
    # reward, which is computed based on whether the label and the parser predictions where correct or not.
    target_q_scores = tf.expand_dims(rewards, 1)
    # print("target_q_scores (as expansion of rewards): ", target_q_scores)
    # target_q_scores_masked = tf.multiply(target_q_scores, tf.cast(actions_hot, tf.int32))
    # action_q_scores_masked = tf.multiply(action_q_scores, actions_hot)
    # print("target q scores maksed ", target_q_scores_masked)
    # print("action q scores maksed ", action_q_scores_masked)
    return states, tf.cast(target_q_scores, tf.float32), actions, action_provenances

  def _make_experience(self, states,
                       pred_heads,  # shape: batch_size*seq_len, 1
                       true_heads,  # shape: batch_size*seq_len, 1
                       pred_label_indexes, # shape: batch_size*seq_len, 1
                       pred_label_hot, # shape: batch_size*seq_len, 43
                       true_labels, # shape: batch_size*seq_len, 1
                       pad_mask, # shape: batch_size*seq_len, 1
                       words, # shape: batch_size*seq_len, 1
                       action_provenances): # shape: batch_size*seq_len, 1
    """Creates an experience for memory replay buffer.

    Also computes total rewards and total records for episode."""
    # print("pred heads ", pred_heads)
    # print("pred labels ", pred_label_indexes)
    # print("action provenances ", action_provenances)
    # input("press")
    total_reward, total_tokens = 0, 0
    experiences = []
    seq_len = states.shape[0]
    for i in range(seq_len):
      if tf.keras.backend.get_value(pad_mask[i]) == True:
        continue
      pred_label_index = tf.keras.backend.get_value(pred_label_indexes[i])
      true_label_index = tf.keras.backend.get_value(true_labels[i])
      pred_label_name = self._label_index_to_name(pred_label_index)
      true_label_name = self._label_index_to_name(true_label_index)
      action_provenance = tf.keras.backend.get_value(action_provenances[i])
      # print(f"""pred heads: {pred_heads[i]}, "pred_heads": {pred_heads},
      #       true_heads: {true_heads[i]}, "true_heads": {true_heads},
      #       label_hot: {pred_label_hot[i]}, "label_hots": {pred_label_hot},
      #       label_preds: {pred_label_indexes[i]}, "label_preds: {pred_label_indexes},
      #       true_labels: {true_labels[i]}, "true_labels: {true_labels},
      #      pred_label_name: {pred_label_name},
      #       true_label_name: {true_label_name}""")
      reward, edge_correct, label_correct = self._compute_reward(
        pred_heads[i], true_heads[i], pred_label_index, true_label_index,
        pred_label_name
      )
      # print(f"reward {reward}, edge correct {edge_correct}, label correct {label_correct}")
      # print("i: ", i)
      # input()
      total_tokens += 1
      total_reward += reward

      experience = Experience(
        state=states[i],
        action=pred_label_index,
        reward=reward,
        action_hot=pred_label_hot[i],
        action_name=pred_label_name,
        true_label=true_labels[i].numpy()[0],
        true_label_name=true_label_name,
        word=self.word_embeddings.itos(idx=words[i].numpy()[0]),
        edge_correct=edge_correct,
        label_correct=label_correct,
        action_provenance=action_provenance
      )
      experiences.append(experience)
    return experiences, total_tokens, total_reward

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
      reward += 50
      if label_correct:
        reward += 100 # when both edge and label is correct, reward is 150.
      # else:
      #   reward -= 100 # when edge is correct but label is not, reward equals 0.
    else:
      if label_correct:
        reward += 100 # when edge is false but label correct, reward is 100.
      else:
        reward -= 100 # if both edge and label is false, reward -100.
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

  @staticmethod
  def _enumerated_tensor(_tensor):
    """Converts a 2D tensor to its enumerated version."""
    enumerated_tensors_list = []
    if not len(_tensor.shape) == 2:
      raise ValueError(f"enumerated tensor only works for 2D tensors. Received tensor of shape {_tensor.shape}")
    batch_size = _tensor.shape[0]
    for i in range(batch_size):
      # print("i ", i)
      # print("tensor[i]", _tensor[i])
      # input("press ")
      _t = tf.constant([i, tf.keras.backend.get_value(_tensor[i][0])])
      enumerated_tensors_list.append(_t)

    _enumerated_tensor = tf.convert_to_tensor(enumerated_tensors_list)
    # print("enumerated_tensor ", _enumerated_tensor)
    # input("press")
    return _enumerated_tensor

if __name__ == "__main__":
  agent = Agent(parser_path="./model/nn/pretrained")
  # agent._enumerated_tensor(tf.Variable([1, 30]))

  train_treebank="tr_boun-ud-test-random50.pbtxt"
  test_treebank = "tr_boun-ud-test-random50.pbtxt"

  train_sentences = agent.prep.prepare_sentence_protos(
    path=os.path.join(_DATA_DIR, train_treebank))
  test_sentences = agent.prep.prepare_sentence_protos(
    path=os.path.join(_TEST_DATA_DIR, test_treebank)
  )
  dataset = agent.prep.make_dataset_from_generator(
    sentences=train_sentences,
    batch_size=10)
  # test_dataset = agent.prep.make_dataset_from_generator(
  #   sentences=test_sentences,
  #   batch_size=20
  # )
  for epoch in range(10000):
    epoch_tokens, epoch_reward = agent.learn(dataset=dataset, epoch=epoch)
    performance_on_epoch = agent._episode_reward(epoch_tokens, epoch_reward)
    agent.epsilon_update(performance_on_epoch)
    if epoch % 20 == 0:
      print(f"********************* epoch {epoch} **************************")
    if epoch % 50 == 0:
      print("total tokens in epoch ", epoch_tokens)
      print("total rewards in epoch ", epoch_reward)
      print(f"performance on epoch {performance_on_epoch}: ")
