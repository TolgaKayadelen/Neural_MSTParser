
import collections
import dataclasses
import logging

import tensorflow as tf
import numpy as np
# import tensorflow_addons as tfa
import tensorflow.keras.backend as K

from input import embeddor
from proto import metrics_pb2
from typing import List, Dict, Tuple

from parser.nn import layer_utils
from tagset.dep_labels import dep_label_enum_pb2 as dep_label_tags
from tensorflow.keras import layers, metrics, losses, optimizers, initializers
from typing import Dict, Tuple

Embeddings = embeddor.Embeddings
Dataset = tf.data.Dataset

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


def softmax(x, axis=1):
  """Softmax activation function.
  # Arguments
      x : Tensor.
      axis: Integer, axis along which the softmax normalization is applied.
  # Returns
      Tensor, output of softmax transformation.
  # Raises
      ValueError: In case `dim(x) == 1`.
  """
  ndim = K.ndim(x)
  if ndim == 2:
    return K.softmax(x)
  elif ndim > 2:
    e = K.exp(x - K.max(x, axis=axis, keepdims=True))
    s = K.sum(e, axis=axis, keepdims=True)
    return e / s
  else:
    raise ValueError('Cannot apply softmax to a tensor that is 1D')


def compute_reward(predicted_head, true_head, predicted_label, true_label, predicted_label_name):
  # print("pr_head ", predicted_head)
  # print("tr head ", true_head)
  # print("pr label ", predicted_label)
  # print("tr label ", true_label)
  # print("pr label name ", predicted_label_name)
  # input("press")
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
      reward += 1
    else:
      reward -= 1
  else:
    if label_correct:
      reward += 1
    else:
      reward -= 1
  # print("reward is ", reward)
  # input("press to cont.")
  return reward, edge_correct, label_correct

def _make_experience(states, head_probs, true_heads, predicted_labels, label_scores, predicted_labels_hot, true_labels):
  """Creates an experience for memory replay buffer.
  Args:
    states = shape (batch_size, 512)
    head_probs = shape (batch_size, seq_len)
    true_heads = indices of true heads. shape = batch_size, 1
    predicted labels = indices of predicted labels. shape = batch_size, 1
    predicted_labels_hot = one hot representation of predicted labels. shape = batch_size, 1, n_labels.
    true_labels = one hot representaton of true labels. shape = batch_size, n_labels.
  """
  # print("states ", states)
  # input("press to cont.")
  experiences = []
  predicted_heads = tf.expand_dims(tf.argmax(head_probs, 1), 1)
  true_labels = tf.argmax(true_labels, 1)
  predicted_label_names = [dep_label_tags.Tag.Name(val.numpy()) for val in predicted_labels]
  true_label_names = [dep_label_tags.Tag.Name(val.numpy()) for val in true_labels]
  # print("true heads ", true_heads)
  # print("predicted heads ", predicted_heads)
  # print("true labels ", true_labels)
  # print("predicted labels ", predicted_labels)
  # print("predicted_label_names", predicted_label_names)
  # print("true label names ", true_label_names)
  # print("label scores ", label_scores)
  # input("press ")
  batch_size = states.shape[0]
  for i in range(batch_size):
    reward, edge_correct, label_correct = compute_reward(predicted_heads[i], true_heads[i],
                                                         predicted_labels[i], true_labels[i],
                                                         predicted_label_names[i])
    experience = Experience(
      state=states[i],
      action=predicted_labels[i],
      action_qs=label_scores[i],
      reward=reward,
      action_hot=predicted_labels_hot[i],
      action_name=predicted_label_names[i],
      true_label=true_labels[i],
      edge_correct=edge_correct,
      label_correct=label_correct
    )
    # print("experience ", experience)
    experiences.append(experience)
    # input("press to cont.")
  return experiences

class CombinedParsingModel(tf.keras.Model):
  "The combined sequential model does labeling and parsing per token."

  def __init__(self, *,
               word_embeddings: Embeddings,
               n_lstm_units: int = 10,
               n_output_classes: int,
               use_pos = True,
               use_morph = True,
               name = "combined_sequantial_with_attention"):

    super(CombinedParsingModel, self).__init__(name=name)

    # which features to use
    self.use_pos = use_pos
    self.use_morph = use_morph

    # initialize word embeddings
    self.word_embeddings = word_embeddings

    # number of output label classes
    self.n_output_classes = n_output_classes
    self.label_scores = {}

    # Define some variables that will be useful in setting up lstm layers.
    # Define some variables that will be useful in setting up lstm layers.
    self.pre_attn_lstm_units = n_lstm_units
    self.pos_attn_lstm_units = n_lstm_units*2

    self.head_loss_function = losses.SparseCategoricalCrossentropy(from_logits=True)

    # Pre attention embedding layers.
    self.word_embeddings_layer = layer_utils.EmbeddingLayer(
      pretrained=word_embeddings, name="word_embeddings_layer"
    )
    if self.use_pos:
      self.pos_embeddings = layer_utils.EmbeddingLayer(
        input_dim=37, output_dim=32,
        name="pos_embeddings_layer", trainable=True
      )

    # Concat layer to concatenate multiple features.
    self.concatenate = layers.Concatenate(name="concat_layer")

    # Pre attention bidirectional LSTM layer. This is the encoder block.
    self.pre_attn_lstm = layer_utils.LSTMBlock(
      n_units=n_lstm_units,
      num_layers=2,
      dropout_rate=0.3,
      name="lstm_block")

    # Attention layers and inputs
    self.s0 = layers.Input(self.pos_attn_lstm_units, name="s0")
    self.c0 = layers.Input(self.pos_attn_lstm_units, name="c0")

    self.attn_repeator = layers.RepeatVector(1, name="repeat_vector_layer")
    self.attn_concatenator = layers.Concatenate(axis=-1, name="attn_concat_layer")
    self.attn_densor1 = layers.Dense(50,
                                     activation = "tanh",
                                     name="attn_dense1")
    self.attn_densor2 = layers.Dense(10,
                                     activation = "tanh",
                                     name="attn_dense2")
    self.attn_densor3 = layers.Dense(1,
                                     activation = "relu",
                                     name="attn_dense3")
    self.attn_activator = layers.Activation(softmax, name = "attention_activation")
    self.attn_dotor = layers.Dot(axes = 1, name="attn_dot")

    # Post attention layer -- this will only be used for producing labeler output.
    self.post_attn_lstm = layers.LSTM(units=self.pos_attn_lstm_units,
                                      return_state=True,
                                      name="post_attn_lstm")

    # Label output layer -- produces label prediction
    # Don't use activation=softmax here because we would like to predict them as raw q values
    # based on the reward.
    self.label_output_layer = layers.Dense(units=n_output_classes,
                                           name="label_predictions")

    # Head output layers -- produces the head prediction.
    self.u_a = layers.Dense(self.pos_attn_lstm_units, activation=None, name="u_a_head_perceptron")
    self.w_a = layers.Dense(self.pos_attn_lstm_units, activation=None, name="w_a_dep_perceptron")
    self.v_a_inv = layers.Dense(1, activation=None, use_bias=False,
                                name="v_a_inv_dense")



  def _one_step_attention(self, a, s_prev):
    """
    Performs one step attention. Outputs a context vector as a dot
    product of the attention weights 'alphas' and the hidden states 'a'
    of the pre-attention BiLSTM.

    Args:
      a: hidden state output of the pre-attention BiLSTM, of shape (m, seq_len, n_dim)
      s_prev: previous hidden state of the post-attention LSTM, np array of shape
      (m, n_dim).

    Returns:
      context: the context vector, input of the post attention LSTM cell
    """
    assert(self.attn_repeator.n == a.shape[1])
    # print("s prev is of shape ", s_prev)

    # Use the repeator to make s_prev of shape (m, seq_len, n_dim).
    s_prev = self.attn_repeator(s_prev)
    # print("s prev is now of shape ", s_prev)

    # Concatenate s_prev with the pre-attention bilstm hidden state output.
    # Concat is of shape: (m, seq_len, n_dim*2)
    concat = self.attn_concatenator([a, s_prev])
    # print("concat is of shape ", concat.shape)
    # input("press to cont.")

    # Propagate concat through a small fully connected NN to compute energies.
    # Energies is of shape (m, seq_len, 1)
    e = self.attn_densor1(concat)
    e_prime = self.attn_densor2(e)
    energies = self.attn_densor3(e_prime)
    # print("energies is of shape ", energies)
    # input("press to cont.")

    # Compute alphas: alphas is the attention weights; the weight the prediction y
    # needs to pay to each timestep of a. This is just softmax activation over
    # energies. Alphas is of shape: m, seq_len, 1 (same shape as energies).
    alphas = self.attn_activator(energies)

    # print("alphas are ", alphas)
    # input("press to cont.")

    # Use dotor over alphas (attention weights) and a (hidden state
    # of the pre-attention BiLSTM) to compute the context vector; the input to
    # the post attention LSTM.
    context = self.attn_dotor([alphas, a])
    # print("context ", context)
    # input("press to cont.")

    return context

  def call(self, inputs):
    """Call function."""
    label_outputs = []
    experiences = []
    head_loss = 0.0
    sent_ids = inputs["sent_ids"][:, :1]
    print("sent ids ", sent_ids)
    # input("press to cont.")
    word_inputs, pos_inputs, morph_inputs = inputs["words"], inputs["pos"], inputs["morph"]
    word_features = self.word_embeddings_layer(word_inputs)
    pos_features = self.pos_embeddings(pos_inputs)

    pad_mask = (word_inputs == 0)
    batch_size, seq_len, _ = word_features.shape
    self.attn_repeator.n = seq_len

    s0 = layers.Input(shape=(self.pos_attn_lstm_units), name="s0")
    s0 = tf.zeros((batch_size, self.pos_attn_lstm_units))
    c0 = layers.Input(shape=(self.pos_attn_lstm_units), name="c0")
    c0 = tf.zeros((batch_size, self.pos_attn_lstm_units))

    concat = self.concatenate([word_features, pos_features, morph_inputs])

    # Pass inputs from pre attention lstm
    sentence_repr = self.pre_attn_lstm(concat)

    s, c = s0, c0
    parent_prob_dict = collections.defaultdict(list)

    for t in range(seq_len):
      context = self._one_step_attention(sentence_repr, s)
      s, _, c = self.post_attn_lstm(inputs=context,
                                    initial_state=[s,c])
      # print("s is ", s)
      # print("c is ", c)
      word_slice = word_inputs[:, t]
      # print("word slice ", word_slice)
      # print("words ", [self.word_embeddings.itos(idx=w.numpy()) for w in word_slice])
      label_output = self.label_output_layer(s)
      label_outputs.append(label_output)
      # print("label output ", label_output)
      # input("press to cont.")
      label_output_indices = tf.argmax(label_output, axis=1)

      # turn label output to one_hot representation.
      label_pred_oh = tf.one_hot(label_output_indices, self.n_output_classes)

      # for concatenation purposes, we expand dims here.
      label_pred_oh = tf.expand_dims(label_pred_oh, 1)
      # print("one hot labels ", label_pred_oh)
      # input("pres to cont.")
      # We skip the first token for dep parsing purposes.
      if t < 1:
        continue
      dependant_slice = tf.expand_dims(sentence_repr[:, t, :], 1)
      dependant_slice = tf.concat([dependant_slice, label_pred_oh], axis=2)
      # print("depdendent slice ", dependant_slice)
      # input("press to cont")
      tile = tf.constant([1, seq_len, 1])
      dependant = tf.tile(dependant_slice, tile)
      # print("dependant is ", dependant)
      # input("press to cont.")
      temp = np.zeros(shape=[batch_size, seq_len, 1], dtype=bool)
      temp[:, t] = True
      # print("temp is ", temp)
      head_mask = tf.convert_to_tensor(temp)
      # print("head mask ", head_mask)
      sentence_repr_concat = tf.concat([sentence_repr,
                                        tf.zeros(shape=[batch_size, seq_len, self.n_output_classes])],
                                       axis=2)
      # input("pres to cont.")
      head_probs = self.v_a_inv(
        tf.nn.tanh(
          self.u_a(sentence_repr_concat) + self.w_a(dependant))
      )
      # Apply 0 to the case where the candidate head is the token itself.
      head_probs = tf.squeeze(tf.where(head_mask, -1e4, head_probs), -1)
      # Also apply 0 to the padded tokens
      if batch_size > 1:
        head_probs = tf.where(pad_mask, -1e4, head_probs)
        # print("head probs after pad mask ", head_probs)

      # expand dims of true_heads to be able to compute loss between true_heads and head_probs.
      true_heads = tf.expand_dims(inputs["heads"][:, t], 1)

      # get the true labels.
      true_labels = inputs["labels"][:, t]

      # compute experiences.
      experiences.extend(_make_experience(
        s, head_probs, true_heads, label_output_indices, label_output, label_pred_oh, true_labels))
      # print(experiences)
      # input("press to cont.")

      head_loss += self.head_loss_function(true_heads, head_probs)
      # print("head loss ", head_loss)
      # input("pres to cont.")

      if batch_size > 1:
        for sent_id, token in zip(sent_ids, head_probs):
          parent_prob_dict[sent_id.numpy()[0]].append(tf.expand_dims(tf.math.exp(token), 0))
      else:
        sent_id = sent_ids[0].numpy()[0]
        parent_prob_dict[sent_id].append(tf.math.exp(head_probs))


    # END of FOR LOOP
    parent_dict = {}
    for key in parent_prob_dict.keys():
      parent_dict[key] = tf.concat(parent_prob_dict[key], 0)
    # print("parent prob dict ", parent_dict)
    label_scores = tf.reshape(tf.concat(label_outputs, axis=1),
                              shape=(batch_size, seq_len, self.n_output_classes))
    return head_loss, parent_dict, experiences, label_scores
    # return self.scores