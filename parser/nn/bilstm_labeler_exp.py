
import os
import logging
import tensorflow as tf
import numpy as np
import time
import collections

from typing import Dict, Tuple
from parser.nn import base_parser
from proto import metrics_pb2
from input import embeddor
from tensorflow.keras import layers, metrics, losses, optimizers
from parser.nn import layer_utils
from util.nn import nn_utils
from tagset.dep_labels import dep_label_enum_pb2 as dep_label_tags
from util import writer

Embeddings = embeddor.Embeddings
Dataset = tf.data.Dataset

Dataset = tf.data.Dataset

class BiLSTMLabeler(base_parser.BaseParser):
  """A bi-lstm labeler that can be used for any kind of sequence labeling tasks."""
  @property
  def _optimizer(self):
    return tf.keras.optimizers.Adam(0.0001, beta_1=0.9, beta_2=0.9)


  def _training_metrics(self):
    return {
      "labels": metrics.SparseCategoricalAccuracy()
    }

  @property
  def _head_loss_function(self):
    """No head loss function for labeler."""
    return None

  @property
  def _label_loss_function(self):
    """Returns loss per token for label prediction.

    As we use the SparseCategoricalCrossentropy function, we expect the target labels to be
    to be provided as integers indexing the correct labels rather than one hot vectors. The predictions
    should be keeping the probs as float values for each label per token.

    For details, refer to:
    https://www.tensorflow.org/api_docs/python/tf/keras/losses/SparseCategoricalCrossentropy"""

    return losses.SparseCategoricalCrossentropy(from_logits=True,
                                                reduction=tf.keras.losses.Reduction.AUTO)

  @property
  def inputs(self):
    word_inputs = tf.keras.Input(shape=(None, ), name="words")
    pos_inputs = tf.keras.Input(shape=(None, ), name="pos")
    morph_inputs = tf.keras.Input(shape=(None, 56), name="morph")
    input_dict = {"words": word_inputs}
    if self._use_pos:
      input_dict["pos"] = pos_inputs
    if self._use_morph:
      input_dict["morph"] = morph_inputs
    return input_dict

  def _n_words_in_batch(self, words, pad_mask=None):
    words_reshaped = tf.reshape(words, shape=pad_mask.shape)
    return len(tf.boolean_mask(words_reshaped, pad_mask))

  def _parsing_model(self, model_name):
    super()._parsing_model(model_name)
    print(f"""Using features pos: {self._use_pos}, morph: {self._use_morph}""")
    model = LSTMLabelingModel(
      n_output_classes=8,
      word_embeddings=self.word_embeddings,
      name=model_name,
      use_pos=self._use_pos,
      use_morph=self._use_morph,
      return_lstm_output=True,
    )
    model(inputs=self.inputs)
    return model

  @property
  def _coarse_dep(self):
    return {
      0: 0, 2: 1, 7: 1, 8: 1, 18: 1, 22: 1, 35: 1, 40: 1,
      12: 2, 19: 2, 27: 2, 32: 2, 34: 2, 41: 2,
      1: 3, 3: 3, 4: 3, 5: 3, 6: 3, 9: 3, 13: 3, 14: 3, 15: 3, 16: 3, 20:3, 21:3,
      22:3, 23:3, 24:3, 25:3, 26:3, 28:3, 29:3, 30:3, 31:3, 33:3, 36:3, 38:3,
      39: 4,
      10: 5, 11: 5,
      17: 6, 37: 6, 42: 7,
    }

  def train_step(self, *,
                 words: tf.Tensor, pos: tf.Tensor, morph: tf.Tensor,
                 dep_labels: tf.Tensor, heads: tf.Tensor) -> Tuple[tf.Tensor, ...]:
    """Runs one training step.
    Args:
        words: tf.Tensor of word indices of shape (batch_size, seq_len) where the seq_len
          is padded with 0s on the right.
        pos: tf.Tensor of pos indices of shape (batch_size, seq_len), of the same shape
          as words.
        morph: tf.Tensor of shape (batch_size, seq_len, n_morph)
        heads: tf.Tensor of shape (batch_size, seq_len) holding correct heads.
        dep_labels: tf.Tensor of shape (batch_size, seq_len), holding correct labels.
    Returns:
      losses: dictionary holding loss values for head and labels.
        label_loss: tf.Tensor of (batch_size*seq_len, 1)
      correct: dictionary holding correct values for heads and labels.
        labels: tf.Tensor of (batch_size*seq_len, 1)
      predictions: dictionary holding correct values for heads and labels.
        labels: tf.Tensor of (batch_size*seq_len, 1)
      pad_mask: tf.Tensor of shape (batch_size*seq_len, 1) where padded words are marked as 0.
    """

    converted_labels = []
    for vector in dep_labels:
      # print("vector ", vector)
      converted = [self._coarse_dep[tf.keras.backend.get_value(label)] for label in vector]
      #print("converted ", converted)
      converted_labels.append(tf.expand_dims(converted, 0))
    converted_dep_labels = tf.concat(converted_labels, axis=0)
    # print("converted dep labels ", converted_dep_labels)
    # input()
    predictions, correct, losses = {}, {}, {}
    with tf.GradientTape() as tape:
      label_scores = self.model({"words": words, "pos": pos, "morph": morph,
                                 "labels": converted_dep_labels}, training=True)
      # print("label scores ", label_scores)
      # input()
      # print("concat ", lstm_output)
      # input("press to continue.")
      pad_mask = self._flatten((words != 0))
      # Get the predicted label indices from the label scores, tensor of shape (batch_size*seq_len, 1)
      label_preds = tf.argmax(label_scores, axis=2)
      label_preds = self._flatten(label_preds)
      # print("label preds ", label_preds)
      # input()
      # Flatten the label scores to (batch_size*seq_len, n_classes) (i.e. 340, 36).
      label_scores = self._flatten(label_scores, outer_dim=label_scores.shape[2])
      # Flatten the correct labels to the shape (batch_size*seq_len, 1) (i.e. 340,1)
      # Index for the right label for each token.
      correct_labels = tf.cast(self._flatten(converted_dep_labels), tf.int64)
      # print("correct labels ", correct_labels)
      # input()
      label_loss = tf.expand_dims(self._label_loss(label_scores, correct_labels), axis=-1)
      grads = tape.gradient(label_loss, self.model.trainable_weights)

    self._optimizer.apply_gradients(zip(grads, self.model.trainable_weights))

    # Update training metrics.
    self._update_training_metrics(
      labels=correct_labels,
      label_scores=label_scores,
      pad_mask=pad_mask)

    losses["labels"] = label_loss
    correct["labels"] = correct_labels
    predictions["labels"] = label_preds

    return predictions, losses, correct, pad_mask


  def test(self, *, dataset: Dataset):
    """Tests the performance of this parser on some dataset."""
    print("Testing on the test set..")
    label_accuracy = tf.keras.metrics.Accuracy()

    # resetting test stats at the beginning.
    for key in self.test_stats:
      self.test_stats[key] = 0.0

    # We traverse the test dataset not batch by batch, but example by example.
    for example in dataset:
      dep_labels = example["dep_labels"]
      # print("dep_labels ", dep_labels)
      converted_labels = [self._coarse_dep[tf.keras.backend.get_value(label)] for label in dep_labels[0]]
      # print("converted ", converted_labels)
      converted_dep_labels = tf.cast(tf.expand_dims(tf.convert_to_tensor(converted_labels), 0), tf.int64)
      # print("converted dep labels ", converted_dep_labels)
      label_scores = self.parse(example["words"], example["pos"], example["morph"], converted_dep_labels)
      label_preds = self._flatten(tf.argmax(label_scores, 2))
      # print("label_preds ", label_preds)
      correct_labels = self._flatten(converted_dep_labels)
      label_accuracy.update_state(correct_labels, label_preds)
      # input()

      correct_predictions_dict = self._correct_predictions(
        label_predictions=label_preds,
        correct_labels=correct_labels,
      )
      self._update_correct_prediction_stats(correct_predictions_dict,
                                            example["words"].shape[1],
                                            stats="test")

    logging.info(f"Test stats: {self.test_stats}")
    test_results = self._compute_metrics(stats="test")
    return test_results

  def parse(self, words, pos, morph, dep_labels):
    """Parse an example with this parser.

    Args:
      example: A single example that holds features in a dictionary.
        words: Tensor representing word embedding indices of words in the sentence.
        pos: Tensor representing pos embedding indices of pos in the sentence.
        morph: Tensor representing morph indices of the morphological features in words in the sentence.

    Returns:
      scores: a dictionary of scores representing edge and label predictions.
        labels: Tensor of shape (1, seq_len, n_labels)
      lstm_output: (Optional) Tensor representing the output from the LSTM layers (before dense application)
        for each token.
    """
    scores  = self.model({"words": words, "pos": pos,
                          "morph": morph, "labels": dep_labels}, training=False)
    return scores


class LSTMLabelingModel(tf.keras.Model):
  """A standalone bidirectional LSTM labeler."""
  def __init__(self, *,
               word_embeddings: Embeddings,
               n_units: int = 256,
               n_output_classes: int,
               use_pos=True,
               use_morph=True,
               return_lstm_output=True,
               name="LSTM_Labeler"):
    super(LSTMLabelingModel, self).__init__(name=name)
    self.use_pos = use_pos
    self.use_morph = use_morph
    self.word_embeddings = layer_utils.EmbeddingLayer(
      pretrained=word_embeddings, name="word_embeddings"
    )
    self.return_lstm_output = return_lstm_output
    if self.use_pos:
      self.pos_embeddings = layer_utils.EmbeddingLayer(
        input_dim=37, output_dim=32,
        name="pos_embeddings",
        trainable=True)
    self.concatenate = layers.Concatenate(name="concat")
    self.lstm_block = layer_utils.LSTMBlock(n_units=n_units,
                                            dropout_rate=0.3,
                                            name="lstm_block"
                                            )
    # Because in the loss function we have from_logits=True, we don't use the
    # param 'activation=softmax' in the layer. The loss function applies softmax to the
    # raw probabilites and then applies crossentropy.
    self.labels = layers.Dense(units=n_output_classes, name="labels")

  def call(self, inputs, training=True):
    """Forward pass.
    Args:
      inputs: Dict[str, tf.keras.Input]. This consist of
        words: Tensor of shape (batch_size, seq_len)
        pos: Tensor of shape (batch_size, seq_len)
        morph: Tensor of shape (batch_size, seq_len, 66)
      The boolean values set up during the initiation of the model determines
      which one of these features to use or not.
    Returns:
      A dict which contains:
        label_scores: [batch_size, seq_len, n_labels] label preds for tokens (i.e. 10, 34, 36)
    """
    word_inputs = inputs["words"]
    word_features = self.word_embeddings(word_inputs)
    concat_list = [word_features]
    if self.use_pos:
      pos_inputs = inputs["pos"]
      pos_features = self.pos_embeddings(pos_inputs)
      concat_list.append(pos_features)
    if self.use_morph:
      morph_inputs = inputs["morph"]
      concat_list.append(morph_inputs)
    if len(concat_list) > 1:
      concat = self.concatenate(concat_list)
      sentence_repr = self.lstm_block(concat, training=training)
      labels = self.labels(sentence_repr)
    else:
      sentence_repr = self.lstm_block(word_features, training=training)
      labels = self.labels(sentence_repr)

    return labels


class LSTMBlock(layers.Layer):
  """A bidirectional LSTM block with 3 Birectional LSTM layers"""
  def __init__(self, *, n_units: int,
               return_sequences: bool = True,
               return_state: bool = False,
               dropout_rate: float = 0.0,
               name="LSTMBlock"):
    super(LSTMBlock, self).__init__(name=name)
    print("Setting up LSTM block with dropout rate ", dropout_rate)
    self.dropout_rate = dropout_rate
    self.lstm1 = layers.Bidirectional(layers.LSTM(
      units=n_units, return_sequences=return_sequences,
      # stateful=True,
      name="lstm1"))
    self.lstm2 = layers.Bidirectional(layers.LSTM(
      units=n_units, return_sequences=return_sequences,
      # stateful=True,
      name="lstm2"))
    self.lstm3 = layers.Bidirectional(layers.LSTM(
      units=n_units, return_sequences=return_sequences,
      return_state=return_state,
      # stateful=True,
      name="lstm3"))
    self.dropout1 = layers.Dropout(rate=dropout_rate, name="dropout1")
    self.dropout2 = layers.Dropout(rate=dropout_rate, name="dropout2")

  def call(self, input_tensor, training=True):
    dropout = self.dropout_rate > 0 and training
    # print("dropout is ", dropout)
    if dropout:
      out = self.lstm1(input_tensor)
      out = self.dropout1(out)
      out = self.lstm2(out)
      out = self.dropout2(out)
      out = self.lstm3(out)
    else:
      out = self.lstm1(input_tensor)
      out = self.lstm2(out)
      out = self.lstm3(out)
    return out
