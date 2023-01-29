
import tensorflow as tf
from parser import base_parser
from parser.utils import architectures, layer_utils
from proto import metrics_pb2
from input import embeddor
from tensorflow.keras import layers, metrics, losses, optimizers

import logging
logging.basicConfig(format='%(levelname)s : %(message)s', level=logging.INFO)

from typing import List
Embeddings = embeddor.Embeddings


class LabelFirstParser(base_parser.BaseParser):
  """A label first parser predicts labels before predicting heads."""

  @property
  def _optimizer(self):
    return tf.keras.optimizers.Adam(0.001, beta_1=0.9, beta_2=0.9)

  def _training_metrics(self):
    return {
      "heads": metrics.SparseCategoricalAccuracy(),
      "labels": metrics.SparseCategoricalAccuracy()
    }

  @property
  def _head_loss_function(self):
    """Returns loss per token for head prediction.
    As we use the SparseCategoricalCrossentropy function, we expect the target labels
    to be provided as integers indexing the correct labels rather than one hot vectors.
    For details, refer to:
    https://www.tensorflow.org/api_docs/python/tf/keras/losses/SparseCategoricalCrossentropy
    """
    return losses.SparseCategoricalCrossentropy(from_logits=True,
                                                reduction=tf.keras.losses.Reduction.NONE)

  @property
  def _label_loss_function(self):
    """Returns loss per token for label prediction.

    As we use the SparseCategoricalCrossentropy function, we expect the target labels to be
    to be provided as integers indexing the correct labels rather than one hot vectors. The predictions
    should be keeping the probs as float values for each label per token.

    For details, refer to:
    https://www.tensorflow.org/api_docs/python/tf/keras/losses/SparseCategoricalCrossentropy"""

    return losses.SparseCategoricalCrossentropy(from_logits=True,
                                                reduction=tf.keras.losses.Reduction.NONE)

  @property
  def inputs(self):
    word_inputs = tf.keras.Input(shape=(None,), name="words")
    pos_inputs = tf.keras.Input(shape=(None,), name="pos")
    morph_inputs = tf.keras.Input(shape=(None, 56), name="morph")
    if not self.one_hot_labels:
      label_inputs = tf.keras.Input(shape=(None, ), name="labels")
    else:
      label_inputs = tf.keras.Input(shape=(None, 43), name="labels")
    input_dict = {"words": word_inputs}
    if self._use_pos:
      input_dict["pos"] = pos_inputs
    if self._use_morph:
      input_dict["morph"] = morph_inputs
    if self._use_dep_labels:
      input_dict["labels"] = label_inputs
    return input_dict

  def _n_words_in_batch(self, words, pad_mask=None):
    words_reshaped = tf.reshape(words, shape=pad_mask.shape)
    return len(tf.boolean_mask(words_reshaped, pad_mask))

  def _parsing_model(self, model_name):
    super()._parsing_model(model_name)
    print(f"""Using features pos: {self._use_pos}, morph: {self._use_morph},
              dep_labels: {self._use_dep_labels}""")
    model = LabelFirstParsingModel(
      n_dep_labels=self._n_output_classes,
      word_embeddings=self.word_embeddings,
      predict=self._predict,
      name=model_name,
      use_pos=self._use_pos,
      use_morph=self._use_morph,
      use_dep_labels=self._use_dep_labels,
      pos_embedding_vocab_size=self.pos_embedding_vocab_size,
      one_hot_labels=self.one_hot_labels,
    )
    model(inputs=self.inputs)
    return model

class LabelFirstParsingModel(tf.keras.Model):
  """Label first parsing model predicts labels before edges."""
  def __init__(self, *,
               n_dep_labels: int,
               word_embeddings: Embeddings,
               name="Label_First_Parsing_Model",
               predict: List[str],
               use_pos:bool = True,
               use_morph:bool=True,
               use_dep_labels:bool=False,
               pos_embedding_vocab_size=37,
               one_hot_labels: False
               ):
    super(LabelFirstParsingModel, self).__init__(name=name)
    self.predict = predict
    self.use_pos = use_pos
    self.use_morph = use_morph
    self.use_dep_labels = use_dep_labels
    self.one_hot_labels = one_hot_labels
    self._null_label = tf.constant(0)

    assert(not("labels" in self.predict and self.use_dep_labels)), "Can't use dep_labels both as feature and label!"
    logging.info(f"Using dep labels as feature: {self.use_dep_labels}")

    self.word_embeddings = layer_utils.EmbeddingLayer(
      pretrained=word_embeddings,
      name="word_embeddings",
      trainable=False)

    if self.use_pos:
      self.pos_embeddings = layer_utils.EmbeddingLayer(
        input_dim=pos_embedding_vocab_size, output_dim=32,
        name="pos_embeddings",
        trainable=True)

    self.concatenate = layers.Concatenate(name="concat")
    self.lstm_block = LSTMBlock(n_units=256,
                                dropout_rate=0.3,
                                name="lstm_block")
   #  self.attention = layer_utils.Attention()

    if "labels" in self.predict:
      self.dep_labels = layers.Dense(units=n_dep_labels, name="labels")
    else:
      if self.use_dep_labels and not self.one_hot_labels:
        # Turkish
        self.label_embeddings = layer_utils.EmbeddingLayer(input_dim=43,
                                                           output_dim=50,
                                                           name="label_embeddings",
                                                           trainable=False)
        # English
        # self.label_embeddings = layer_utils.EmbeddingLayer(input_dim=51,
        #                                                   output_dim=100,
        #                                                   name="label_embeddings",
        #                                                   trainable=True)


    self.head_perceptron = layer_utils.Perceptron(n_units=256,
                                                  activation="relu",
                                                  name="head_mlp")
    self.dep_perceptron = layer_utils.Perceptron(n_units=256,
                                                 activation="relu",
                                                 name="dep_mlp")
    self.edge_scorer = layer_utils.EdgeScorer(n_units=256, name="edge_scorer")
    logging.info((f"Set up {name} to predict {predict}"))


  def call(self, inputs, training=True): # inputs = Dict[str, tf.keras.Input]
    """Forward pass.
    Args:
      inputs: Dict[str, tf.keras.Input]. This consist of
        words: Tensor of shape (batch_size, seq_len)
        pos: Tensor of shape (batch_size, seq_len)
        morph: Tensor of shape (batch_size, seq_len, 66)
        dep_labels: Tensor of shape (batch_size, seq_len)
      The boolean values set up during the initiation of the model determines
      which one of these features to use or not.
    Returns:
      A dict which conteins:
        edge_scores: [batch_size, seq_len, seq_len] head preds for all tokens (i.e. 10, 34, 34)
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
    if self.use_dep_labels:
      label_inputs = inputs["labels"]
      if not self.one_hot_labels:
        label_features = self.label_embeddings(label_inputs)
        # print("label features ", label_features)
        # input()
        concat_list.append(label_features)
      else:
        # print("label inputs ", label_inputs)
        # input()
        concat_list.append(label_inputs)
    if len(concat_list) > 1:
      sentence_repr = self.concatenate(concat_list)
      # print("sentence repr ", sentence_repr)
      # input()
    else:
      sentence_repr = word_features

    sentence_repr = self.lstm_block(sentence_repr, training=training)
    # print("sentence repr ", sentence_repr)
    # sentence_repr = self.attention(sentence_repr)
    if "labels" in self.predict:
      dep_labels = self.dep_labels(sentence_repr)
      h_arc_head = self.head_perceptron(dep_labels)
      h_arc_dep = self.dep_perceptron(dep_labels)
      edge_scores = self.edge_scorer(h_arc_head, h_arc_dep)
      return {"edges": edge_scores,  "labels": dep_labels}
    else:
      h_arc_head = self.head_perceptron(sentence_repr)
      h_arc_dep = self.dep_perceptron(sentence_repr)
      edge_scores = self.edge_scorer(h_arc_head, h_arc_dep)
      return {"edges": edge_scores,  "labels": self._null_label}


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
    # self.dropout3 = layers.Dropout(rate=dropout_rate, name="dropout3")

  def call(self, input_tensor, training=True):
    dropout = self.dropout_rate > 0 and training
    # print("dropout is ", dropout)
    if dropout:
      out = self.lstm1(input_tensor)
      out = self.dropout1(out)
      out = self.lstm2(out)
      out = self.dropout2(out)
      out = self.lstm3(out)
      # out = self.dropout3(out)
    else:
      out = self.lstm1(input_tensor)
      out = self.lstm2(out)
      out = self.lstm3(out)
    return out