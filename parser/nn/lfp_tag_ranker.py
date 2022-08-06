
import datetime
import logging

import tensorflow as tf
import numpy as np
from parser.nn import base_parser, architectures
from proto import metrics_pb2
from input import embeddor
from tensorflow.keras import layers, metrics, losses, optimizers
from parser.nn import layer_utils
from parser.nn import load_models
from tagset.dep_labels import dep_label_enum_pb2 as dep_label_tags
from util import writer


logging.basicConfig(format='%(levelname)s : %(message)s', level=logging.INFO)

from typing import List, Tuple
Embeddings = embeddor.Embeddings

def _label_index_to_name(label_index):
  return dep_label_tags.Tag.Name(label_index)

def _label_name_to_index(label_name):
  return dep_label_tags.Tag.Value(label_name)

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

class LFPTagRanker(base_parser.BaseParser):
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
    label_inputs = tf.keras.Input(shape=(None,), name="labels")
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
      use_dep_labels=self._use_dep_labels
    )
    # model(inputs=self.inputs)
    return model

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
        head_loss: tf.Tensor of (batch_size*seq_len, 1)
        label_loss: tf.Tensor of (batch_size*seq_len, 1)
      correct: dictionary holding correct values for heads and labels.
        heads: tf.Tensor of (batch_size*seq_len, 1)
        labels: tf.Tensor of (batch_size*seq_len, 1)
      predictions: dictionary holding correct values for heads and labels.
        heads: tf.Tensor of (batch_size*seq_len, 1)
        labels: tf.Tensor of (batch_size*seq_len, 1)
      pad_mask: tf.Tensor of shape (batch_size*seq_len, 1) where padded words are marked as 0.
    """
    predictions, correct, losses = {}, {}, {}
    label_embeddings = self.model.label_embeddings.get_weights()
    # print("label embeddings ", label_embeddings)
    with tf.GradientTape() as tape:

      # Head scores = (batch_size, seq_len, seq_len), Label scores = (batch_size, seq_len, n_labels)

      scores = self.model({"words": words, "pos": pos, "morph": morph,
                           "labels": dep_labels}, label_embeddings=label_embeddings, training=True)

      head_scores, label_scores = scores["edges"], scores["labels"]

      pad_mask = self._flatten((words != 0))

      if "heads" in self._predict:
        # Get the predicted head indices from the head scores, tensor of shape (batch_size*seq_len, 1)
        head_preds = self._flatten(tf.argmax(head_scores, axis=2))

        # Flatten the head scores to (batch_size*seq_len, seq_len) (i.e. 340, 34).
        # Holds probs for each token's head prediction.
        head_scores = self._flatten(head_scores, outer_dim=head_scores.shape[2])

        # Flatten the correct heads to the shape (batch_size*seq_len, 1) (i.e. 340,1)
        # Index for the right head for each token.
        correct_heads = self._flatten(heads)

        # Compute loss
        head_loss = tf.expand_dims(self._head_loss(head_scores, correct_heads), axis=-1)

      if "labels" in self._predict:

        # Get the predicted label indices from the label scores, tensor of shape (batch_size*seq_len, 1)
        label_preds = self._flatten(tf.argmax(label_scores, axis=2))

        # Flatten the label scores to (batch_size*seq_len, n_classes) (i.e. 340, 36).
        label_scores = self._flatten(label_scores, outer_dim=label_scores.shape[2])

        # Flatten the correct labels to the shape (batch_size*seq_len, 1) (i.e. 340,1)
        # Index for the right label for each token.
        correct_labels = self._flatten(dep_labels)

        label_loss = tf.expand_dims(self._label_loss(label_scores, correct_labels), axis=-1)

    # Compute gradients.
    if "heads" in  self._predict and "labels" in self._predict:
      grads = tape.gradient([head_loss, label_loss], self.model.trainable_weights)
    elif "heads" in self._predict:
      grads = tape.gradient(head_loss, self.model.trainable_weights)
    elif "labels" in self._predict:
      grads = tape.gradient(label_loss, self.model.trainable_weights)
    else:
      raise ValueError("No loss value to compute gradient for.")

    self._optimizer.apply_gradients(zip(grads, self.model.trainable_weights))

    # Update training metrics.
    self._update_training_metrics(
      heads=correct_heads if "heads" in self._predict else None,
      head_scores=head_scores if "heads" in self._predict else None,
      labels=correct_labels if "labels" in self._predict else None,
      label_scores=label_scores if "labels" in self._predict else None,
      pad_mask=pad_mask)

    # Fill in the return values
    if "heads" in self._predict:
      losses["heads"] = head_loss
      correct["heads"] = correct_heads
      predictions["heads"] = head_preds

    if "labels" in self._predict:
      losses["labels"] = label_loss
      correct["labels"] = correct_labels
      predictions["labels"] = label_preds

    return predictions, losses, correct, pad_mask

  def parse(self, example):
    """Parse an example with this parser.

    Args:
      example: A single example that holds features in a dictionary.
        words: Tensor representing word embedding indices of words in the sentence.
        pos: Tensor representing pos embedding indices of pos in the sentence.
        morph: Tensor representing morph indices of the morphological features in words in the sentence.

    Returns:
      scores: a dictionary of scores representing edge and label predictions.
        edges: Tensor of shape (1, seq_len, seq_len)
        labels: Tensor of shape (1, seq_len, n_labels)
    """
    label_embeddings = self.model.label_embeddings.get_weights()
    words, pos, morph, dep_labels = (example["words"], example["pos"],
                                     example["morph"], example["dep_labels"])
    scores = self.model({"words": words, "pos": pos,
                         "morph": morph, "labels": dep_labels}, label_embeddings=label_embeddings,
                        training=False)
    return scores

class LabelFirstParsingModel(tf.keras.Model):
  """Label first parsing model predicts labels before edges."""
  def __init__(self, *,
               n_dep_labels: int,
               word_embeddings: Embeddings,
               name="Label_First_Parsing_Model",
               predict: List[str],
               use_pos:bool = True,
               use_morph:bool=True,
               use_dep_labels:bool=False # for cases where we predict only edges using dep labels as gold.
               ):
    super(LabelFirstParsingModel, self).__init__(name=name)
    self.predict = predict
    self.use_pos = use_pos
    self.use_morph = use_morph
    self.use_dep_labels = use_dep_labels
    self._null_label = tf.constant(0)

    assert(not("labels" in self.predict and self.use_dep_labels)), "Can't use dep_labels both as feature and label!"
    self.word_embeddings = word_embeddings
    self.word_embeddings_layer = layer_utils.EmbeddingLayer(
      pretrained=word_embeddings,
      name="word_embeddings_layer",
      trainable=False)

    if self.use_pos:
      self.pos_embeddings = layer_utils.EmbeddingLayer(
        input_dim=37, output_dim=32,
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
      if self.use_dep_labels: # using dep labels as gold features.
        self.label_embeddings = layer_utils.EmbeddingLayer(input_dim=43,
                                                           output_dim=50,
                                                           name="label_embeddings",
                                                           trainable=True)

    self.head_perceptron = layer_utils.Perceptron(n_units=256,
                                                  activation="relu",
                                                  name="head_mlp")
    self.dep_perceptron = layer_utils.Perceptron(n_units=256,
                                                 activation="relu",
                                                 name="dep_mlp")
    self.edge_scorer = layer_utils.EdgeScorer(n_units=256, name="edge_scorer")
    logging.info((f"Set up {name} to predict {predict}"))


  def call(self, inputs, label_embeddings, training=True): # inputs = Dict[str, tf.keras.Input]
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
    tokens = [self.word_embeddings.itos(idx=index.numpy()) for index in word_inputs[0]]
    word_features = self.word_embeddings_layer(word_inputs)
    concat_list = [word_features]
    if self.use_pos:
      pos_inputs = inputs["pos"]
      pos_features = self.pos_embeddings(pos_inputs)
      concat_list.append(pos_features)
    if self.use_morph:
      morph_inputs = inputs["morph"]
      concat_list.append(morph_inputs)
    if self.use_dep_labels:
      label_inputs=inputs["labels"]
      # print('label inputs ', label_inputs)
      masked_labels = []
      for vector in label_inputs:
        if any(_label_index_to_name(index.numpy()) == "root" for index in vector):
          # print(list(_label_index_to_name(index.numpy()) for index in vector))
          tag_index = _label_name_to_index("root")
          # keep the root value and replace all other with 0
          vector = tf.multiply(vector, tf.cast(vector==tag_index, tf.int64))
          masked_labels.append(tf.expand_dims(vector, 0))
      label_inputs=tf.concat(masked_labels, axis=0)
      # print("label inputs ", label_inputs)
      # get the indices of the label we are keeping
      indices = _enumerated_tensor(tf.expand_dims(tf.argmax(label_inputs, 1), 1))
      # print("indices ", indices)
      # pass the updated inputs through embedding
      label_features = self.label_embeddings(label_inputs)
      # print("label features ", label_features)
      # input()
      # get the slices where the embedding belongs to the kept label
      label_slices = tf.gather_nd(label_features, indices)
      # print("label slices ", label_slices)
      # change all the rest except for the label slice to t.ones.
      mask_features = tf.ones_like(label_features)
      # print("mask features ", mask_features)

      label_features = tf.tensor_scatter_nd_update(mask_features, indices=indices, updates=label_slices)
      # print("label features ", label_features)
      # input()

      # print('label features ', label_features)
      concat_list.append(label_features)
    if len(concat_list) > 1:
      sentence_repr = self.concatenate(concat_list)
      # print(sentence_repr)
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
    self.dropout3 = layers.Dropout(rate=dropout_rate, name="dropout3")

  def call(self, input_tensor, training=True):
    dropout = self.dropout_rate > 0 and training
    # print("dropout is ", dropout)
    if dropout:
      out = self.lstm1(input_tensor)
      out = self.dropout1(out)
      out = self.lstm2(out)
      out = self.dropout2(out)
      out = self.lstm3(out)
      out = self.dropout3(out)
    else:
      out = self.lstm1(input_tensor)
      out = self.lstm2(out)
      out = self.lstm3(out)
    return out


if __name__ == "__main__":
  # use_pretrained_weights_from_labeler = True
  labels_to_keep = ["root", "obl", "nmod_poss", "amod",
                    "obj", "conj", "advmod", "det", "acl",
                    "case", "cc",  "advcl", "compound", "flat", "nmod",
                    "advmod_emph", "nummod", "ccomp", "compound_lvc", "cop", "csubj",
                    "compound_redup", "discourse", "aux_q", "parataxis", "aux", "mark",
                    "iobj", "cc_preconj", "appos",  "clf", "xcomp", "vocative"]
  word_embeddings = load_models.load_word_embeddings()
  prep = load_models.load_preprocessor(word_embeddings)
  label_feature = next(
    (f for f in prep.sequence_features_dict.values() if f.name == "dep_labels"), None)
  for label in labels_to_keep:
    try:
      index = _label_name_to_index(label)
    except KeyError:
      logging.Error(f"Couldn't find name for {label}")
    logging.info(f"-----------> TESTING LABEL {label} <----------------")
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = f"debug/lfp_tag_ranker_{label}/" + current_time
    parser_model_name = f"lfp_tag_ranking_model_{label}"
    logging.info(f"model name {parser_model_name}")
    parser = LFPTagRanker(word_embeddings=prep.word_embeddings,
                          n_output_classes=label_feature.n_values,
                          predict=["heads",
                                 # "labels"
                                 ],
                          features=["words",
                                    "pos",
                                    "morph",
                                    "heads",
                                    "dep_labels"],
                          log_dir=log_dir,
                          test_every=1,
                          model_name=parser_model_name)


  # get the data
    train_treebank= "tr_boun-ud-train-random10.pbtxt"
    test_treebank = "tr_boun-ud-test-random10.pbtxt"
    train_dataset, test_dataset = load_models.load_data(preprocessor=prep,
                                                        train_treebank=train_treebank,
                                                        batch_size=10,
                                                        test_treebank=test_treebank)

    _metrics = parser.train(dataset=train_dataset, epochs=2, test_data=test_dataset)
    print(_metrics)
    writer.write_proto_as_text(_metrics, f"./model/nn/plot/final/{parser_model_name}_metrics.pbtxt")
    # nn_utils.plot_metrics(name=parser_model_name, metrics=metrics)
    # parser.save_weights()
    # logging.info("weights saved!")
