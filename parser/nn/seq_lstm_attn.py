"""
This is a sequential attention based labeler, where, in contrast to the previous
sequential labelers, we don't pass the predicted label from the previous token t
as input for the next label prediction, but instead only pass the hidden state and
the cell state.

We set up the model this way because there's not as strong a dependency between
adjacent labels in a dependency labeling task.
"""

import logging
import tensorflow as tf
import numpy as np
import tensorflow.keras.backend as K

from input import embeddor
from proto import metrics_pb2
from typing import List, Dict, Tuple

from parser.nn import base_parser
from parser.nn import layer_utils
from tagset.dep_labels import dep_label_enum_pb2 as dep_label_tags
from tensorflow.keras import layers, metrics, losses, optimizers, initializers
from typing import Dict, Tuple

Dataset = tf.data.Dataset
Embeddings = embeddor.Embeddings


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

class SeqLSTMAttnModel(tf.keras.Model):

  def __init__(self, *,
    word_embeddings: Embeddings,
    n_units: int = 256,
    n_output_classes: int,
    use_pos = True,
    use_morph = True,
    name = "Seqeuantial_LSTM_attention"):

    super(SeqLSTMAttnModel, self).__init__(name=name)

    self.n_lstm_units = n_units
    self.n_s = self.n_lstm_units*2
    self.use_pos = use_pos
    self.use_morph = use_morph
    self.word_embeddings = word_embeddings
    self.n_output_classes = n_output_classes
    self.scores = {}

    # Pre attention layers
    self.word_embeddings_layer = layer_utils.EmbeddingLayer(
      pretrained=word_embeddings, name="word_embeddings_layer"
    )
    if self.use_pos:
      self.pos_embeddings = layer_utils.EmbeddingLayer(
        input_dim=37, output_dim=32,
        name="pos_embeddings", trainable=True
      )
    self.concatenate = layers.Concatenate(name="concat")

    # The pre attention bidirectional LSTM layer.
    self.pre_attn_lstm = layer_utils.LSTMBlock(
                                            n_units=n_units,
                                            num_layers=2,
                                            name="lstm_block")

    # Attention layers
    # We will override the repeat factor of this in repeator in call function
    # depending on the sequence length of the particular batch.
    self.s0 = layers.Input(self.n_s, name="s0")
    self.c0 = layers.Input(self.n_s, name="c0")
    self.attn_repeator = layers.RepeatVector(1)
    self.attn_concatenator = layers.Concatenate(axis=-1)
    self.attn_densor1 = layers.Dense(50, activation = "tanh")
    self.attn_densor2 = layers.Dense(10, activation = "tanh")
    self.attn_densor3 = layers.Dense(1, activation = "relu")
    self.attn_activator = layers.Activation(softmax, name = "attention_weights")
    self.attn_dotor = layers.Dot(axes = 1)

    # Post attention layers
    self.post_attn_lstm = layers.LSTM(units=self.n_s,
                                      return_state=True)
    self.output_layer = layers.Dense(units=n_output_classes,
                                     activation=softmax,
                                     name="output")
    # self.output_layer = layers.Dense(units=n_dep_labels, name="labels")

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
    """Call function.
    Args:
        X: inputs
        Tx: length of the input sequence
        n_a: hidden state size of the Bi-LSTM.
        n_s = hidden state size of the post-attention LSTM.
    """
    outputs = []
    word_inputs = inputs["words"]
    # print("word inputs ", word_inputs)
    word_features = self.word_embeddings_layer(word_inputs)
    batch_size, seq_len, _ = word_features.shape
    self.attn_repeator.n = seq_len

    concat_list = [word_features]
    s0 = layers.Input(shape=(self.n_s), name="s0")
    # print("s0 shape ", s0.shape)
    s0 = tf.zeros((batch_size, self.n_s))
    # print("s0 shape ", s0.shape)
    c0 = layers.Input(shape=(self.n_s), name="c0")
    # print("c0 shape ", c0.shape)
    c0 = tf.zeros((batch_size, self.n_s))
    # print("c0 shape ", c0.shape)

    if self.use_pos:
      pos_inputs = inputs["pos"]
      pos_features = self.pos_embeddings(pos_inputs)
      concat_list.append(pos_features)

    if self.use_morph:
      morph_inputs = inputs["morph"]
      concat_list.append(morph_inputs)

    if len(concat_list) > 1:
      sentence_repr = self.concatenate(concat_list)
    else:
      sentence_repr = word_features

    # Pass inputs from pre attention lstm
    a = self.pre_attn_lstm(sentence_repr)
    # print("a shape ", a.shape)
    s = s0
    c = c0

    for t in range(seq_len):
      context = self._one_step_attention(a, s)
      # print("context is ", context)
      # print("context shape is ", context.shape)
      # print("s is ", s)
      # print("c is ", c)
      # input("press to cont.")
      s, _, c = self.post_attn_lstm(inputs=context,
                                     initial_state=[s,c])
      # print("s is ", s)
      # print("c is ", c)

      out = self.output_layer(s)
      # print("output is ", out)
      # input("press to cont.")
      outputs.append(out)
    # print("outputs ", outputs)
    label_scores = tf.reshape(tf.concat(outputs, axis=1),
                              shape=(batch_size, seq_len, self.n_output_classes))
    # print("outputs concat: ", label_scores)
    self.scores["labels"] = label_scores
    return self.scores


class SeqLSTMAttnLabeler(base_parser.BaseParser):
  def __init__(self, *,
               word_embeddings: Embeddings,
               n_output_classes: int = None,
               predict: List[str],
               features: List[str] = ["words"],
               model_name: str,
               log_dir: str,
               test_every: int = 10):
    super(SeqLSTMAttnLabeler, self).__init__(word_embeddings=word_embeddings,
                                             n_output_classes=n_output_classes,
                                             predict=predict,
                                             features=features,
                                             model_name=model_name,
                                             log_dir=log_dir,
                                             test_every=test_every)
    self.optimizer = tf.keras.optimizers.Adam(0.01, beta_1=0.9, beta_2=0.9, clipnorm=1.)

  @property
  def _optimizer(self):
    raise NotImplementedError("Please use the .optimizer attribute instead.")


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

    return losses.SparseCategoricalCrossentropy(from_logits=False,
                                                reduction=tf.keras.losses.Reduction.NONE)

  @property
  def inputs(self):
    word_inputs = tf.keras.Input(shape=(None, ), name="words")
    pos_inputs = tf.keras.Input(shape=(None, ), name="pos")
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
    if words.shape[0] * words.shape[1] > pad_mask.shape[0] * pad_mask.shape[1]:
      # skip the dummy top token
      words = words[:, 1:]
    tf.assert_equal(words.shape[0] * words.shape[1], pad_mask.shape[0] * pad_mask.shape[1])
    words_reshaped = tf.reshape(words, shape=pad_mask.shape)
    # print("words reshaped ", words_reshaped)
    n_words_in_batch = len(tf.boolean_mask(words_reshaped, pad_mask))
    return n_words_in_batch

  def _parsing_model(self, model_name):
    super()._parsing_model(model_name, sequential=True)
    print(f"""Using features pos: {self._use_pos}, morph: {self._use_morph}""")
    model = SeqLSTMAttnModel(
      n_output_classes=self._n_output_classes,
      word_embeddings=self.word_embeddings,
      name=model_name,
      use_pos=self._use_pos,
      use_morph=self._use_morph
    )
    # model(inputs=self.inputs)
    return model

  @tf.function
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
    if "heads" in self._predict:
      raise ValueError("Cannot predict heads using dependency labeler.")

    predictions, correct, losses = {}, {}, {}
    # print("words ", words)
    # print("words ", words[:, 1:])
    with tf.GradientTape() as tape:
      scores = self.model({"words": words, "pos": pos, "morph": morph,
                           "labels": dep_labels}, training=True)

      # Remove the 0th token from scores, preds, and labels, as it's dummy.
      label_scores = scores["labels"][:, 1:, :]
      # print("scores ", label_scores)
      preds = tf.argmax(label_scores, axis=2)
      # print("preds ", preds)
      correct_labels = dep_labels[:, 1:]
      # print("correct labels ", correct_labels)

      # Sanity check
      tf.assert_equal(correct_labels.shape, preds.shape)
      tf.assert_equal(tf.argmax(label_scores, axis=2), preds)
      # input("press to cont.")

      # Flatten the label scores to (batch_size*seq_len, n_classes) (i.e. 340, 36).
      label_scores = self._flatten(label_scores, outer_dim=label_scores.shape[2])
      # print(f"label scores after flatten {label_scores}, {label_scores.shape}")
      # input("press to cont.")
      # Flatten the correct labels to the shape (batch_size*seq_len, 1) (i.e. 340,1)
      # Correct labels are indices for the right label for each token.
      correct_labels = self._flatten(correct_labels)
      # print(f"correct labels after flattened {correct_labels}, {correct_labels.shape}")
      preds_flattened = self._flatten(preds)
      # print(f"label preds after flattened {preds_flattened}, {preds_flattened.shape}")
      # input("press to cont.")
      label_loss = tf.expand_dims(self._label_loss(label_scores, correct_labels), axis=-1)

      # nan_loss = any(tf.math.is_nan(label_loss))
      # if nan_loss:
      #  print("label loss ", label_loss)
      #  print("labe scores ", label_scores)
      #  input("press to cont.")

      # if nan_loss:
      #  for grad in grads:
      #    print("grad ", grad)
      #    input("press to cont.")

    grads = tape.gradient(label_loss, self.model.trainable_weights)
    self.optimizer.apply_gradients(zip(grads, self.model.trainable_weights))
    pad_mask = self._flatten(words[:, 1:] != 0)
    # print("pad mask ", pad_mask)
    # input("press to cont.")
    # Update training metrics.
    self._update_training_metrics(
      labels=correct_labels,
      label_scores=label_scores,
      pad_mask=pad_mask)

    losses["labels"] = label_loss
    # print("label losses in train step", losses["labels"])
    correct["labels"] = correct_labels
    predictions["labels"] = preds_flattened

    return predictions, losses, correct, pad_mask

  def test(self, *, dataset: Dataset):
    print("Testing on the test set. Using batch mode..")
    # resetting test stats at the beginning.
    for key in self.test_stats:
      self.test_stats[key] = 0.0

    for step, batch in enumerate(dataset):
      words, pos, morph = (batch["words"], batch["pos"], batch["morph"])
      dep_labels = batch["dep_labels"]
      scores = self.model({"words": words, "pos": pos, "morph": morph,
                           "labels": dep_labels}, training=False)

      # Remove the 0th token from scores, preds, and labels, as it's dummy.
      label_scores = scores["labels"][:, 1:, :]
      preds = tf.argmax(label_scores, axis=2)
      # print("preds ", preds)
      correct_labels = dep_labels[:, 1:]
      # print("correct labels ", correct_labels)
      # input("press to cont.")

      # Sanity check
      tf.assert_equal(correct_labels.shape, preds.shape)
      tf.assert_equal(tf.argmax(label_scores, axis=2), preds)

      label_scores = self._flatten(label_scores, outer_dim=label_scores.shape[2])
      # print(f"label scores after flatten {label_scores}, {label_scores.shape}")

      # Flatten the correct labels to the shape (batch_size*seq_len, 1) (i.e. 340,1)
      # Correct labels are indices for the right label for each token.
      correct_labels = self._flatten(correct_labels)
      # print(f"correct labels after flattened {correct_labels}, {correct_labels.shape}")
      preds_flattened = self._flatten(preds)
      # print("preds flatted ", preds_flattened)

      pad_mask = self._flatten(words[:, 1:] != 0)
      # print("words ", words)
      words = words[:, 1:]
      # print("words after ", words)
      # print("pad mask ", pad_mask)

      correct_predictions_dict = self._correct_predictions(
        label_predictions=preds_flattened,
        correct_labels=correct_labels,
        pad_mask=pad_mask
      )

      n_words_in_batch = self._n_words_in_batch(words, pad_mask=pad_mask)

      self._update_correct_prediction_stats(correct_predictions_dict,
                                            n_words_in_batch=n_words_in_batch,
                                            stats="test")
    logging.info(f"Test stats: {self.test_stats}")
    test_results = self._compute_metrics(stats="test")
    return test_results

  @tf.function
  def parse(self, example: Dict, training=False):
    """Parse an example with this parser.

    Args:
      example: A single example that holds features in a dictionary.
        words: Tensor representing word embedding indices of words in the sentence.
        pos: Tensor representing pos embedding indices of pos in the sentence.
        morph: Tensor representing morph indices of the morphological features in words in the sentence.

    Returns:
      label_preds
    """
    words, pos, morph = (example["words"], example["pos"], example["morph"])
    scores = self.model({"words": words, "pos": pos, "morph": morph},
                        training=False)
    # Remove the 0th token from scores.
    label_scores = scores["labels"][:, 1:, :]
    # print("label scores ", label_scores)
    label_preds = tf.argmax(label_scores, axis=2)
    # print("label preds ", label_preds)
    return label_preds
