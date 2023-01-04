import os
import logging
import tensorflow as tf

from parser import base_parser
from parser.utils import layer_utils
from proto import metrics_pb2
from tensorflow.keras import layers, metrics, losses, optimizers
from typing import Dict, Tuple
from input import embeddor

Dataset = tf.data.Dataset
Embeddings = embeddor.Embeddings

class BiLSTMLabeler(base_parser.BaseParser):
  """A bi-lstm labeler that can be used for any kind of sequence labeling tasks."""
  @property
  def _optimizer(self):
    return tf.keras.optimizers.Adam(0.001, beta_1=0.9, beta_2=0.9)

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
                                                reduction=tf.keras.losses.Reduction.NONE)

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
      n_output_classes=self._n_output_classes,
      word_embeddings=self.word_embeddings,
      name=model_name,
      use_pos=self._use_pos,
      use_morph=self._use_morph,
      return_lstm_output=False,
    )
    model(inputs=self.inputs)
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
    pad_mask = self._flatten((words != 0))
    with tf.GradientTape() as tape:
      scores = self.model({"words": words, "pos": pos, "morph": morph, "labels": dep_labels}, training=True)
      label_scores = scores["labels"]
      # print("label scores ", label_scores)
      # input("press to continue.")
      # Get the predicted label indices from the label scores, tensor of shape (batch_size*seq_len, 1)
      label_preds = self._flatten(tf.argmax(label_scores, axis=2))
      if self._top_k:
        _, top_k_label_preds = tf.math.top_k(label_scores, k=self._k)
        top_k_label_preds = self._flatten(top_k_label_preds, outer_dim=top_k_label_preds.shape[2])

      # Flatten the label scores to (batch_size*seq_len, n_classes) (i.e. 340, 36).
      label_scores = self._flatten(label_scores, outer_dim=label_scores.shape[2])
      # Flatten the correct labels to the shape (batch_size*seq_len, 1) (i.e. 340,1)
      # Index for the right label for each token.
      correct_labels = self._flatten(dep_labels)
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
    predictions["top_k_labels"] = top_k_label_preds if self._top_k else None

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
      scores = self.parse(example)
      label_scores = scores["labels"]
      label_preds = self._flatten(tf.argmax(label_scores, 2))
      if self._top_k:
        _, top_k_label_preds = tf.math.top_k(label_scores, k=self._k)
        top_k_label_preds = self._flatten(top_k_label_preds, outer_dim=top_k_label_preds.shape[2])
      correct_labels = self._flatten(example["dep_labels"])
      label_accuracy.update_state(correct_labels, label_preds)

      correct_predictions_dict = self._correct_predictions(
        label_predictions=label_preds,
        correct_labels=correct_labels,
        top_k_label_predictions=top_k_label_preds if self._top_k else None
      )
      self._update_correct_prediction_stats(correct_predictions_dict,
                                            example["words"].shape[1],
                                            stats="test")

    logging.info(f"Test stats: {self.test_stats}")
    test_results = self._compute_metrics(stats="test")
    return test_results

  def parse(self, example: Dict):
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
    words, pos, morph, dep_labels = (example["words"], example["pos"],
                                     example["morph"], example["dep_labels"])

    scores = self.model({"words": words, "pos": pos, "morph": morph,
                         "labels": dep_labels}, training=False)

    return {"labels": scores["labels"],  "edges": None}



### The Keras Model

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

  def call(self, inputs):
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
      sentence_repr = self.lstm_block(concat)
      labels = self.labels(sentence_repr)
    else:
      sentence_repr = self.lstm_block(word_features)
      labels = self.labels(sentence_repr)
    return {"labels": labels,  "edges": None}
