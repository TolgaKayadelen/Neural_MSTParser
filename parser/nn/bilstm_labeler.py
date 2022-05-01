import os
import logging
import tensorflow as tf

from parser.nn import base_parser, architectures
from proto import metrics_pb2
from tensorflow.keras import layers, metrics, losses, optimizers
from typing import Dict, Tuple

Dataset = tf.data.Dataset

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
    model = architectures.LSTMLabelingModel(
      n_output_classes=self._n_output_classes,
      word_embeddings=self.word_embeddings,
      name=model_name,
      use_pos=self._use_pos,
      use_morph=self._use_morph,
      return_lstm_output=True,
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
    with tf.GradientTape() as tape:
      scores, lstm_output = self.model({"words": words, "pos": pos, "morph": morph,
                                       "labels": dep_labels}, training=True)
      label_scores = scores["labels"]
      # print("label scores ", label_scores)
      # print("concat ", lstm_output)
      # input("press to continue.")
      pad_mask = self._flatten((words != 0))
      # Get the predicted label indices from the label scores, tensor of shape (batch_size*seq_len, 1)
      label_preds = self._flatten(tf.argmax(label_scores, axis=2))
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
      scores, _ = self.parse(example)
      label_scores = scores["labels"]
      label_preds = self._flatten(tf.argmax(label_scores, 2))
      correct_labels = self._flatten(example["dep_labels"])
      label_accuracy.update_state(correct_labels, label_preds)

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
    if self.model.return_lstm_output:
      scores, lstm_output = self.model({"words": words, "pos": pos,
                            "morph": morph, "labels": dep_labels}, training=False)
      return scores, lstm_output
    else:
      scores,  = self.model({"words": words, "pos": pos,
                             "morph": morph, "labels": dep_labels}, training=False)
      return scores, None


