import collections
import logging
import os
import sys
import time

from input import embeddor
import tensorflow as tf
import tensorflow_addons as tfa
import matplotlib as mpl
import numpy as np

from abc import ABC, abstractmethod
from parser.nn import architectures
from proto import metrics_pb2
from tagset.reader import LabelReader
from tensorflow.keras import layers, metrics, losses, optimizers
from typing import List, Dict, Tuple
from util.nn import nn_utils

# Set up basic configurations
logging.basicConfig(format='%(levelname)s : %(message)s', level=logging.INFO)

# Set up type aliases
Dataset = tf.data.Dataset
Embeddings = embeddor.Embeddings
Metrics = metrics_pb2.Metrics

# Path for saving or loading pretrained models.
_MODEL_DIR = "./model/nn/pretrained"

class BaseParser(ABC):
  """The base parser implements methods that are shared between all parsers."""

  def __init__(self, *,
               word_embeddings: Embeddings,
               n_output_classes: int,
               predict: List[str],
               features: List[str] = ["words"],
               model_name: str):
    # Embeddings
    self.word_embeddings = word_embeddings

    self._n_output_classes = n_output_classes

    self._predict = predict

    self.features = features

    self.model_name = model_name

    self.model = self._parsing_model(model_name)

    self.training_stats = collections.Counter()

    self.test_stats = collections.Counter()

    self._eval_metrics = self.eval_metrics()

    self._training_metrics = self._training_metrics()

    assert(self._predict == self.model.predict), "Inconsistent Configuration"

  @property
  @abstractmethod
  def _optimizer(self):
    pass

  @abstractmethod
  def _training_metrics(self):
    pass

  @property
  @abstractmethod
  def _head_loss_function(self):
    pass

  @property
  @abstractmethod
  def _label_loss_function(self):
    pass

  @property
  @abstractmethod
  def inputs(self):
    pass

  @abstractmethod
  def _parsing_model(self, model_name):
    """Defines the parsing/labeling model. Subclasses should call the model they want
    from architectures.
    """
    self._use_pos = "pos" in self.features
    self._use_morph = "morph" in self.features
    self._use_dep_labels = False
    if "dep_labels" in self.features:
      if "labels" in self._predict:
        logging.warning(
          """Dep labels are requested as features but setting labels as prediction target.
          Ignoring the dep_labels as feature.""")
      else:
        self._use_dep_labels = True

  @abstractmethod
  def _n_words_in_batch(self, words, pad_mask=None):
    """Returns number of words that is in each batch without padding."""
    pass

  @staticmethod
  def _log(description: str, results: Dict):
    """Special log wrapper that logs a description message and key,values in a dictionary."""
    logging.info(description)
    for key, value in results.items():
      if type(value) in [metrics.SparseCategoricalAccuracy, metrics.CategoricalAccuracy]:
        logging.info(f"{key}:  {value.result().numpy()}")
      else:
        logging.info(f"{key}:  {value}")

  @staticmethod
  def uas(n_correct_heads, n_tokens):
    """Unlabeled attachment score."""
    return n_correct_heads / n_tokens

  @staticmethod
  def las(n_correct_heads_and_labels, n_tokens):
    """Labeled attachment score."""
    return n_correct_heads_and_labels / n_tokens

  @staticmethod
  def ls(n_correct_labels, n_tokens):
    """Label score. i.e. percentage of correctly labeled tokens."""
    return n_correct_labels / n_tokens

  @staticmethod
  def load(name: str, path=None):
    """Loads a pretrainged model from path."""
    if path is None:
      path = os.path.join(_MODEL_DIR, name)
    self.model.load_weights(os.path.join(path, name))
    logging.info(f"Loaded model from model named: {name} in {_MODEL_DIR}")

  @staticmethod
  def _get_predicted_labels(scores):
    """Returns predicted labels for model scores.

    The scores should be a tf.Tensor of shape (batch_size, seq_len, label_probs) where
    the second axis hold target label predictions for each token in the sequence.
    """
    return tf.argmax(scores, 2)

  @staticmethod
  def _flatten(_tensor, outer_dim=1):
    """Flattens a 3D tensor into a 2D one.

    A tensor of [batch_size, seq_len, outer_dim] is converted to ([batch_size*seq_len], outer_dim)
    ie. into a two-dim tensor
    """
    batch_size, seq_len = _tensor.shape[0], _tensor.shape[1]
    return tf.reshape(_tensor, shape=(batch_size*seq_len, outer_dim))

  def __str__(self):
    return str(self.model.summary())

  def eval_metrics(self) -> Metrics:
    """Sets up metrics to track for this parser."""
    metrics_list = ("uas", "uas_test", "head_loss")
    if "labels" in self._predict:
      metrics_list += ("ls", "ls_test", "las", "las_test", "label_loss")
    return nn_utils.set_up_metrics(*metrics_list)


  def  _compute_correct_predictions_in_step(self, *,
                                            head_predictions,
                                            correct_heads,
                                            label_predictions=None,
                                            correct_labels=None,
                                            pad_mask=None
                                            ):
    """Computes correctly predicted edges and labels.

    Args:
      head_predictions: tensor of shape (n, 1) heads predicted by the system for
        each token in the batch.
      correct_heads: tensor of shape (n, 1) the correct heads for this batch for
        each token in the batch.
      label_predictions: tensor of shape (n, 1), similar to head_predictions.
      correct_labels: tensor of shape (n, 1), similar to correct_heads.
      pad_mask: mask of padded tokens for this batch.

    Returns:
      A dictionary that contains:
      chp: correct_head_predictions, a boolean np.array where correct predictions have value True.
      clp: correct_label_predictions, a boolean np.array as above.
      n_chp: number of correctly predicted heads.
      n_clp: number of correctly predicted labels.
    """

    if pad_mask is None:
      pad_mask = np.full(shape=head_predictions.shape, fill_value=True, dtype=bool)
    correct_head_preds = tf.boolean_mask(head_predictions == correct_heads, pad_mask)
    n_correct_head_preds = np.sum(correct_head_preds)

    if "labels" in self._predict:
      correct_label_preds = tf.boolean_mask(label_predictions == correct_labels, pad_mask)
      n_correct_label_preds = np.sum(correct_label_preds)
    else:
      correct_label_preds=None
      n_correct_label_preds=None

    return {
      "chp": correct_head_preds,
      "n_chp": n_correct_head_preds,
      "clp": correct_label_preds,
      "n_clp": n_correct_label_preds,
    }

  def _update_training_metrics(self, *,
                               heads, head_scores,
                               labels=None, label_scores=None,
                               pad_mask=None):
    """Updates the training metrics after each epoch.

    Args:
      heads: tensor of shape (n, 1) where n equals (batch_size*seq_len), representing correct index
        for each token.
      head_scores: tensor of shape (n, seq_len). n equals (batch_size*seq_len). Holds probability values
        for each token's head prediction.
      labels: tensor of shape (n, 1), similar to heads.
      label_scores: tensor of shape (n, n_labels), similar to label_scores.
    """
    # Updates training metrics.
    self._training_metrics["heads"].update_state(heads, head_scores, sample_weight=pad_mask)
    if labels is not None:
      self._training_metrics["labels"].update_state(labels, label_scores, sample_weight=pad_mask)

  def _update_eval_metrics(self, train_metrics, loss_metrics, test_metrics):
    """Updates eval metrics (UAS, LAS, LS) for training and test data as well as loss metrics."""
    if test_metrics is None:
      all_metrics = {**train_metrics, **loss_metrics}
    else:
      all_metrics = {**train_metrics, **loss_metrics, **test_metrics}
    for key, value in all_metrics.items():
      if not self._eval_metrics.metric[key].tracked:
        logging.info(f"Metric {key} is not tracked.")
      else:
        try:
          self._eval_metrics.metric[key].value_list.value.append(value)
        except KeyError:
          logging.error(f"No such metric as {key}!!")

  def _update_stats(self, correct_predictions_dict, n_words_in_batch, stats="training"):
    """Updates parsing stats at the end of each training or test step.

    The stats we keep track of are the following:
      n_tokens: total number of tokens in the data.
      n_chp: number of correctly predicted heeds.
      n_clp: number of correctly predicted labels.
      n_chlp: number of tokens for which both head and label is correctly predicted.

    These are later used for computing eval metrics like UAS, LS, and LAS.
    """
    if stats == "training":
      stats = self.training_stats
    else:
      stats = self.test_stats
    stats["n_tokens"] += n_words_in_batch
    stats["n_chp"] += correct_predictions_dict["n_chp"]
    if correct_predictions_dict["n_clp"] is not None:
      stats["n_clp"] += correct_predictions_dict["n_clp"]
      h = correct_predictions_dict["chp"]
      l = correct_predictions_dict["clp"]
      if not len(h) == len(l):
        raise RuntimeError("Fatal: Mismatch in the number of heads and labels.")
      stats["n_chlp"] += np.sum(
        [1 for tok in zip(h, l) if tok[0] == True and tok[1] == True]
      )

  def _head_loss(self, head_scores, correct_heads):
    """Computes loss for head predictions of the parser."""
    return self._head_loss_function(correct_heads, head_scores)

  def _label_loss(self, label_scores, correct_labels):
    """Computes loss for label predictions for the parser."""
    return self._label_loss_function(correct_labels, label_scores)

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
    with tf.GradientTape() as tape:

      # Head scores = (batch_size, seq_len, seq_len), Label scores = (batch_size, seq_len, n_labels)
      scores = self.model({"words": words, "pos": pos, "morph": morph,
                           "labels": dep_labels}, training=True)
      head_scores, label_scores = scores["edges"], scores["labels"]

      # Get the predicted head indices from the head scores, tensor of shape (batch_size*seq_len, 1)
      head_preds = self._flatten(tf.argmax(head_scores, axis=2))

      # Flatten the head scores to (batch_size*seq_len, seq_len) (i.e. 340, 34).
      # Holds probs for each token's head prediction.
      head_scores = self._flatten(head_scores, outer_dim=head_scores.shape[2])

      # Flatten the correct heads to the shape (batch_size*seq_len, 1) (i.e. 340,1)
      # Index for the right head for each token.
      correct_heads = self._flatten(heads)
      pad_mask = self._flatten((words != 0))

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
    if "labels" in self._predict:
      grads = tape.gradient([head_loss, label_loss], self.model.trainable_weights)
    else:
      grads = tape.gradient(head_loss, self.model.trainable_weights)

    self._optimizer.apply_gradients(zip(grads, self.model.trainable_weights))

    # Update training metrics.
    self._update_training_metrics(
      heads=correct_heads, head_scores=head_scores,
      labels=correct_labels if "labels" in self._predict else None,
      label_scores=label_scores if "labels" in self._predict else None,
      pad_mask=pad_mask)

    # Fill in the return values
    losses["heads"] = head_loss
    correct["heads"] = correct_heads
    predictions["heads"] = head_preds

    if "labels" in self._predict:
      losses["labels"] = label_loss
      correct["labels"] = correct_labels
      predictions["labels"] = label_preds

    return predictions, losses, correct, pad_mask


  def train(self, *,
            dataset: Dataset,
            epochs: int = 10,
            test_data: Dataset=None):
    """Training.

    Args:
      dataset: Dataset containing features and labels.
      epochs: number of epochs.
      test_data: Dataset containing features and labels for test set.
    Returns:
      training_results: logs of scores and losses at the end of training.
    """

    for epoch in range(1, epochs+1):
      test_results_for_epoch = None

      # Reset the training metrics
      for metric in self._training_metrics:
        self._training_metrics[metric].reset_states()

      # Reset the training stats
      for key in self.training_stats:
        self.training_stats[key] = 0.0

      logging.info(f"\n\n{'->' * 12} Training Epoch: {epoch} {'<-' * 12}\n\n")
      start_time = time.time()

      for step, batch in enumerate(dataset):

        words, pos, morph = batch["words"], batch["pos"], batch["morph"]
        dep_labels, heads = batch["dep_labels"], batch["heads"]

        # Get loss values, predictions, and correct heads/labels for this training step.
        predictions, losses, correct, pad_mask = self.train_step(words=words,
                                                                 pos=pos,
                                                                 morph=morph,
                                                                 dep_labels=dep_labels,
                                                                 heads=heads)

        # Get the correct heads/labels and correctly predicted heads/labels for this step.
        correct_predictions_dict = self._compute_correct_predictions_in_step(
          head_predictions=predictions["heads"],
          correct_heads=correct["heads"],
          label_predictions=predictions["labels"] if "labels" in self._predict else None,
          correct_labels=correct["labels"] if "labels" in self._predict else None,
          pad_mask=pad_mask
         )

        n_words_in_batch = self._n_words_in_batch(words=words,
                                                  pad_mask=pad_mask)

        # Update the statistics for correctly predicted heads/labels after this step.
        self._update_stats(correct_predictions_dict, n_words_in_batch)

      # Log stats at the end of epoch
      logging.info(f"Training stats: {self.training_stats}")

      # Compute UAS, LS, and LAS based on stats at the end of epoch.
      training_results_for_epoch = {
        "uas": self.uas(self.training_stats["n_chp"], self.training_stats["n_tokens"]),
        "ls": self.ls(self.training_stats["n_clp"], self.training_stats["n_tokens"]),
        "las": self.las(self.training_stats["n_chlp"], self.training_stats["n_tokens"]),
      }
      self._log(description=f"Training results after epoch {epoch}",
           results=training_results_for_epoch)


      self._log(description=f"Training metrics after epoch {epoch}",
                results=self._training_metrics)

      loss_results_for_epoch = {
        "head_loss": tf.reduce_mean(losses["heads"]).numpy(),
        "label_loss": tf.reduce_mean(losses["labels"]).numpy() if "labels" in self._predict else None
      }
      if epoch % 2 == 0 and test_data:
        test_results_for_epoch = self.test(dataset=test_data)
        self._log(description=f"Test results after epoch {epoch}", results=test_results_for_epoch)

      logging.info(f"Time for epoch {time.time() - start_time}")


      # Update the eval metrics based on training, test results and loss values.
      self._update_eval_metrics(
        train_metrics=training_results_for_epoch,
        loss_metrics=loss_results_for_epoch,
        test_metrics=test_results_for_epoch,
      )

    return self._eval_metrics


  def test(self, *, dataset: Dataset):
    """Tests the performance of this parser on some dataset."""
    print("Testing on the test set..")
    head_accuracy = tf.keras.metrics.Accuracy()
    label_accuracy = tf.keras.metrics.Accuracy()

    # resetting test stats at the beginning.
    for key in self.test_stats:
      self.test_stats[key] = 0.0

    # We traverse the test dataset not batch by batch, but example by example.
    for example in dataset:
      scores = self.parse(example)
      head_scores, label_scores = scores["edges"], scores["labels"]
      head_preds = self._flatten(tf.argmax(head_scores, 2))
      correct_heads = self._flatten(example["heads"])
      head_accuracy.update_state(correct_heads, head_preds)
      if "labels" in self._predict:
        # Get label predictions and correct labels
        label_preds = self._flatten(tf.argmax(label_scores, 2))
        correct_labels = self._flatten(example["dep_labels"])
        label_accuracy.update_state(correct_labels, label_preds)

      correct_predictions_dict = self._compute_correct_predictions_in_step(
        head_predictions=head_preds,
        correct_heads=correct_heads,
        label_predictions=label_preds  if "labels" in self._predict else None,
        correct_labels=correct_labels  if "labels" in self._predict else None,
      )
      self._update_stats(correct_predictions_dict,
                         example["words"].shape[1], stats="test")

    logging.info(f"Test stats: {self.test_stats}")
    test_results = {
      "uas_test": self.uas(self.test_stats["n_chp"], self.test_stats["n_tokens"]),
      "ls_test": self.ls(self.test_stats["n_clp"], self.test_stats["n_tokens"]),
      "las_test": self.las(self.test_stats["n_chlp"], self.test_stats["n_tokens"])
    }
    return test_results

  def parse(self, example: Dict):
    """Parse an example with this parser.

    Args:
      example: A single example that holds features in a dictionary.
        words: Tensor representing word embedding indices of words in the sentence.
        pos: Tensor representing pos embedding indices of pos in the sentence.
        morph: Tensor representing morh indices of the morphological features in words in the sentence.

    Returns:
      scores: a dictionary of scores representing edge and label predictions.
        edges: Tensor of shape (1, seq_len, seq_len)
        labels: Tensor of shape (1, seq_len, n_labels)
    """
    words, pos, morph, dep_labels = (example["words"], example["pos"],
                                     example["morph"], example["dep_labels"])
    scores = self.model({"words": words, "pos": pos,
                         "morph": morph, "labels": dep_labels}, training=False)
    return scores

  def save(self, suffix: int=0):
    """Saves the model to path"""
    model_name = self.model.name
    try:
      path = os.path.join(_MODEL_DIR, self.model.name)
      if suffix > 0:
        path += str(suffix)
        model_name = self.model.name+str(suffix)
      os.mkdir(path)
      self.model.save_weights(os.path.join(path, model_name), save_format="tf")
      logging.info(f"Saved model to  {path}")
    except FileExistsError:
      logging.warning(f"A model with the same name exists, suffixing {suffix+1}")
      self.save(suffix=suffix+1)