import collections
import logging
import datetime
import os
import sys
import time

from input import embeddor
import tensorflow as tf
# import tensorflow_addons as tfa
import matplotlib as mpl
import numpy as np

from abc import ABC, abstractmethod
from data.treebank import sentence_pb2
from data.treebank import treebank_pb2
from parser.utils import architectures
from proto import metrics_pb2
from tagset.reader import LabelReader
from tensorflow.keras import layers, metrics, losses, optimizers
from typing import List, Dict, Tuple
from tagset.dep_labels import dep_label_enum_pb2 as dep_label_tags
from tagset.fine_pos import fine_tag_enum_pb2 as pos_tags
from tagset.coarse_pos import coarse_tag_enum_pb2 as category_tags
from tagset import reader
from util import writer
from util.nn import nn_utils

# Set up basic configurations
logging.basicConfig(format='%(levelname)s : %(message)s', level=logging.INFO)

# Set up type aliases
Dataset = tf.data.Dataset
Embeddings = embeddor.Embeddings
Metrics = metrics_pb2.Metrics

# Path for saving or loading pretrained models.
_MODEL_DIR = "./model/nn/pretrained/prod"

class BaseParser(ABC):
  """The base parser implements methods that are shared between all parsers."""

  def __init__(self, *,
               word_embeddings: Embeddings,
               language="tr",
               n_output_classes: int = None,
               predict: List[str],
               features: List[str] = ["words"],
               model_name: str,
               log_dir = None,
               test_every: int = 5,
               top_k: bool = False,
               k: int = 5,
               pos_embedding_vocab_size=37,
               one_hot_labels=False):

    self.language = language
    # Embeddings
    self.word_embeddings = word_embeddings

    self._n_output_classes = n_output_classes

    self._predict = predict

    self.features = features

    self.pos_embedding_vocab_size=pos_embedding_vocab_size

    self.one_hot_labels = one_hot_labels

    self._top_k = top_k

    self._k = k

    self.model_name = model_name

    self.model = self._parsing_model(model_name)

    self.training_stats = collections.Counter()

    self.test_stats = collections.Counter()

    self._metrics = self._metrics()

    self._training_metrics = self._training_metrics()

    self._test_every = test_every
    logging.info(f"Testing every {self._test_every}")

    self._log_dir = log_dir

    self.label_reader = reader.LabelReader.get_labels("dep_labels", self.language)

    if top_k and k <= 1:
      raise ValueError(f"k has to be greater than 1 if top_k is True. Received k = {k}")


    if self._log_dir:
      self.loss_summary_writer = tf.summary.create_file_writer(log_dir + "/loss")
      self.train_summary_writer = tf.summary.create_file_writer(log_dir + "/train")
      self.test_summary_writer = tf.summary.create_file_writer(log_dir + "/test")
      logging.info(f"Logging to {self._log_dir}")

    assert(all(val in ["heads", "labels"] for val in self._predict)), "Invalid prediction target!"

  # TODO: Make it an attribute. Making it a property causes tf.function fail in subclasses.
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
  def _parsing_model(self, model_name, sequential=False):
    """Defines the parsing/labeling model. Subclasses should call the model they want
    from architectures.
    """
    self._use_pos = "pos" in self.features
    self._use_morph = "morph" in self.features
    self._use_dep_labels = False
    if "dep_labels" in self.features:
      if "labels" in self._predict and not sequential:
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

  @staticmethod
  def _label_name_to_index(label_name):
    return dep_label_tags.Tag.Value(label_name)

  @staticmethod
  def _pos_index_to_name(pos_index):
    return pos_tags.Tag.Name(pos_index)

  @staticmethod
  def _pos_name_to_index(pos_name):
    return pos_tags.Tag.Value(pos_name)

  @staticmethod
  def _cat_index_to_name(cat_index):
    return category_tags.Tag.Name(cat_index)

  @staticmethod
  def _cat_name_to_index(cat_name):
    return category_tags.Tag.Value(cat_name)

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

  def __str__(self):
    return str(self.model.summary())

  def _label_index_to_name(self, label_index):
    # print("label index ", label_index)
    # input()
    try:
      name = self.label_reader.itov(label_index[0])
    except:
      if type(label_index) != int:
        label_index = int(label_index)
      name = self.label_reader.itov(label_index)
    return name

  def _metrics(self) -> Metrics:
    """Sets up metrics to track for this parser."""
    metrics_list = ()
    if "heads" in self._predict:
      metrics_list += ("uas", "uas_test", "head_loss")
    if "labels" in self._predict:
      metrics_list += ("ls", "ls_test", "las", "las_test", "label_loss")
    return nn_utils.set_up_metrics(*metrics_list)

  # @tf.function
  def  _correct_predictions(self, *,
                            head_predictions=None,
                            correct_heads=None,
                            label_predictions=None,
                            correct_labels=None,
                            top_k_label_predictions=None,
                            pad_mask=None
                            ):
    """Computes correctly predicted edges and labels and relevant stats for them.

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
        chp: correct_head_predictions, a boolean np.array where correct predictions are True.
        clp: correct_label_predictions, a boolean np.array as above.
        n_chp: number of correctly predicted heads.
        n_clp: number of correctly predicted labels.
    """
    correct_predictions_dict = {"chp": None, "n_chp": None, "clp": None, "n_clp": None,
                                "n_clp_topk": None}

    if pad_mask is None:
      if head_predictions is not None:
        pad_mask = np.full(shape=head_predictions.shape, fill_value=True, dtype=bool)
      elif label_predictions is not None:
        pad_mask = np.full(shape=label_predictions.shape, fill_value=True, dtype=bool)
      else:
        raise RuntimeError("Fatal: Either head predictions or label predictions should exist.")

    if "heads" in self._predict:
      correct_head_preds = tf.boolean_mask(head_predictions == correct_heads, pad_mask)
      n_correct_head_preds = tf.math.reduce_sum(tf.cast(correct_head_preds, tf.int32))
      correct_predictions_dict["chp"] = correct_head_preds
      correct_predictions_dict["n_chp"] = n_correct_head_preds.numpy()

    if "labels" in self._predict:
      # print("pad mask ", pad_mask)
      correct_label_preds = tf.boolean_mask(label_predictions == correct_labels, pad_mask)
      n_correct_label_preds = tf.math.reduce_sum(tf.cast(correct_label_preds, tf.int32))

      correct_predictions_dict["clp"] = correct_label_preds
      correct_predictions_dict["n_clp"] = n_correct_label_preds.numpy()

      if self._top_k:
        # check whether correct labels appears in the top k predictions.
        corr_in_topk_unmasked = tf.expand_dims(
          tf.reduce_any(correct_labels == tf.cast(top_k_label_predictions, tf.int64), axis=1), 1
        )
        # print("corr in topk unmasked ", corr_in_topk_unmasked)
        corr_in_topk = tf.boolean_mask(corr_in_topk_unmasked, pad_mask)
        # print("corr in topk ", corr_in_topk)
        n_corr_in_topk = np.sum(corr_in_topk)
        # print("n corr in topk ", n_corr_in_topk)
        # input()
        correct_predictions_dict["n_clp_topk"] = n_corr_in_topk
        # input("press to cont.")

    return correct_predictions_dict

  def _update_training_metrics(self, *,
                               heads=None,
                               head_scores=None,
                               labels=None,
                               label_scores=None,
                               pad_mask=None):
    """Updates the training metrics after each epoch.

    Args:
      heads: tensor of shape (n, 1) where n equals (batch_size*seq_len), representing
        correct index for each token.
      head_scores: tensor of shape (n, seq_len). n equals (batch_size*seq_len). Holds
        probability values for each token's head prediction.
      labels: tensor of shape (n, 1), similar to heads.
      label_scores: tensor of shape (n, n_labels), similar to label_scores.
    """
    # Updates training metrics.
    if heads is not None:
      self._training_metrics["heads"].update_state(heads, head_scores, sample_weight=pad_mask)

    if labels is not None:
      self._training_metrics["labels"].update_state(labels, label_scores, sample_weight=pad_mask)

  def _update_all_metrics(self, train_metrics, loss_metrics, test_metrics):
    """Updates eval metrics (UAS, LAS, LS) for training and test data as well as loss metrics."""
    if test_metrics is None:
      all_metrics = {**train_metrics, **loss_metrics}
    else:
      all_metrics = {**train_metrics, **loss_metrics, **test_metrics}
    for key, value in all_metrics.items():
      if not self._metrics.metric[key].tracked:
        logging.info(f"Metric {key} is not tracked.")
      else:
        try:
          # TODO: fix this at the point of the caller site and always send a real number.
          if tf.is_tensor(value):
            value = value.numpy()
          # input("press to cont.")
          self._metrics.metric[key].value_list.value.append(value)
        except KeyError:
          logging.error(f"No such metric as {key}!!")

  def _update_correct_prediction_stats(self,
                                       correct_predictions_dict,
                                       n_words_in_batch,
                                       stats="training"):
    """Updates parsing stats at the end of each training or test step.

    The stats we keep track of are the following:
      n_tokens: total number of tokens in the data.
      n_chp: number of correctly predicted heads.
      n_clp: number of correctly predicted labels.
      n_chlp: number of tokens for which both head and label is correctly predicted.

    These are later used for computing eval metrics like UAS, LS, and LAS.
    """
    if stats == "training":
      stats = self.training_stats
    else:
      stats = self.test_stats

    # print("words in batch ", n_words_in_batch)
    stats["n_tokens"] += n_words_in_batch
    h, l = None, None

    # Correct head predictions.
    if correct_predictions_dict["n_chp"] is not None:
      stats["n_chp"] += correct_predictions_dict["n_chp"]
      h = correct_predictions_dict["chp"]

    # Correct label predictions.
    if correct_predictions_dict["n_clp"] is not None:
      stats["n_clp"] += correct_predictions_dict["n_clp"]
      l = correct_predictions_dict["clp"]
    if self._top_k:
      stats["n_clp_topk"] += correct_predictions_dict["n_clp_topk"]

    # Tokens where both head and label predictions are correct.
    if h is not None and l is not None:
      if not len(h) == len(l):
        raise RuntimeError("Fatal: Mismatch in the number of heads and labels.")
      stats["n_chlp"] += np.sum(
        [1 for tok in zip(h, l) if tok[0] == True and tok[1] == True]
      )

  def _compute_metrics(self, stats="training"):
    """Computes metrics for uas, ls, and las after each epoch.

    Computes metrics based on the training and test stats.
    """
    _metrics = {}
    if stats == "test":
      stats = self.test_stats
      metric_suffix = lambda k: k+"_test"
    else:
      stats = self.training_stats
      metric_suffix = lambda k: k

    if "heads" in self._predict:
      _metrics[metric_suffix("uas")] = self.uas(stats["n_chp"], stats["n_tokens"])
    if "labels" in self._predict:
      _metrics[metric_suffix("ls")] = self.ls(stats["n_clp"], stats["n_tokens"])
      if self._top_k:
        _metrics[metric_suffix("ls_topk")] = self.ls(stats["n_clp_topk"], stats["n_tokens"])
    if "heads" in self._predict and "labels" in self._predict:
      _metrics[metric_suffix("las")] = self.las(stats["n_chlp"], stats["n_tokens"])
    return _metrics

  @tf.function
  def _head_loss(self, head_scores, correct_heads):
    """Computes loss for head predictions of the parser."""
    return self._head_loss_function(correct_heads, head_scores)

  @tf.function
  def _label_loss(self, label_scores, correct_labels):
    """Computes loss for label predictions for the parser."""
    return self._label_loss_function(correct_labels, label_scores)

  # @tf.function
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
        # print("head scores ", head_scores)
        # print("corr heads ", correct_heads)
        # print("head loss ", head_loss)
        # input()
        joint_loss = head_loss

      if "labels" in self._predict:

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
        # print("label scores ", label_scores)
        # print("corr labels ", correct_labels)
        # print("lbel loss ", label_loss)
        # input()
        joint_loss = head_loss + label_loss

    if "heads" in  self._predict and "labels" in self._predict:
      # grads = tape.gradient(joint_loss, self.model.trainable_weights)
      # biaffine parser likes separate optimization.
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
      predictions["top_k_labels"] = top_k_label_preds if self._top_k else None

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
    logging.info("Training started")
    early_stop_after_epochs = 0
    for epoch in range(1, epochs+1):
      test_results_for_epoch = None

      # Reset the training metrics before each epoch.
      for metric in self._training_metrics:
        self._training_metrics[metric].reset_states()

      # Reset the training stats before each epoch.
      for key in self.training_stats:
        self.training_stats[key] = 0.0

      logging.info(f"\n\n{'->' * 12} Training Epoch: {epoch} {'<-' * 12}\n\n")
      start_time = time.time()

      losses = collections.defaultdict(list)
      for step, batch in enumerate(dataset):
        words = batch["words"]
        dep_labels, heads = batch["dep_labels"], batch["heads"]
        pos = batch["pos"] if "pos" in batch.keys() else None
        morph = batch["morph"] if "morph" in batch.keys() else None

        # Get loss values, predictions, and correct heads/labels for this training step.
        predictions, batch_loss, correct, pad_mask = self.train_step(words=words,
                                                                     pos=pos,
                                                                     morph=morph,
                                                                     dep_labels=dep_labels,
                                                                     heads=heads)

        # Get the correct heads/labels and correctly predicted heads/labels for this step.
        correct_predictions_dict = self._correct_predictions(
          head_predictions=predictions["heads"] if "heads" in self._predict else None,
          correct_heads=correct["heads"] if "heads" in self._predict else None,
          label_predictions=predictions["labels"] if "labels" in self._predict else None,
          correct_labels=correct["labels"] if "labels" in self._predict else None,
          top_k_label_predictions=predictions["top_k_labels"] if "labels" in self._predict else None,
          pad_mask=pad_mask
         )

        # TODO: only send the pad mask shape to this function rather than pad_mask.
        n_words_in_batch = self._n_words_in_batch(words=words,
                                                  pad_mask=pad_mask)

        # Update the statistics for correctly predicted heads/labels after this step.
        self._update_correct_prediction_stats(correct_predictions_dict, n_words_in_batch)

        if "labels" in self._predict:
          losses["labels"].append(tf.reduce_sum(batch_loss["labels"]))
        if "heads" in self._predict:
          losses["heads"].append(tf.reduce_sum(batch_loss["heads"])) # * 1. / batch_size?
        # end inner for

      # Log stats at the end of epoch
      logging.info(f"Training stats: {self.training_stats}")

      # Compute UAS, LS, and LAS metrics based on stats at the end of epoch.
      training_results_for_epoch = self._compute_metrics()

      loss_results_for_epoch = {
        "head_loss": tf.reduce_mean(losses["heads"]).numpy() if "heads" in self._predict else None,
        "label_loss": tf.reduce_mean(losses["labels"]).numpy() if "labels" in self._predict else None
      }

      self._log(description=f"Training results after epoch {epoch}",
                results=training_results_for_epoch)

      self._log(description=f"Training metrics after epoch {epoch}",
                results=self._training_metrics)

      self._log(description=f"Loss after epoch {epoch}",
                results=loss_results_for_epoch)

      if epoch % self._test_every == 0 or epoch == epochs:
        log_las = False
        if self._log_dir:
          if "labels" in self._predict:
            with self.loss_summary_writer.as_default():
              tf.summary.scalar("label loss", loss_results_for_epoch["label_loss"], step=epoch)
            with self.train_summary_writer.as_default():
              tf.summary.scalar("label score", training_results_for_epoch["ls"], step=epoch)
            log_las=True

          if "heads" in self._predict:
            with self.loss_summary_writer.as_default():
              tf.summary.scalar("head loss", loss_results_for_epoch["head_loss"], step=epoch)
            with self.train_summary_writer.as_default():
              tf.summary.scalar("uas", training_results_for_epoch["uas"], step=epoch)
              if log_las:
                tf.summary.scalar("las", training_results_for_epoch["las"], step=epoch)

        if test_data is not None:
          test_results_for_epoch = self.test(dataset=test_data)
          if self._log_dir:
            with self.test_summary_writer.as_default():
              for key, value in test_results_for_epoch.items():
                print("key ", key, test_results_for_epoch[key])
                tf.summary.scalar(key, test_results_for_epoch[key], step=epoch)
          self._log(description=f"Test results after epoch {epoch}",
                    results=test_results_for_epoch)

      logging.info(f"Time for epoch {time.time() - start_time}")

      # Update the eval metrics based on training, test results and loss values.
      self._update_all_metrics(
        train_metrics=training_results_for_epoch,
        loss_metrics=loss_results_for_epoch,
        test_metrics=test_results_for_epoch,
      )
      if training_results_for_epoch["uas"] > 0.99 or training_results_for_epoch["ls"] > 0.99:
        if early_stop_after_epochs == 0:
          c = input("continue training: y/n")
          if c == "yes" or c == "y":
            print("continuing for 5 more epochs")
            early_stop_after_epochs = 5
          else:
            break
        else:
          early_stop_after_epochs -= 1
    # At the end of training, parse the data with the learned weights and save it as proto.
    self.parse_and_save(test_data)
    return self._metrics

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
      if "heads" in self._predict:
        head_preds = self._flatten(tf.argmax(head_scores, 2))
        # print("head preds ", head_preds)
        correct_heads = self._flatten(example["heads"])
        # print("correct heads ", correct_heads)
        head_accuracy.update_state(correct_heads, head_preds)
        # input()
      if "labels" in self._predict:
        # Get label predictions and correct labels
        label_preds = self._flatten(tf.argmax(label_scores, 2))
        if self._top_k:
          _, top_k_label_preds = tf.math.top_k(label_scores, k=self._k)
          top_k_label_preds = self._flatten(top_k_label_preds, outer_dim=top_k_label_preds.shape[2])
        correct_labels = self._flatten(example["dep_labels"])
        label_accuracy.update_state(correct_labels, label_preds)

      correct_predictions_dict = self._correct_predictions(
        head_predictions=head_preds if "heads" in self._predict else None,
        correct_heads=correct_heads if "heads" in self._predict else None,
        label_predictions=label_preds  if "labels" in self._predict else None,
        correct_labels=correct_labels  if "labels" in self._predict else None,
        top_k_label_predictions=top_k_label_preds if self._top_k else None,
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
        edges: Tensor of shape (1, seq_len, seq_len)
        labels: Tensor of shape (1, seq_len, n_labels)
    """
    words, dep_labels = (example["words"], example["dep_labels"])
    pos = example["pos"] if "pos" in example.keys() else None
    morph = example["morph"] if "morph" in example.keys() else None
    scores = self.model({"words": words, "pos": pos, "morph": morph,
                         "labels": dep_labels}, training=False)
    return scores

  def save_weights(self, suffix: int=0):
    """Saves the model weights to path in tf format."""
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
      self.save_weights(suffix=suffix+1)

  def load_weights(self, *, name: str, path=None):
    """Loads a pretrained model weights."""
    if path is None:
      path = os.path.join(_MODEL_DIR, name)
    else:
      path = os.path.join(path, name)
    load_status = self.model.load_weights(os.path.join(path, name))
    logging.info(f"Loaded model from model named: {name} in: {_MODEL_DIR}")
    load_status.assert_consumed()

  def parse_and_save(self, dataset: Dataset):
    """Parses a set of sentences with this parser and saves the gold and predicted outputs to a directory."""
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    gold_treebank_name = f"{self.model_name}_{current_time}_gold.pbtxt"
    parsed_treebank_name = f"{self.model_name}_{current_time}_parsed.pbtxt"
    eval_path = "./eval/eval_data"
    gold_treebank = treebank_pb2.Treebank()
    parsed_treebank = treebank_pb2.Treebank()

    for example in dataset:
      gold_sentence_pb2 = gold_treebank.sentence.add()
      parsed_sentence_pb2 = parsed_treebank.sentence.add()
      sent_id, tokens, dep_labels, heads = (example["sent_id"], example["tokens"],
                                            example["dep_labels"], example["heads"])
      # first populate gold treebank with the gold annotations
      index = 0
      # print("gold labels ", dep_labels)
      # input()
      for token, dep_label, head in zip(tokens[0], dep_labels[0], heads[0]):
        # print("token ", token, "dep label ", dep_label , "head ", head)
        # input()
        gold_sentence_pb2.sent_id = sent_id[0][0].numpy()
        token = gold_sentence_pb2.token.add(
          word=tf.keras.backend.get_value(token),
          label=self._label_index_to_name(tf.keras.backend.get_value(dep_label)),
          index=index)
        token.selected_head.address=tf.keras.backend.get_value(head)
        index += 1

      # next populate parsed data
      scores = self.parse(example)
      head_scores, label_scores = scores["edges"], scores["labels"]
      # get the heads and labels from parsed example
      if "heads" in self._predict:
        heads = tf.argmax(head_scores, axis=2)
        # print("parsed_heads ", heads)
      if "labels" in self._predict:
        dep_labels = tf.argmax(label_scores, axis=2)
        # print("parsed_labels ", dep_labels)
      index = 0
      # print("test labels ", dep_labels)
      # input()
      for token, dep_label, head in zip(tokens[0], dep_labels[0], heads[0]):
        parsed_sentence_pb2.sent_id = sent_id[0][0].numpy()
        token = parsed_sentence_pb2.token.add(
          word=tf.keras.backend.get_value(token),
          label=self._label_index_to_name(tf.keras.backend.get_value(dep_label)),
          index=index)
        token.selected_head.address=tf.keras.backend.get_value(head)
        index += 1

    writer.write_proto_as_text(gold_treebank, os.path.join(eval_path, gold_treebank_name))
    writer.write_proto_as_text(parsed_treebank, os.path.join(eval_path, parsed_treebank_name))
