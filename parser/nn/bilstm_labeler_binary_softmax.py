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
from parser.nn import load_models, layer_utils
from util.nn import nn_utils
from tagset.dep_labels import dep_label_enum_pb2 as dep_label_tags
from util import writer

Embeddings = embeddor.Embeddings
Dataset = tf.data.Dataset

class BiLSTMLabelerBinarySoftmax(base_parser.BaseParser):
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

    return losses.SparseCategoricalCrossentropy(from_logits=False,
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
    pass

  def _parsing_model(self, model_name):
    super()._parsing_model(model_name)
    print(f"""Using features pos: {self._use_pos}, morph: {self._use_morph}""")
    model = LSTMBinaryLabelingModelSoftmax(
      word_embeddings=self.word_embeddings,
      name=model_name,
      use_pos=self._use_pos,
      use_morph=self._use_morph,
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
        label_loss: tf.Tensor of (batch_size*seq_len, 1)
      correct: dictionary holding correct values for heads and labels.
        labels: tf.Tensor of (batch_size*seq_len, 1)
      predictions: dictionary holding correct values for heads and labels.
        labels: tf.Tensor of (batch_size*seq_len, 1)
      pad_mask: tf.Tensor of shape (batch_size*seq_len, 1) where padded words are marked as 0.
    """
    masked_labels = []
    for vector in dep_labels:
      if any(self._label_index_to_name(index.numpy()) == "root" for index in vector):
        # print(list(self._label_index_to_name(index.numpy()) for index in vector))
        tag_index = self._label_name_to_index("root")
        # keep the root value and replace all other with 0
        vector = tf.multiply(vector, tf.cast(vector==tag_index, tf.int64))
        masked_labels.append(tf.expand_dims(vector, 0))
    label_inputs=tf.concat(masked_labels, axis=0)
    # print("label inputs ", label_inputs)
    labels = tf.cast(tf.cast(label_inputs, tf.bool), tf.float32)
    # print("labels ", labels)
    total_labels = np.sum(labels)
    # print("total_labels ", total_labels)
    # input()
    predictions, correct, losses = {}, {}, {}
    pad_mask = self._flatten((words != 0))

    with tf.GradientTape() as tape:
      label_scores = self.model({"words": words, "pos": pos, "morph": morph,
                                 "labels": dep_labels}, training=True)

      # print("label scores ", label_scores)
      positive_label_predictions = label_scores[:, :, 1]
      # print("positive label preds ", positive_label_predictions)
      # input()
      positive_label_predictions = tf.argmax(positive_label_predictions, axis=1)
      # print("positive label preds ", positive_label_predictions)
      # print("correct labels argmax ", tf.argmax(labels, 1))
      # positive_label_predictions = tf.one_hot(positive_label_predictions, labels.shape[1])
      # print("positive label preds ", positive_label_predictions)
      # input()
      correctly_predicted_roots = positive_label_predictions == tf.argmax(labels, 1)
      total_correct_predictions = np.sum(correctly_predicted_roots)
      # print("corr pred roots ", correctly_predicted_roots)
      # print("total_correct_predictions ", total_correct_predictions)
      # input()
      label_preds = tf.argmax(label_scores, axis=2)
      # print("label preds ", label_preds)
      label_preds = self._flatten(label_preds)
      # print("label preds ", label_preds)
      # input()
      # Flatten the label scores to (batch_size*seq_len, n_classes) (i.e. 340, 36).
      label_scores = self._flatten(label_scores, outer_dim=label_scores.shape[2])
      # print("label scores ", label_scores)
      # Flatten the correct labels to the shape (batch_size*seq_len, 1) (i.e. 340,1)
      # Index for the right label for each token.
      correct_labels = self._flatten(labels)
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

    return losses, total_correct_predictions, total_labels

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
    for epoch in range(1, epochs+1):
      total_correct_predictions = 0
      total_labels = 0
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
        words, pos, morph = batch["words"], batch["pos"], batch["morph"]
        dep_labels, heads = batch["dep_labels"], batch["heads"]

        # Get loss values, predictions, and correct heads/labels for this training step.
        batch_loss, total_corr, total_l = self.train_step(words=words,
                                                          pos=pos,
                                                          morph=morph,
                                                          dep_labels=dep_labels,
                                                          heads=heads)
        total_correct_predictions += total_corr
        total_labels += total_l
        losses["labels"].append(tf.reduce_sum(batch_loss["labels"]))

      # Log stats at the end of epoch
      logging.info(f"Training stats: {self.training_stats}")

      print("total correct predictions in train ", total_correct_predictions)
      print("total labels in train", total_labels)

      training_results_for_epoch = total_correct_predictions / total_labels

      loss_results_for_epoch = {
        "head_loss": tf.reduce_mean(losses["heads"]).numpy() if "heads" in self._predict else None,
        "label_loss": tf.reduce_mean(losses["labels"]).numpy() if "labels" in self._predict else None
      }
      logging.info(f"Training results after epoch {epoch}: {training_results_for_epoch}")

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

        if test_data is not None:
          test_results_for_epoch = self.test(dataset=test_data)
          if self._log_dir:
            with self.test_summary_writer.as_default():
                tf.summary.scalar("test_results", test_results_for_epoch, step=epoch)
          logging.info(f"Test results after epoch {epoch}: {test_results_for_epoch}")

      logging.info(f"Time for epoch {time.time() - start_time}")

      # Update the eval metrics based on training, test results and loss values.
      # self._update_all_metrics(
      #   train_metrics=training_results_for_epoch,
      #   loss_metrics=loss_results_for_epoch,
      #   test_metrics=test_results_for_epoch,
      # )

    return self._metrics


  def test(self, *, dataset: Dataset):
    """Tests the performance of this parser on some dataset."""
    print("Testing on the test set..")
    label_accuracy = tf.keras.metrics.Accuracy()

    # resetting test stats at the beginning.
    for key in self.test_stats:
      self.test_stats[key] = 0.0

    total_labels = 0
    total_correct_predictions = 0
    for example in dataset:
      dep_labels = example["dep_labels"][0]
      # print("dep labels ", dep_labels)
      if not any(self._label_index_to_name(index.numpy()) == "root" for index in dep_labels):
        # print(list(self._label_index_to_name(index.numpy()) for index in dep_labels))
      # else:
        continue
      tag_index = self._label_name_to_index("root")
          # keep the root value and replace all other with 0
      dep_labels = tf.multiply(dep_labels, tf.cast(dep_labels==tag_index, tf.int64))
      # print("dep labels ", dep_labels)
      # label_inputs=tf.concat(masked_labels, axis=0)
      # print("label inputs ", label_inputs)
      labels = tf.cast(tf.cast(dep_labels, tf.bool), tf.float32)
      # print("labels ", labels)
      # input()
      total_labels += np.sum(labels)
      label_scores = self.parse(example)
      # print("label scores ", label_scores)
      positive_label_predictions = label_scores[:, :, 1]
      # print("positive label preds ", positive_label_predictions)
      # input()
      positive_label_predictions = tf.argmax(positive_label_predictions, axis=1)
      # print("positive label preds ", positive_label_predictions)
      # print("correct labels argmax ", tf.argmax(labels))
      # positive_label_predictions = tf.one_hot(positive_label_predictions, labels.shape[1])
      # print("positive label preds ", positive_label_predictions)
      # input()
      correctly_predicted_roots = positive_label_predictions == tf.argmax(labels)
      total_correct_predictions += np.sum(correctly_predicted_roots)
      # print("corr pred roots ", correctly_predicted_roots)
      # print("total_correct_predictions ", total_correct_predictions)
      # input()
      label_preds = tf.argmax(label_scores, axis=2)
      # print("label preds ", label_preds)
      # input()
      label_preds = self._flatten(label_preds)
      correct_labels = tf.expand_dims(labels, 1)
      # print("label preds ", label_preds)
      # print("correct labels ", correct_labels)
      # input()
      label_accuracy.update_state(correct_labels, label_preds)
      # print("total labels ", total_labels)
      # print("total corr preds ", total_correct_predictions)


    test_results = total_correct_predictions / total_labels
    print(f"test results: {test_results}")
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


    scores = self.model({"words": words, "pos": pos,
                        "morph": morph, "labels": dep_labels}, training=False)
    return scores



class LSTMBinaryLabelingModelSoftmax(tf.keras.Model):
  """A standalone bidirectional LSTM labeler."""
  def __init__(self, *,
               word_embeddings: Embeddings,
               n_units: int = 256,
               use_pos=True,
               use_morph=True,
               name="LSTM_Labeler"):
    super(LSTMBinaryLabelingModelSoftmax, self).__init__(name=name)
    self.use_pos = use_pos
    self.use_morph = use_morph
    self.word_embeddings = layer_utils.EmbeddingLayer(
      pretrained=word_embeddings, name="word_embeddings",
      trainable=False
    )

    if self.use_pos:
      self.pos_embeddings = layer_utils.EmbeddingLayer(
        input_dim=37, output_dim=32,
        name="pos_embeddings",
        trainable=True)
    self.concatenate = layers.Concatenate(name="concat")
    self.lstm_block = LSTMBlock(n_units=n_units,
                                dropout_rate=0.3,
                                name="lstm_block"
                                )

    self.labels = layers.Dense(units=2,
                               activation="softmax",
                               name="roots")

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
        label_scores: [batch_size, seq_len, n_labels] label preds for tokens (i.e. 10, 34, n_output_classes)
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
      scores = self.labels(sentence_repr)
    else:
      sentence_repr = self.lstm_block(word_features, training=training)
      scores = self.labels(sentence_repr)
    return scores


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
    # self.lstm3 = layers.Bidirectional(layers.LSTM(
    #   units=n_units, return_sequences=return_sequences,
    #   return_state=return_state,
      # stateful=True,
    #   name="lstm3"))
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
      # out = self.dropout2(out)
      # out = self.lstm3(out)
    else:
      out = self.lstm1(input_tensor)
      out = self.lstm2(out)
      # out = self.lstm3(out)
    return out

if __name__ == "__main__":
  word_embeddings = load_models.load_word_embeddings()
  prep = load_models.load_preprocessor(word_embeddings)
  parser = BiLSTMLabelerBinarySoftmax(word_embeddings=prep.word_embeddings,
                                      n_output_classes=2,
                                      predict=["labels"],
                                      features=["words", "pos", "morph"],
                                      model_name="root_classifier",
                                      test_every=1)

  train_treebank= "tr_boun-ud-train.tfrecords"
  test_treebank = "tr_boun-ud-test.tfrecords"
  # test_treebank = None
  train_dataset, test_dataset = load_models.load_data(preprocessor=prep,
                                                      train_treebank=train_treebank,
                                                      batch_size=250,
                                                      test_treebank=test_treebank,
                                                      type="tfrecords")
  # for batch in train_dataset:
  #   print(batch)
  #   input()
  # for batch in test_dataset:
  #   print(batch)
  #   input()

  _metrics = parser.train(dataset=train_dataset, epochs=75, test_data=test_dataset)
  print(_metrics)
