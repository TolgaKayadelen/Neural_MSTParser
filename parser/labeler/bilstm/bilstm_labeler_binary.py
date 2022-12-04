"""The binary bilstm labeler learns to predict one label at a time, e.g. it only learns to predict roots."""


import os
import logging
import tensorflow as tf
import numpy as np

from typing import Dict, Tuple
from parser.nn import base_parser
from proto import metrics_pb2
from input import embeddor
from tensorflow.keras import layers, metrics, losses, optimizers
from parser.nn import layer_utils, load_models
from util.nn import nn_utils
from tagset.dep_labels import dep_label_enum_pb2 as dep_label_tags
from util import writer

Embeddings = embeddor.Embeddings
Dataset = tf.data.Dataset

class BiLSTMLabelerBinary(base_parser.BaseParser):
  """A binary class bi-lstm labeler."""
  @property
  def _optimizer(self):
    return tf.keras.optimizers.Adam(0.001, beta_1=0.9, beta_2=0.9)


  def _training_metrics(self):
    return {
      "labels": metrics.BinaryAccuracy()
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

    return losses.BinaryCrossentropy(from_logits=False)

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
    model = LSTMBinaryLabelingModel(
      word_embeddings=self.word_embeddings,
      name=model_name,
      use_pos=self._use_pos,
      use_morph=self._use_morph,
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
    masked_labels = []
    for vector in dep_labels:
      # only keep the root label as gold
      if any(self._label_index_to_name(index.numpy()) == "root" for index in vector):
        print(list(self._label_index_to_name(index.numpy()) for index in vector))
        tag_index = self._label_name_to_index("root")
        # keep the root value and replace all other with 0
        vector = tf.multiply(vector, tf.cast(vector==tag_index, tf.int64))
        masked_labels.append(tf.expand_dims(vector, 0))
    label_inputs=tf.concat(masked_labels, axis=0)
    print("label inputs ", label_inputs)
    labels = tf.cast(tf.cast(label_inputs, tf.bool), tf.float32)
    predictions, correct, losses = {}, {}, {}
    with tf.GradientTape() as tape:
      # preds are raw (unnormalized) label scores.
      preds = self.model({"words": words, "pos": pos, "morph": morph,
                          "labels": labels}, training=True)
      preds = tf.squeeze(preds)

      print("preds ", preds)
      print("labels ", labels)
      loss = self._label_loss(labels, preds)
      print("loss ", loss)
      pad_mask = self._flatten((words != 0))
      print("pad mask ", pad_mask)
      input()
      preds_flat = tf.reshape(preds, (preds.shape[0]*preds.shape[1], 1))
      labels_flat = tf.reshape(labels, (labels.shape[0]*labels.shape[1], 1))
      print("preds-flat", preds_flat)
      print("labels-flat ", labels_flat)
      input()
      grads = tape.gradient(loss, self.model.trainable_weights)

    self._optimizer.apply_gradients(zip(grads, self.model.trainable_weights))

    # Update training metrics.
    self._update_training_metrics(
      labels=labels_flat,
      label_scores=preds_flat,
      pad_mask=pad_mask)

    predicted_roots = tf.one_hot(tf.argmax(preds, axis=1), preds.shape[1])
    print("predicted roots ", predicted_roots)
    correctly_predicted_roots = tf.argmax(predicted_roots, 1) == tf.argmax(labels, 1)
    total_correct_roots = np.sum(correctly_predicted_roots)
    print("corr pred roots ", correctly_predicted_roots)
    print("total corr roots ", total_correct_roots)
    input()

    losses["labels"] = loss
    correct["labels"] = labels_flat
    predictions["labels"] = preds_flat

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
      words, pos, morph = batch["words"], batch["pos"], batch["morph"]
      dep_labels, heads = batch["dep_labels"], batch["heads"]

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
        losses["heads"].append(tf.reduce_sum(batch_loss["heads"]))
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

  return self._metrics



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

class LSTMBinaryLabelingModel(tf.keras.Model):
  """A standalone bidirectional LSTM labeler."""
  def __init__(self, *,
             word_embeddings: Embeddings,
             n_units: int = 256,
             use_pos=True,
             use_morph=True,
             name="LSTM_Labeler"):
    super(LSTMBinaryLabelingModel, self).__init__(name=name)
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
    self.lstm_block = layer_utils.LSTMBlock(n_units=n_units,
                                            dropout_rate=0.3,
                                            name="lstm_block"
                                            )

    self.labels = layers.Dense(units=1,
                               activation="sigmoid",
                               name="roots")

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
    return labels

if __name__ == "__main__":
  word_embeddings = load_models.load_word_embeddings()
  prep = load_models.load_preprocessor(word_embeddings)
  parser = BiLSTMLabelerBinary(word_embeddings=prep.word_embeddings,
                               n_output_classes=2,
                               predict=["labels"],
                               features=["words", "pos", "morph"],
                               model_name="root_classifier")

  train_treebank= "tr_boun-ud-train-random10.pbtxt"
  test_treebank = "tr_boun-ud-test-random10.pbtxt"
  train_dataset, test_dataset = load_models.load_data(preprocessor=prep,
                                                      train_treebank=train_treebank,
                                                      batch_size=2,
                                                      test_treebank=test_treebank)

  _metrics = parser.train(dataset=train_dataset, epochs=2, test_data=test_dataset)
  print(_metrics)
  # writer.write_proto_as_text(_metrics, f"./model/nn/plot/final/{parser_model_name}_metrics.pbtxt")
