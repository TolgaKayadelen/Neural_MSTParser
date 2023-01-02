import numpy as np
import tensorflow as tf
import time
import collections

from parser import base_parser
from parser.utils import layer_utils
from parser.utils import load_models
from proto import metrics_pb2
from input import embeddor
from tensorflow.keras import layers, metrics, losses, optimizers

import logging
logging.basicConfig(format='%(levelname)s : %(message)s', level=logging.INFO)

from typing import List, Dict
Embeddings = embeddor.Embeddings
Dataset = tf.data.Dataset

class LabelFirstParserTopk(base_parser.BaseParser):
  """A label first parser predicts labels before predicting heads."""
  def __init__(self, *,
               word_embeddings: Embeddings,
               preprocessor=None,
               predict: List[str],
               features: List[str] = ["words"],
               model_name: str,
               log_dir: str,
               test_every: int = 5,
               label_embedding_weights = None,
               labeler_name: str = None):
    super(LabelFirstParserTopk, self).__init__(word_embeddings=word_embeddings,
                                               predict=predict,
                                               features=features,
                                               model_name=model_name,
                                               log_dir=log_dir,
                                               test_every=test_every)
    self.label_embedding_weights = label_embedding_weights
    self.model.label_embeddings.set_weights(label_embedding_weights)
    self.model.label_embeddings.trainable=False
    for layer in self.model.layers:
      # print(layer.name, layer.trainable)
      if layer.name == "label_embeddings":
        assert layer.trainable == False
        for a, b in zip(layer.weights, label_embedding_weights):
          np.testing.assert_allclose(a, b)

    logging.info("Set model label embedding weights from pretrained label weights!")
    self.prep = preprocessor
    self.pretrained_labeler = self._load_labeler(labeler_name)

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

  def _load_labeler(self, labeler_name):
    return load_models.load_labeler(labeler_name, self.prep)

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
    model = LabelFirstParsingTopKModel(
      n_dep_labels=self._n_output_classes,
      word_embeddings=self.word_embeddings,
      predict=self._predict,
      name=model_name,
      use_pos=self._use_pos,
      use_morph=self._use_morph,
      use_dep_labels=self._use_dep_labels,
      one_hot_labels=self.one_hot_labels,
    )
    # model(inputs=self.inputs)
    return model


  def train(self, *,
            dataset: Dataset, epochs: int=10, test_data: Dataset = None):
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
        words, pos, morph, heads = batch["words"], batch["pos"], batch["morph"], batch["heads"]
        label_scores, _ = self.pretrained_labeler.model({"words": words, "pos": pos, "morph": morph},
                                                        training=False)
        top_scores, top_k_labels  = tf.math.top_k(label_scores, k=5)
        # print(top_scores)
        # input()
        # print(top_k_labels)
        # input()

        predictions, batch_loss, correct, pad_mask = self.train_step(words=words,
                                                                     pos=pos,
                                                                     morph=morph,
                                                                     dep_labels=top_k_labels,
                                                                     heads=heads)
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
        losses["heads"].append(tf.reduce_sum(batch_loss["heads"])) # * 1. / batch_size?

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
      if epoch % (self._test_every * 2) == 0 and epoch > 100:
        c = input("continue training")
        if c == "yes" or c == "y":
          print("continuing for 10 more epochs")
        else:
          break
    # At the end of training, parse the data with the learned weights and save it as proto.
    # self.parse_and_save(test_data)
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
    words, pos, morph = (example["words"], example["pos"], example["morph"])
    label_scores, _ = self.pretrained_labeler.model({"words": words, "pos": pos, "morph": morph},
                                                    training=False)
    top_scores, top_k_labels  = tf.math.top_k(label_scores, k=5)
    # print(top_scores, top_k_labels)
    # input("top scores, top k labels")
    scores = self.model({"words": words, "pos": pos,
                         "morph": morph, "labels": top_k_labels},
                        training=False)
    return scores

class LabelFirstParsingTopKModel(tf.keras.Model):
  """Label first parsing model predicts labels before edges."""
  def __init__(self, *,
               n_dep_labels: int,
               word_embeddings: Embeddings,
               name="Label_First_Parsing_TopK_Model",
               predict: List[str],
               use_pos:bool = True,
               use_morph:bool=True,
               use_dep_labels:bool=True,
               one_hot_labels: bool = False):
    super(LabelFirstParsingTopKModel, self).__init__(name=name)
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
        input_dim=37, output_dim=32,
        name="pos_embeddings",
        trainable=True)

    if self.use_dep_labels:
      self.label_embeddings = layer_utils.EmbeddingLayer(input_dim=43,
                                                         output_dim=50,
                                                         name="label_embeddings",
                                                         trainable=False)
    self.reshape_labels = tf.keras.layers.Reshape((-1, 250), input_shape=(5, 50))

    self.concatenate = layers.Concatenate(name="concat")
    self.lstm_block = LSTMBlock(n_units=256,
                                dropout_rate=0.3,
                                name="lstm_block")
   #  self.attention = layer_utils.Attention()

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
    # print("word features ", word_features)
    # input()
    if self.use_pos:
      pos_inputs = inputs["pos"]
      pos_features = self.pos_embeddings(pos_inputs)
      concat_list.append(pos_features)
      # print("pos features ", pos_features)
      # input()
    if self.use_morph:
      morph_inputs = inputs["morph"]
      concat_list.append(morph_inputs)
      # print("morph features ", morph_inputs)
      # input()
    label_inputs = inputs["labels"]
    label_features = self.label_embeddings(label_inputs)
    # print("label features ", label_features)
    # input()
    label_features_reshaped = self.reshape_labels(label_features)
    if tf.rank(label_features_reshaped) > 3:
      label_features_reshaped = tf.squeeze(label_features_reshaped)
    # print(label_features_reshaped)
    # input("label features reshaped")
    concat_list.append(label_features_reshaped)
    sentence_repr = self.concatenate(concat_list)
    # print("sentence repr ", sentence_repr)
    # input()
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