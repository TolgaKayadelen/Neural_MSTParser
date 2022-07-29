import os
import logging
import datetime
import time

import tensorflow_addons as tfa
import tensorflow as tf
import numpy as np

from input import preprocessor, embeddor
from util.nn import nn_utils
from parser.nn import base_parser
from parser.nn.combined_model import CombinedParsingModel
from parser.nn.combined_parser_memory import Memory
from tensorflow.keras import metrics
from typing import List, Dict, Tuple

Embeddings = embeddor.Embeddings
Dataset = tf.data.Dataset

class CombinedParser(base_parser.BaseParser):
  """The sequential parser is an MST parser which predicts head for a token in sequence."""

  def __init__(self, *,
               word_embeddings: Embeddings,
               n_output_classes: int = None,
               predict: List[str],
               features: List[str] = ["words"],
               model_name: str,
               log_dir: str,
               test_every: int = 10):
    super(CombinedParser, self).__init__(word_embeddings=word_embeddings,
                                             n_output_classes=n_output_classes,
                                             predict=predict,
                                             features=features,
                                             model_name=model_name,
                                             log_dir=log_dir,
                                             test_every=test_every)
    self.memory = Memory(memory_size=50, batch_size=10)
    # print("self model ", self.model)
    # print([layer.name for layer in self.model.layers])
    self.labeler_layers = self.model.layers[0:-3]
    self.parser_layers = self.model.layers[-3:]
    self.parser_optimizer = tf.keras.optimizers.Adam(0.001, beta_1=0.9, beta_2=0.9)
    self.labeler_optimizer = tf.keras.optimizers.RMSprop(0.1)
    # self.optimizers_and_layers = [
    #   (self.parser_optimizer, self.parser_layers),
    #   (self.labeler_optimizer, self.labeler_layers)
    # ]
    # print("parser_layers ", [name for name in self.parser_layers])
    # print("labeler_layers ", [name for name in self.labeler_layers])
    # self.optimizer = tfa.optimizers.MultiOptimizer(self.optimizers_and_layers)

    # print("optimizers and layers ", self.optimizers_and_layers)
    # input("press to cont.")

  @property
  def _optimizer(self):
    pass

  def _training_metrics(self):
    return {
      "heads" : metrics.SparseCategoricalAccuracy(),
    }

  @property
  def _head_loss_function(self):
    """Returns loss per token head prediction."""
    raise NotImplementedError("The parsing model should implement this!")

  @property
  def _label_loss_function(self):
    """Loss function to compute state_action value loss."""
    return tf.keras.losses.MeanSquaredError()

  @property
  def inputs(self):
    word_inputs = tf.keras.Input(shape=(None, ), name="words")
    pos_inputs = tf.keras.Input(shape=(None, ), name="pos")
    morph_inputs = tf.keras.Input(shape=(None, 66), name="morph")
    label_inputs = tf.keras.Input(shape=(None, ), name="labels")
    input_dict = {"words": word_inputs}

    if self._use_pos:
      input_dict["pos"] = pos_inputs
    if self._use_morph:
      input_dict["morph"] = morph_inputs
    if self._use_dep_labels:
      input_dict["labels"] = label_inputs
    print(label_inputs)
    return input_dict

  def _n_words_in_batch(self, words, pad_mask=None):
    words_reshaped = tf.reshape(words, shape=pad_mask.shape)
    return len(tf.boolean_mask(words_reshaped, pad_mask))

  def _parsing_model(self, model_name):
    super()._parsing_model(model_name)
    print(f"""Using features
      pos : {self._use_pos}, morph: {self._use_morph} and dep_labels {self._use_dep_labels}""")

    model = CombinedParsingModel(
      word_embeddings = self.word_embeddings,
      n_output_classes=self._n_output_classes,
      name=model_name,
      use_pos = self._use_pos,
      use_morph = self._use_morph,
    )
    # model(inputs=self.inputs)
    # for layer in model.layers:
    #   print("layer name: ", layer.name)
    #   print("trainable: ", layer.trainable)
    return model

  def train_step(self, words, pos, morph, dep_labels, heads, sent_ids=None):
    """Runs one training step."""

    with tf.GradientTape() as parser_tape, tf.GradientTape() as labeler_tape:
      head_loss, parent_prob_dict, experiences, label_scores =  self.model({"words": words, "pos": pos, "morph": morph,
                                                                            "labels": dep_labels,
                                                                            "heads": heads, "sent_ids": sent_ids},
                                                                             training=True)

      # print("experiences ", experiences)
      self.memory.extend(experiences)
      # print("memory ", self.memory)
      # print("memory len ", len(self.memory))
      # input("press to cont.")
      states, actions, action_qs, rewards, actions_hot, action_names = self.memory.random_sample()
      # print("sample ", states, actions, rewards, actions_hot, action_names)
      print("action_qs: ", action_qs)
      print("actions hot ", actions_hot)
      print("rewards ", rewards)
      # in fact target_mask is the target_qs here.
      # it's just that we don't compute any loss based on non existing (zerod-out) losses.
      target_mask = tf.multiply(target_qs, tf.squeeze(actiions_hot))
      print("target mask ", target_mask)
      input("press to cont.")
      # in this setting, action_qs are label scores, rewards are target qs.
      # so the loss should be computed between these two, but only making sure we update the
      # relevant node.
      label_loss = self._label_loss_function(tf.expand_dims(rewards, 1), target_mask)
      print("label loss ", label_loss)
      input("press to cont.")

    # print("parser trainable weights ", [layer.trainable_weights for layer in self.parser_layers])
    #input("press to cont.")
    # print("labeler trainable weights ", [layer.trainable_weights for layer in self.labeler_layers])
    print("head loss ", head_loss)
    print("------------------------")
    input("will print grads now .... ")

    p_grads = parser_tape.gradient(head_loss, [layer.trainable_weights for layer in self.parser_layers])
    l_grads = labeler_tape.gradient(label_loss, [layer.trainable_weights for layer in self.labeler_layers])
    print("p grads ", p_grads)
    print("l grads ", l_grads)
    input("press to cont.")

    # Compute pad mask
    pad_mask = (words != 0)[:, 1:]
    pad_mask = self._flatten(pad_mask)

    # Get preds
    parent_prob_table = list(parent_prob_dict.values())
    _preds = [tf.math.top_k(prob_table).indices for prob_table in parent_prob_table]
    preds = tf.cast(tf.concat(_preds, 0), tf.int64)
    # print("preds from parent prob table ", preds)

    # Get the correct heads
    correct_heads = self._flatten(heads[:, 1:]) # slicing the 0th token out.

    # Get the parent_prob_table as a tensor
    parent_prob_table = tf.concat(parent_prob_table, 0)
    # print("parent prob table ", parent_prob_table)

    # Update training metrics
    self._update_training_metrics(
      heads=correct_heads,
      head_scores=parent_prob_table,
      pad_mask=pad_mask)
    return loss, correct_heads, preds, pad_mask


  def train(self, *,
            dataset: Dataset,
            epochs: int = 10,
            test_data: Dataset=None):
    for epoch in range(1, epochs+1):
      test_results_for_epoch = None
      epoch_loss = 0
      # Reset the training metrics before each epoch.
      for metric in self._training_metrics:
        self._training_metrics[metric].reset_states()

      # Reset the training stats before each epoch.
      for key in self.training_stats:
        self.training_stats[key] = 0.0

      logging.info(f"\n\n{'->' * 12} Training Epoch: {epoch} {'<-' * 12}\n\n")
      start_time = time.time()

      for step, batch in enumerate(dataset):
        loss, correct_heads, predictions, pad_mask = self.train_step(
          words=batch["words"], pos=batch["pos"], morph=batch["morph"],
          dep_labels=batch["dep_labels"], heads=batch["heads"], sent_ids=batch["sent_id"]
        )

        n_words_in_batch = np.sum(pad_mask)
        epoch_loss += loss / n_words_in_batch
        # print("words in batch ", n_words_in_batch)
        correct_predictions_dict = self._correct_predictions(
          head_predictions = predictions,
          correct_heads = correct_heads,
          pad_mask=pad_mask
        )

        self._update_correct_prediction_stats(correct_predictions_dict, n_words_in_batch)

      # Log stats at  the end of epoch
      logging.info(f"Training stats: {self.training_stats}")

      # Compute metrics
      training_results_for_epoch = self._compute_metrics()

      self._log(description=f"Training results after epoch {epoch}",
                results=training_results_for_epoch)

      self._log(description=f"Training metrics after epoch {epoch}",
                results=self._training_metrics)

      loss_results_per_epoch = {
        "head_loss": epoch_loss.numpy()
      }

      if (epoch % self._test_every == 0 or epoch == epochs) and test_data is not None:
        logging.info("Testing on test data")
        test_results_for_epoch = self.test(dataset=test_data)
        self._log(description=f"Test results after epoch {epoch}",
                  results=test_results_for_epoch)

      logging.info(f"Time for epoch {time.time() - start_time}")
      # input("press to cont.")
      self._update_all_metrics(
        train_metrics=training_results_for_epoch,
        loss_metrics=loss_results_per_epoch,
        test_metrics=test_results_for_epoch
      )

    return self._metrics

  def test(self, *, dataset: Dataset):
    """Tests the performance of the parser on the dataset."""
    accuracy = metrics.SparseCategoricalAccuracy()

    for key in self.test_stats:
      self.test_stats[key] = 0.0

    for sentence in dataset:
      words = sentence["words"]
      # print("words ", words)
      parent_prob_dict= self.parse(sentence)
      batch_size, sequence_length = words.shape[0], words.shape[1]


      if batch_size > 1:
        pad_mask = (words != 0)[:, 1:]
        pad_mask = self._flatten(pad_mask)
      else:
        pad_mask=None
      # print("pad mask ", pad_mask)

      # Get preds
      parent_prob_table = list(parent_prob_dict.values())
      _preds = [tf.math.top_k(prob_table).indices for prob_table in parent_prob_table]
      # print("_preds ", _preds)
      preds = tf.cast(tf.concat(_preds, 0), tf.int64)
      # print("preds ", preds)
      parent_prob_table = tf.concat(parent_prob_table, 0)

      # Get correct heads
      heads = sentence["heads"]
      correct_heads = self._flatten(heads[:, 1:]) # slicing the 0th token out.
      # print("correct heads ", correct_heads)
      accuracy.update_state(correct_heads, parent_prob_table)
      # Update stats
      correct_predictions_dict = self._correct_predictions(
        head_predictions=preds,
        correct_heads=correct_heads,
        pad_mask=pad_mask
      )
      # print("correct predictions dict ", correct_predictions_dict)

      if pad_mask is not None:
        n_tokens = np.sum(pad_mask)
        # print("pad mask ", pad_mask)
        logging.info(f"n_tokens in batch using pad mask: {n_tokens}")
      else:
        n_tokens, _ = correct_heads.shape
        logging.info(f"n_tokens in batch using correct heads: {n_tokens}")
      # input("press to cont.")
      self._update_correct_prediction_stats(
        correct_predictions_dict, n_tokens,
        stats="test"
      )
      print("test stats ", self.test_stats)
    # Compute metrics
    test_results = self._compute_metrics(stats="test")
    print("test results ", test_results)
    return test_results



  def parse(self, example):
    """Parses an example with this parser.

    Returns: parent_prob_table.
    """
    words, pos, morph, dep_labels, heads, sent_ids = (
      example["words"], example["pos"],
      example["morph"], example["dep_labels"],
      example["heads"], example["sent_id"])
    _, parent_prob_dict = self.model({"words" : words,
                                      "pos" : pos,
                                      "morph" : morph,
                                      "labels" : dep_labels,
                                      "heads" : heads,
                                      "sent_ids": sent_ids},
                                     training=False)
    return parent_prob_dict


if __name__ ==  "__main__":
  current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
  log_dir = "debug/combined_parser/" + current_time
  embeddings = nn_utils.load_embeddings()
  word_embeddings = embeddor.Embeddings(
    name="word2vec", matrix=embeddings
  )
  prep = preprocessor.Preprocessor(
    word_embeddings=word_embeddings,
    features=["words", "pos", "morph", "heads", "dep_labels", "sent_id"],
    labels="heads, dep_labels",
    one_hot_features=["dep_labels"]
  )
  label_feature = next(
    (f for f in prep.sequence_features_dict.values() if f.name == "dep_labels"), None)

  # print(f"label feature {label_feature}")
  # input("press to cont.")

  parser = CombinedParser(
    word_embeddings=prep.word_embeddings,
    n_output_classes=label_feature.n_values,
    predict=["heads", "labels"],
    features=["words", "pos", "morph", "heads", "dep_labels"],
    log_dir=log_dir,
    test_every=1,
    model_name="combined_parser"
  )
  # print("parser ", parser)
  _DATA_DIR="data/UDv29/train/tr"
  _TEST_DATA_DIR="data/UDv29/test/tr"

  train_treebank="tr_boun-ud-train-random500.pbtxt"
  test_treebank = "tr_boun-ud-test-random50.pbtxt"


  train_sentences = prep.prepare_sentence_protos(
    path=os.path.join(_DATA_DIR, train_treebank))
  test_sentences = prep.prepare_sentence_protos(
    path=os.path.join(_TEST_DATA_DIR, test_treebank)
  )
  dataset = prep.make_dataset_from_generator(
    sentences=train_sentences,
    batch_size=2)
  test_dataset = prep.make_dataset_from_generator(
    sentences=test_sentences,
    batch_size=1
  )
  '''
  dataset = prep.read_dataset_from_tfrecords(
    records=os.path.join(_DATA_DIR, train_treebank),
    batch_size=500
  )
  test_dataset=prep.read_dataset_from_tfrecords(
    records=os.path.join(_TEST_DATA_DIR, test_treebank),
    batch_size=1
  )
  '''

  # for batch in dataset:
  #  print(batch["heads"])
  metrics = parser.train(dataset=dataset, test_data=test_dataset, epochs=75)
  print(metrics)
  # parser.save_weights()
  # logging.info("weights saved!")


