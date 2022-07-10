import os
import logging
import time
import collections
import datetime

import tensorflow as tf
import numpy as np
from parser.nn import base_parser, architectures, layer_utils
from util.nn import nn_utils
from input import embeddor, preprocessor
from tensorflow.keras import layers, metrics, losses, optimizers

Embeddings = embeddor.Embeddings
Dataset = tf.data.Dataset

class SequentialParser(base_parser.BaseParser):
  """The sequential parser is an MST parser which predicts head for a token in sequence."""

  @property
  def _optimizer(self):
    return tf.keras.optimizers.Adam(0.001, beta_1=0.9, beta_2=0.9)

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
    raise NotImplementedError("No label loss function for this parser!")

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

    model = SequentialParsingModel(
      word_embeddings = self.word_embeddings,
      name=model_name,
      use_pos = self._use_pos,
      use_morph = self._use_morph,
      use_dep_labels = self._use_dep_labels
    )
    # model(inputs=self.inputs)
    return model

  def train_step(self, words, pos, morph, dep_labels, heads, sent_ids=None):
    """Runs one training step."""
    with tf.GradientTape() as tape:
      loss, parent_prob_dict =  self.model({"words": words, "pos": pos, "morph": morph,
                                              "labels": dep_labels,
                                              "heads": heads, "sent_ids": sent_ids}, training=True)


    # Compute gradients and apply backprop.
    grads = tape.gradient(loss, self.model.trainable_weights)
    self._optimizer.apply_gradients(zip(grads, self.model.trainable_weights))

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

      if (epoch % 3 == 0 or epoch == epochs) and test_data is not None:
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
      parent_prob_dict= self.parse(sentence)

     #  pad_mask = (sentence["words"] != 0)[:, 1:]
     #  pad_mask = self._flatten(pad_mask)

      # Get preds
      parent_prob_table = list(parent_prob_dict.values())
      _preds = [tf.math.top_k(prob_table).indices for prob_table in parent_prob_table]
      preds = tf.cast(tf.concat(_preds, 0), tf.int64)
      parent_prob_table = tf.concat(parent_prob_table, 0)


      # Get correct heads
      heads = sentence["heads"]
      correct_heads = self._flatten(heads[:, 1:]) # slicing the 0th token out.
      accuracy.update_state(correct_heads, parent_prob_table)

      # Update stats
      correct_predictions_dict = self._correct_predictions(
        head_predictions=preds,
        correct_heads=correct_heads
      )

      n_tokens, _ = correct_heads.shape
      self._update_correct_prediction_stats(
        correct_predictions_dict, n_tokens,
        stats="test"
      )

    # Compute metrics
    test_results = self._compute_metrics(stats="test")
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
                                       "sent_ids": sent_ids}, training=False)
    return parent_prob_dict




# From the model we get the parent probabilities.
class SequentialParsingModel(tf.keras.Model):
  def __init__(self, *, word_embeddings: Embeddings,
               name="SequentialParser",
               use_pos: bool = True,
               use_morph: bool = True,
               use_dep_labels: bool = True,
               use_label_embeddings=False,
               ):
    super(SequentialParsingModel, self).__init__(name=name)
    self.use_pos = use_pos
    self.use_morph = use_morph
    self.use_dep_labels = use_dep_labels
    self.use_label_embeddings = use_label_embeddings
    self.bilstm_output_size = 512
    # we pass logits to the Cross Entropy
    self.loss_function = losses.SparseCategoricalCrossentropy(from_logits=True)

    self.word_embeddings = layer_utils.EmbeddingLayer(
      pretrained=word_embeddings, name="word_embeddings"
    )

    if self.use_pos:
      self.pos_embeddings = layer_utils.EmbeddingLayer(
        input_dim=37, output_dim=32,
        name="pos_embeddings",
        trainable=True
      )

    if self.use_dep_labels and self.use_label_embeddings:
      self.label_embeddings = layer_utils.EmbeddingLayer(
        input_dim=43, output_dim=50, name="label_embeddings", trainable=True
      )

    self.concatenate = layers.Concatenate(name="concat")
    self.encoder = layer_utils.LSTMBlock(n_units=256,
                                         dropout_rate=0.3,
                                         name="lstm_encoder")

    # here we define the perceptrons of the model
    self.u_a = layers.Dense(self.bilstm_output_size, activation=None)
    self.w_a = layers.Dense(self.bilstm_output_size, activation=None)
    self.v_a_inv = layers.Dense(1, activation=None, use_bias=False,
                                name="v_a_inv")

  def call(self, inputs, training=True):
    """Forward pass."""
    word_inputs = inputs["words"]
    sent_ids = inputs["sent_ids"]
    sent_ids = sent_ids[:, :1]
    pad_mask = (word_inputs == 0)
    word_features = self.word_embeddings(word_inputs)
    batch_size, sequence_length = word_inputs.shape
    # logging.info(f"batch size = {batch_size}, seq_len = {sequence_length}")
    concat_list = [word_features]
    loss = 0.0
    if self.use_pos:
      pos_inputs = inputs["pos"]
      pos_features = self.pos_embeddings(pos_inputs)
      concat_list.append(pos_features)
    if self.use_morph:
      morph_inputs = inputs["morph"]
      concat_list.append(morph_inputs)
    if self.use_dep_labels:
      label_inputs = inputs["labels"]
      if self.use_label_embeddings:
        label_features = self.label_embeddings(label_inputs)
        concat_list.append(label_features)
      else:
        concat_list.append(label_inputs)
    if len(concat_list) > 1:
      sentence_repr = self.concatenate(concat_list)
    else:
      sentence_repr = word_features

    sentence_repr = self.encoder(sentence_repr)
    # this will be a dict of arrays
    parent_prob_dict = collections.defaultdict(list)

    for i in range(1, sequence_length):
      dependant_slice = tf.expand_dims(sentence_repr[:, i, :], 1)
      # print("sentence repr ", sentence_repr)
      # print("dependency slice ", dependant_slice)
      # input("press to cont.")


      tile = tf.constant([1, sequence_length, 1])
      # print("tile is  ", tile)
      # input("press  to cont.")
      dependant = tf.tile(dependant_slice, tile)
      # print("dependant ", dependant)
      # input("press")
      # print("sentence repr ", sentence_repr)
      # input("press")

      head_mask = tf.expand_dims(tf.reduce_all(tf.equal(sentence_repr, dependant), axis=2), -1)
      # print("head mask is ", head_mask)
      # input("press")
      # computing head probability
      # these are not normalized (softmaxed) values because the loss function normalizes them.
      # the below computation computes the associative score between source and target.
      head_probs = self.v_a_inv(
        tf.nn.tanh(
          self.u_a(sentence_repr) + self.w_a(dependant))
      )
      # print("head probs before ", head_probs)
      # Apply 0 to the case where the candidate head is the token itself.
      head_probs = tf.squeeze(tf.where(head_mask, -1e4, head_probs), -1)
      # print("head probs after ", head_probs)
      # input("press")
      # Also apply 0 to the padded tokens
      if batch_size > 1:
        head_probs = tf.where(pad_mask, -1e4, head_probs)
      # print("head_probs after applying mask", head_probs)
      true_heads = tf.expand_dims(inputs["heads"][:, i], 1)
      # print("true heads ", true_heads)

      # Compute the loss
      # In the computation of the loss, we leave out the 0th token as well
      # Since the loop starts from the 1st token all the time.
      loss += self.loss_function(true_heads, head_probs)
      if batch_size > 1:
        # unpadded_tokens = tf.ragged.boolean_mask(
        #  head_probs, mask=tf.math.logical_not(pad_mask))
        # print("unpadded tokens ", unpadded_tokens)
        for sent_id, token in zip(sent_ids, head_probs): # unpadded_tokens):
          # print("slice is ", sent_id, token)
          parent_prob_dict[sent_id.numpy()[0]].append(tf.expand_dims(tf.math.exp(token), 0))
      else:
        sent_id = sent_ids[0].numpy()[0]
        parent_prob_dict[sent_id].append(tf.math.exp(head_probs))

    parent_dict= {}
    for key in parent_prob_dict.keys():
      parent_dict[key] = tf.concat(parent_prob_dict[key], 0)
    # print("parent prob dict at last", parent_dict)

    # print("loss is: ", loss)
    return loss, parent_dict

if __name__ ==  "__main__":
  current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
  log_dir = "debug/sequential_parser/" + current_time
  embeddings = nn_utils.load_embeddings()
  word_embeddings = embeddor.Embeddings(
    name="word2vec", matrix=embeddings
  )
  prep = preprocessor.Preprocessor(
    word_embeddings=word_embeddings,
    features=["words", "pos", "morph", "heads", "dep_labels", "sent_id"],
    labels="heads",
    one_hot_features=["dep_labels"]
  )

  parser = SequentialParser(
    word_embeddings=prep.word_embeddings,
    predict=["heads"],
    features=["words", "pos", "morph", "dep_labels", "sent_id"],
    log_dir=log_dir,
    test_every=2,
    model_name="Sequential_Parser_Batch"
  )
  # print("parser ", parser)
  _DATA_DIR="data/UDv29/train/tr"
  _TEST_DATA_DIR="data/UDv29/test/tr"
  train_treebank="tr_boun-ud-train.pbtxt"
  test_treebank = "tr_boun-ud-test.pbtxt"
  train_sentences = prep.prepare_sentence_protos(
    path=os.path.join(_DATA_DIR, train_treebank))
  test_sentences = prep.prepare_sentence_protos(
    path=os.path.join(_TEST_DATA_DIR, test_treebank)
  )
  dataset = prep.make_dataset_from_generator(
    sentences=train_sentences,
    batch_size=50)
  test_dataset = prep.make_dataset_from_generator(
    sentences=test_sentences,
    batch_size=1
  )
  # for batch in dataset:
  #  print(batch["heads"])
  metrics = parser.train(dataset=dataset, test_data=test_dataset, epochs=100)
  print(metrics)
