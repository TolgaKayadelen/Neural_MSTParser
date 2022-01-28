import os
import logging
import time

import tensorflow as tf
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

  def train_step(self, words, pos, morph, dep_labels, heads):
    """Runs one training step."""
    with tf.GradientTape() as tape:
      loss, parent_prob_table = self.model({"words": words, "pos": pos, "morph": morph,
                                             "labels": dep_labels,
                                              "heads": heads}, training=True)
    # Compute gradients and apply backprop.
    grads = tape.gradient(loss, self.model.trainable_weights)
    self._optimizer.apply_gradients(zip(grads, self.model.trainable_weights))

    # Update training metrics
    # print("parent prob table ", parent_prob_table)
    # Get the predictions
    _preds = tf.math.top_k(parent_prob_table)
    preds = tf.cast(_preds.indices, tf.int64)

    # Get the correct heads
    correct_heads = self._flatten(heads[:, 1:]) # slicing the 0th token out.
    # print("correct heads ", correct_heads)
    self._update_training_metrics(
      heads=correct_heads,
      head_scores=parent_prob_table)


    return loss, correct_heads, preds, parent_prob_table


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

        sentence_loss, correct_heads, predictions, parent_prob_table = self.train_step(
          words=batch["words"], pos=batch["pos"], morph=batch["morph"],
          dep_labels=batch["dep_labels"], heads=batch["heads"]
        )
        n_tokens , _ = correct_heads.shape
        epoch_loss += sentence_loss / n_tokens
        # print("predictinos ", predictions)
        correct_predictions_dict = self._correct_predictions(
          head_predictions = predictions,
          correct_heads = correct_heads
        )
        # print("correct predictions dict ", correct_predictions_dict)
        # print("n tokens ", n_tokens)
        self._update_correct_prediction_stats(correct_predictions_dict, n_tokens)

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

      if (epoch % 10 == 0 or epoch == epochs) and test_data is not None:
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
      parent_prob_table = self.parse(sentence)

      # Get predictions
      _preds = tf.math.top_k(parent_prob_table)
      predictions = tf.cast(_preds.indices, tf.int64)

      # Get correct heads
      heads = sentence["heads"]
      correct_heads = self._flatten(heads[:, 1:]) # slicing 0th token out.
      accuracy.update_state(correct_heads, predictions)

      # Update stats
      correct_predictions_dict = self._correct_predictions(
        head_predictions=predictions,
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
    words, pos, morph, dep_labels, heads = (example["words"], example["pos"],
                                            example["morph"], example["dep_labels"],
                                            example["heads"])
    _, parent_prob_table = self.model({"words" : words,
                                       "pos" : pos,
                                       "morph" : morph,
                                       "labels" : dep_labels,
                                       "heads" : heads}, training=False)
    return parent_prob_table




# From the model we get the parent probabilities.
class SequentialParsingModel(tf.keras.Model):
  def __init__(self, *, word_embeddings: Embeddings,
               name="SequentialParser",
               use_pos: bool = True,
               use_morph: bool = True,
               use_dep_labels: bool = False,
               ):
    super(SequentialParsingModel, self).__init__(name=name)
    self.use_pos = use_pos
    self.use_morph = use_morph
    self.use_dep_labels = use_dep_labels,
    self.bilstm_output_size = 12
    # we pass logits to the Cross Entropy
    self.loss_function = losses.SparseCategoricalCrossentropy(from_logits=True)

    self.word_embeddings = layer_utils.EmbeddingLayer(
      pretrained=word_embeddings, name="word_embeddings"
    )

    if self.use_pos:
      self.pos_embeddings = layer_utils.EmbeddingLayer(
        input_dim=35, output_dim=32,
        name="pos_embeddings",
        trainable=True
      )

    self.concatenate = layers.Concatenate(name="concat")
    self.encoder = layer_utils.LSTMBlock(n_units=6,
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
    word_features = self.word_embeddings(word_inputs)
    batch_size, sequence_length = word_inputs.shape
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
      concat_list.append(label_inputs)
    if len(concat_list) > 1:
      sentence_repr = self.concatenate(concat_list)
    else:
      sentence_repr = word_features

    sentence_repr = self.encoder(sentence_repr)
    # print("sentence repr ", sentence_repr)
    parent_prob_table = []
    for i in range(1, sequence_length):
      dependant_slice = sentence_repr[:, i, :]
      dependant = tf.expand_dims(tf.ones([sequence_length, 1]) * dependant_slice, 0)
      # print("dependant ", dependant_slice)
      mask = tf.expand_dims(tf.reduce_all(tf.equal(sentence_repr, dependant), axis=2), -1)

      # computing head probability
      # these are not normalized (softmaxed) values because the loss function normalizes them.
      # the below computation computes the associative score between source and target.
      head_probs = self.v_a_inv(
        tf.nn.tanh(
          self.u_a(sentence_repr) + self.w_a(dependant))
      )
      # Apply 0 to the case where the candidate head is the token itself.
      head_probs = tf.squeeze(tf.where(mask, -1e4, head_probs), -1)
      # print("head_probs", head_probs)
      heads = inputs["heads"]
      true_head = tf.expand_dims(heads[:, i], 1)
      # print("heads ", heads)
      # print("true head ", true_head)
      # print("predicted head ", tf.argmax(head_probs, axis=1))
      # Compute the loss
      # In the computation of the loss, we leave out the 0th token as well
      # Since the loop starts from the 1st token all the time.
      loss += self.loss_function(true_head, head_probs)
      parent_prob_table.append(tf.math.exp(head_probs))
      # print("loss ", loss)
    # print("parent prob table ", parent_prob_table)
    parent_prob_table = tf.concat(parent_prob_table, 0)
    return loss, parent_prob_table

if __name__ ==  "__main__":
  embeddings = nn_utils.load_embeddings()
  word_embeddings = embeddor.Embeddings(
    name="word2vec", matrix=embeddings
  )
  prep = preprocessor.Preprocessor(
    word_embeddings=word_embeddings,
    features=["words", "pos", "morph", "heads", "dep_labels"],
    labels="heads",
    one_hot_features=["dep_labels"]
  )

  parser = SequentialParser(
    word_embeddings=prep.word_embeddings,
    predict=["heads"],
    features=["words", "pos", "morph", "dep_labels"],
    model_name="Sequential_Parser"
  )
  # print("parser ", parser)
  _DATA_DIR="data/UDv23/Turkish/training"
  _TEST_DATA_DIR="data/UDv23/Turkish/test"
  train_treebank="treebank_train_0_10.pbtxt"
  test_treebank = "treebank_test_0_3_gold.pbtxt"
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
  # for batch in dataset:
  #  print(batch["heads"])
  metrics = parser.train(dataset=dataset, test_data=test_dataset, epochs=70)
  print(metrics)
