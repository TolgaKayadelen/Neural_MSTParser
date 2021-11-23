
import os
import logging

import tensorflow as tf
import numpy as np

from parser.nn import base_parser, architectures
from util.nn import nn_utils
from util import converter
from input import embeddor, preprocessor
from proto import metrics_pb2
from tensorflow.keras import layers, metrics, losses, optimizers


logging.basicConfig(format='%(levelname)s : %(message)s', level=logging.INFO)

class BiaffineParser(base_parser.BaseParser):
  """The Biaffine Parser implementation as presented by Dozat and Manning (2018)"""

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
    return losses.SparseCategoricalCrossentropy(from_logits=True,
                                                reduction=tf.keras.losses.Reduction.NONE)

  @property
  def _label_loss_function(self):
    return losses.SparseCategoricalCrossentropy(from_logits=True,
                                                reduction=tf.keras.losses.Reduction.NONE)

  @property
  def inputs(self):
    word_inputs = tf.keras.Input(shape=(None, ), name="words")
    pos_inputs = tf.keras.Input(shape=(None, ), name="pos")
    morph_inputs = tf.keras.Input(shape=(None, 66), name="morph")
    input_dict = {"words": word_inputs}
    if self._use_pos:
      input_dict["pos"] = pos_inputs
    if self._use_morph:
      input_dict["morph"] = morph_inputs
    return input_dict

  def _arc_maps(self, heads:tf.Tensor, labels:tf.Tensor):
    """"Returns a list of tuples mapping heads, dependents and labels.

    For each sentence in a training batch, this function creates a list of
    lists with the values [batch_idx, head_idx, dep_idx, label_idx].

    In the Biaffine parser configuration, the label scores is of shape
    [batch_size, n_labels, seq_len, seq_len]. That holds the probability score
    of seeing each label in the tagset when each token x is a dependent to every
    other token y.

    Using the correct head and label indices as input, this _arc_maps function
    prepares a list of actual head and dependent maps together with the
    correct label indexes. This map is then used to strip out the probability scores
    that the model has predicted for each label for the actal heads and dependents
    in the gold data in _train_step().

    This is then used in scoring label loss.
    """
    arc_maps = []
    for sentence_idx, sentence in enumerate(heads):
      for token_idx, head in enumerate(sentence):
        arc_map = []
        arc_map.append(sentence_idx) # batch index
        arc_map.append(head.numpy()) # head index
        arc_map.append(token_idx) # dependent index
        arc_map.append(labels[sentence_idx, token_idx].numpy()) # label index
        arc_maps.append(arc_map)
    return np.array(arc_maps)

  def _n_words_in_batch(self, words, pad_mask=None):
    words_reshaped = tf.reshape(words, shape=pad_mask.shape)
    return len(tf.boolean_mask(words_reshaped, pad_mask))

  def _parsing_model(self, model_name):
    super()._parsing_model(model_name)
    model = architectures.BiaffineParsingModel(
      n_dep_labels=self._n_output_classes,
      word_embeddings=self.word_embeddings,
      predict=self._predict,
      use_pos=self._use_pos,
      use_morph=self._use_morph
    )
    return model

  def train_step(self, *,
                 words: tf.Tensor, pos: tf.Tensor, morph: tf.Tensor,
                 dep_labels: tf.Tensor, heads: tf.Tensor):
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
      pad_mask: tf.Tensor of shape (batch_size*seq_len, 1) where padded words are
        marked as 0.
    """
    predictions, correct, losses = {}, {}, {}
    arc_maps = self._arc_maps(heads, dep_labels)
    # print("arc maps ", arc_maps)
    # input("press to cont.")

    with tf.GradientTape() as tape:
      scores = self.model({"words": words, "pos": pos, "morph": morph}, training=True)

      # head scores = (batch_size, seq_len, seq_len)
      # label_scores = (batch_size, n_labels, seq_len, seq_len)
      head_scores, label_scores = scores["edges"], scores["labels"]

      # Get the predicted head indices.
      head_preds = self._flatten(tf.argmax(head_scores, axis=2))

      # Flatten the head scores to (batch_size*seq_len, seq_len) (i.e. 340, 34).
      head_scores = self._flatten(head_scores, outer_dim=head_scores.shape[2])

      # Flatten the correct heads to (batch_size*seq_len, 1)
      correct_heads = self._flatten(heads)
      pad_mask = self._flatten(words != 0)

      # Compute head loss
      head_loss = tf.expand_dims(self._head_loss(head_scores, correct_heads), axis=-1)

      # Compute label loss
      # First transpose the label scores to [batch_size, seq_len, seq_len, n_classes]
      label_scores = tf.transpose(label_scores, perm=[0,2,3,1])

      # get the logits (label prediction scores) from label scores
      # logits is of shape: (batch_size*seq_len, n_classes), i.e. (190, 36)
      logits = tf.gather_nd(label_scores, indices=arc_maps[:, :3])

      # correct labels is of shape (batch_size*seq_len, 1), i.e. (190, 1)
      correct_labels = tf.expand_dims(tf.convert_to_tensor(arc_maps[:, 3]), axis=-1)

      # get label loss
      label_loss = self._label_loss(label_scores=logits, correct_labels=correct_labels)

      # the label preds is of shape (batch_size*seq_len, 1)
      label_preds = tf.expand_dims(tf.argmax(logits, axis=1), axis=-1)

    # Compute gradients
    grads = tape.gradient([head_loss, label_loss], self.model.trainable_weights)

    self._optimizer.apply_gradients(zip(grads, self.model.trainable_weights))

    # Update training metrics
    self._update_training_metrics(
      heads=correct_heads, head_scores=head_scores,
      labels=correct_labels, label_scores=logits,
      pad_mask=pad_mask
    )

    # Fill in return values
    losses["heads"] = head_loss
    correct["heads"] = correct_heads
    predictions["heads"] = head_preds

    losses["labels"] = label_loss
    correct["labels"] = correct_labels # ??
    predictions["labels"] = label_preds

    return predictions, losses, correct, pad_mask


  def test(self, *, dataset):
    print("Testing on the test set")
    head_accuracy = tf.keras.metrics.Accuracy()
    label_accuracy = tf.keras.metrics.Accuracy()

    # resetting test stats at the beginning.
    for key in self.test_stats:
      self.test_stats[key] = 0.0

    for example in dataset:
      scores = self.parse(example)

      # Compute head accuracy
      head_scores, label_scores = scores["edges"], scores["labels"]
      head_preds = self._flatten(tf.argmax(head_scores, 2))
      correct_heads = self._flatten(example["heads"])
      head_accuracy.update_state(correct_heads, head_preds)


      # Compute label accuracy
      arc_maps = self._arc_maps(example["heads"], example["dep_labels"])
      label_scores = tf.transpose(label_scores, perm=[0,2,3,1])
      logits = tf.gather_nd(label_scores, indices=arc_maps[:, :3])
      correct_labels = tf.expand_dims(tf.convert_to_tensor(arc_maps[:, 3]), axis=-1)
      label_preds = tf.expand_dims(tf.argmax(logits, axis=1), axis=-1)
      label_accuracy.update_state(correct_labels, label_preds)


      correct_predictions_dict = self._compute_correct_predictions_in_step(
        head_predictions=head_preds,
        correct_heads=correct_heads,
        label_predictions=label_preds,
        correct_labels=correct_labels
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


if __name__ == "__main__":
  embeddings = nn_utils.load_embeddings()
  word_embeddings = embeddor.Embeddings(name="word2vec", matrix=embeddings)
  prep = preprocessor.Preprocessor(
    word_embeddings=word_embeddings,
    features=["words", "pos", "morph", "heads", "dep_labels"],
    labels=["heads", "dep_labels"]
  )
  label_feature = next(
    (f for f in prep.sequence_features_dict.values() if f.name == "dep_labels"), None)

  parser = BiaffineParser(word_embeddings=prep.word_embeddings,
                          n_output_classes=label_feature.n_values,
                          predict=["heads", "labels"],
                          features=["words", "pos", "morph"],
                          model_name="tests_biaffine_base_parser")

  _DATA_DIR="data/UDv23/Turkish/training"
  _TEST_DATA_DIR="data/UDv23/Turkish/test"
  train_treebank="treebank_train_0_50.pbtxt"
  test_treebank = "treebank_test_0_10.conllu"
  train_sentences = prep.prepare_sentence_protos(path=os.path.join(_DATA_DIR,
                                                                   train_treebank))
  dataset = prep.make_dataset_from_generator(
    sentences=train_sentences,
    batch_size=5
  )
  if test_treebank is not None:
    test_sentences = prep.prepare_sentence_protos(path=os.path.join(_TEST_DATA_DIR, test_treebank))
    test_dataset = prep.make_dataset_from_generator(
      sentences=test_sentences,
      batch_size=1
    )
  else:
    test_dataset=None
  metrics = parser.train(dataset=dataset, epochs=10, test_data=test_dataset)
  print(metrics)
