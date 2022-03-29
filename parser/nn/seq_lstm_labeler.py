"""In this sequential version of the LSTM labeler, we pass (embedding
vector of) the label  predicted in the previous timestep as input
(together with other features) to the next timestep.

In test time, tokens are labeled one by one rather than all in once, hence
the name sequential_lstm_labeler."""


import os
import logging
import tensorflow as tf
import numpy as np
from parser.nn import layer_utils

from input import embeddor
from parser.nn import base_parser
from proto import metrics_pb2
from tagset.dep_labels import dep_label_enum_pb2 as dep_label_tags
from tensorflow.keras import layers, metrics, losses, optimizers
from typing import Dict, Tuple

Dataset = tf.data.Dataset
Embeddings = embeddor.Embeddings

class SeqLSTMLabelingModel(tf.keras.Model):
  """Sequential LSTM labeler."""
  def __init__(self, *,
               word_embeddings: Embeddings,
               n_units: int = 100,
               n_output_classes: int,
               use_pos=True,
               use_morph=True,
               use_previous_token_label=False,
               return_lstm_output=False,
               name="LSTM_Labeler"):
    super(SeqLSTMLabelingModel, self).__init__(name=name)
    self.use_pos = use_pos
    self.use_morph = use_morph
    self.use_previous_token_label = use_previous_token_label
    self.dep_label_keys = list(dep_label_tags.Tag.DESCRIPTOR.values_by_name.keys())
    self.word_embeddings = word_embeddings
    self.word_embeddings_layer = layer_utils.EmbeddingLayer(
      pretrained=word_embeddings, name="word_embeddings_layer"
    )
    self.return_lstm_output = return_lstm_output
    if self.use_pos:
      self.pos_embeddings = layer_utils.EmbeddingLayer(
        input_dim=35, output_dim=32,
        name="pos_embeddings",
        trainable=True)
    if use_previous_token_label:
      self.label_embeddings = layer_utils.EmbeddingLayer(input_dim=36,
                                                         output_dim=50,
                                                         name="label_embeddings",
                                                         trainable=True)
    self.concatenate = layers.Concatenate(name="concat")
    self.lstm_block = layer_utils.LSTMBlock(n_units=n_units,
                                            # dropout_rate=0.3,
                                            name="lstm_block"
                                            )
    # Because in the loss function we have from_logits=True, we don't use the
    # param 'activation=softmax' in the layer. The loss function applies softmax to the
    # raw probabilites and then applies crossentropy.
    self.labels = layers.Dense(units=n_output_classes, name="labels")

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
        label_scores: [batch_size, seq_len, n_labels] label preds for tokens (i.e. 10, 34, 36)
    """
    word_inputs = inputs["words"]
    # print("word inputs ", word_inputs)
    word_features = self.word_embeddings_layer(word_inputs)
    concat_list = [word_features]
    if self.use_pos:
      pos_inputs = inputs["pos"]
      pos_features = self.pos_embeddings(pos_inputs)
      concat_list.append(pos_features)

    if self.use_morph:
      morph_inputs = inputs["morph"]
      concat_list.append(morph_inputs)

    if training and self.use_previous_token_label:
      label_inputs = inputs["labels"]
      # print("label inputs ", label_inputs)
      # slice the label inputs; remove the last token labels
      label_features = label_inputs[:, :-1]
      # print("label features ", label_features)
      # input("press to cont.")
      # create an (n,1) dimensional zero vector; n=batch_size
      # zeros = tf.expand_dims(tf.zeros_like(label_features[:, 0]), -1)
      # print("zeros ", zeros)
      # insert the zero vector to the beginning of the label_inputs
      # label_features = tf.concat([zeros, label_features], axis=1)
      # print("label features ", label_features)
      # input("press to cont.")
      # this shifted labels will be used as features while predicting
      label_features = self.label_embeddings(label_features)
      # print("label features ", label_features)
      # input("press to cont.")
      pad_ = tf.expand_dims(tf.ones_like(label_features[:, 0, :]), 1)
      # print("pad ", pad_)
      # input("press to cont.")
      label_features = tf.concat([pad_, label_features], axis=1)
      # print("label features after pad concat", label_features)
      # input("press to cont.")
      concat_list.append(label_features)
      concat = self.concatenate(concat_list)
      # print("concat in training ", concat.shape)
      # input("press to cont.")
      sentence_repr = self.lstm_block(concat)
      label_scores = self.labels(sentence_repr)
      # print(" TOP label tag ")
      # print(self.label_embeddings.get_weights()[0][dep_label_tags.TOP])
      if self.return_lstm_output:
        return {"labels": label_scores}, sentence_repr
      else:
        return {"labels": label_scores}

    elif not training and self.use_previous_token_label:
      # logging.info("Running model in test mode!")
      label_preds = []
      label_embeddings = self.label_embeddings.get_weights()
      tokens = [self.word_embeddings.itos(idx=index.numpy()) for index in word_inputs[0]]
      # print("tokens ", tokens)
      # input("press to cont.")
      batch_size, sequence_length = word_inputs.shape
      # print("seq len ", sequence_length)
      concat = self.concatenate(concat_list)
      # print("concat in test ", concat)
      # input("press to cont.")
      for i in range(1, sequence_length):
        # print("word inputs ", word_inputs)
        word = word_inputs[0, i]
        # print(f"word is {word} and token is {tokens[i]}")
        # input("press to cont.")
        # we start from the first (not 0th) token.
        token_slice = concat[:, i, :]
        # print("token slice ", token_slice)
        # input("press to cont.")
        # validata that we are capturing the embeddings for the right word
        # word_embeddings = self.word_embeddings.get_weights()
        # print(word_embeddings[0][word])
        if i == 1:
          label_emb = tf.expand_dims(label_embeddings[0][dep_label_tags.TOP], 0)
        token_input = tf.concat([token_slice, label_emb], axis=1)
        # print("token input ", token_input)
        # input("press to cont.")
        # we convert from [1,448] to [1,1,448] because the layer expects a batch_size.
        lstm_out = self.lstm_block(tf.expand_dims(token_input, 0))
        label_score = self.labels(lstm_out)

        # get the predicted labels info and embedding
        predicted_label = tf.argmax(label_score, axis=2)[0][0]
        label_preds.append(predicted_label.numpy())
        # print(self.dep_label_keys)
        # print("predicted_label ", predicted_label.numpy())
        # print(self.dep_label_keys[predicted_label])
        # input("press to cont.")

        label_emb = tf.expand_dims(label_embeddings[0][predicted_label], 0)
        # print("pred label emb ", label_emb)
        # input("press to cont.")
      return label_preds


class SeqLSTMLabeler(base_parser.BaseParser):
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
    morph_inputs = tf.keras.Input(shape=(None, 66), name="morph")
    label_inputs = tf.keras.Input(shape=(None,), name="labels")
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
    super()._parsing_model(model_name, sequential=True)
    print(f"""Using features pos: {self._use_pos}, morph: {self._use_morph}, 
           previous token label {self._use_dep_labels}""")
    model = SeqLSTMLabelingModel(
      n_output_classes=self._n_output_classes,
      word_embeddings=self.word_embeddings,
      name=model_name,
      use_pos=self._use_pos,
      use_morph=self._use_morph,
      use_previous_token_label=self._use_dep_labels,
      return_lstm_output=True,
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
    if "heads" in self._predict:
      raise ValueError("Cannot predict heads using dependency labeler.")

    predictions, correct, losses = {}, {}, {}
    with tf.GradientTape() as tape:
      scores, lstm_output = self.model({"words": words, "pos": pos, "morph": morph,
                                        "labels": dep_labels}, training=True)
      label_scores = scores["labels"]
      print("label scores ", label_scores)
      input("press to cont.")
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
      label_preds = tf.convert_to_tensor(self.parse(example, training=False),
                                         dtype=tf.int64)
      label_preds = tf.expand_dims(label_preds, 0)
      # print("label preds ", label_preds)
      correct_labels = example["dep_labels"][:, 1:]
      # print("correct labels ", correct_labels)
      # input("press to cont.")
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

  def parse(self, example: Dict, training=False):
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
    words, pos, morph = (example["words"], example["pos"], example["morph"])
    if training:
      scores, lstm_output = self.model({"words": words, "pos": pos,
                                        "morph": morph},
                                        training=True)
      return scores, lstm_output
    else:
      label_preds  = self.model({"words": words, "pos": pos,
                                "morph": morph}, training=False)
      return label_preds

