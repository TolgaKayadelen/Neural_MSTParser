import logging
import numpy as np
import datetime
import os

import tensorflow as tf
from tensorflow.keras import layers, metrics, losses, optimizers

from input import embeddor
from ranker import ranker_preprocessor
from ranker import data_extractor
from parser.utils import load_models, layer_utils
from util import writer
from util.nn import nn_utils

Embeddings = embeddor.Embeddings

_MODEL_DIR = "./ranker/models"

class Ranker:
  word_embeddings=load_models.load_word_embeddings()
  prep = load_models.load_preprocessor(word_embeddings=word_embeddings)
  label_feature = next((f for f in prep.sequence_features_dict.values() if f.name == "dep_labels"), None)
  n_output_classes=label_feature.n_values

  def __init__(self, *, labeler_model_name: str, parser_model_name:str):
    self.labeler=data_extractor.load_labeler(self.prep, self.n_output_classes, labeler_model_name)
    self.parser=data_extractor.load_parser(self.prep, self.n_output_classes, parser_model_name)
    self.label_ranker = self._set_up_label_ranker()
    self._loss_fn = losses.MeanAbsoluteError()
    self._optimizer =  optimizers.Adam(0.01, beta_1=0.9, beta_2=0.9)

  def _set_up_label_ranker(self):
    label_embeddings, pos_embeddings = False, False
    for layer in self.parser.model.layers:
      if layer.name == "label_embeddings":
        label_embedding_weights = layer.get_weights()
        label_embeddings=True
        break
    if not label_embeddings:
      raise NameError("Parser doesn't have a layer named label_embeddings.")

    for layer in self.labeler.model.layers:
      if layer.name == "pos_embeddings":
        pos_embedding_weights = layer.get_weights()
        pos_embeddings=True
        break
    if not pos_embeddings:
      raise NameError("Labeler doesn't have a layer named pos_embeddings.")

    label_ranker = LabelRankerModel(word_embeddings=self.word_embeddings,
                                    pos_embedding_weights=pos_embedding_weights,
                                    label_embedding_weights=label_embedding_weights)
    label_ranker.pos_embeddings.set_weights(pos_embedding_weights)
    label_ranker.pos_embeddings.trainable=False
    label_ranker.label_embeddings.set_weights(label_embedding_weights)
    label_ranker.label_embeddings.trainable=False
    for layer in label_ranker.layers:
      # print(layer.name, layer.trainable)
      if layer.name ==  "pos_embeddings":
        assert layer.trainable == False
        for a, b in zip(layer.weights, pos_embedding_weights):
          # print(a,b)
          np.testing.assert_allclose(a, b)
      if layer.name == "label_embeddings":
        assert layer.trainable == False
        for a, b in zip(layer.weights, label_embedding_weights):
          # print(a,b)
          np.testing.assert_allclose(a, b)
    return label_ranker

  def train(self, dataset, epochs):
    logging.info("Ranker training started.")
    losses = []
    for epoch in range(1, epochs+1):
      logging.info(f"\n\n{'->' * 12} Training Epoch: {epoch} {'<-' * 12}\n\n")
      for step, batch in enumerate(dataset):
        rewards = batch["hypo_reward"]
        scores, loss = self.train_step(batch, rewards)
        # print("scores ", scores)
        # print("loss ", loss)
        losses.append(loss)
      print("step ", step)
      print("Epoch loss ", tf.reduce_mean(losses))

  def train_step(self, inputs, rewards):
    # print("rewards ", rewards)
    with tf.GradientTape() as tape:
      scores = self.label_ranker(inputs)
      loss = self._loss_fn(rewards, scores)
    grads = tape.gradient(loss, self.label_ranker.trainable_weights)
    self._optimizer.apply_gradients(zip(grads, self.label_ranker.trainable_weights))
    return scores, loss

  def save_weights(self, suffix: int=0):
    """Saves the model weights to path in tf format."""
    model_name = self.label_ranker.name
    try:
      path = os.path.join(_MODEL_DIR, self.label_ranker.name)
      if suffix > 0:
        path += str(suffix)
        model_name = self.label_ranker.name+str(suffix)
      os.mkdir(path)
      self.label_ranker.save_weights(os.path.join(path, model_name), save_format="tf")
      logging.info(f"Saved model to  {path}")
    except FileExistsError:
      logging.warning(f"A model with the same name exists, suffixing {suffix+1}")
      self.save_weights(suffix=suffix+1)


class LabelRankerModel(tf.keras.Model):
  def __init__(self, *,
               word_embeddings: Embeddings,
               pos_embedding_weights,
               label_embedding_weights,
               name="label_ranker"):
    super(LabelRankerModel, self).__init__(name=name)

    self.word_embeddings = layer_utils.EmbeddingLayer(
      pretrained=word_embeddings,
      name="word_embeddings")

    self.pos_embeddings = layer_utils.EmbeddingLayer(
      input_dim=37, output_dim=32,
      name="pos_embeddings")

    self.label_embeddings = layer_utils.EmbeddingLayer(input_dim=43,
                                                       output_dim=50,
                                                       name="label_embeddings")
    self.word_embeddings.trainable=False
    self.concatenate = layers.Concatenate(name="concat")
    self.dense1 = layers.Dense(512, name="densor1")
    self.dense2 = layers.Dense(256, name="densor2")
    self.dense3 = layers.Dense(128, name="densor3")
    self.dense4 = layers.Dense(64, name="densor4")
    self.dense5 = layers.Dense(32, name="densor5")
    self.dense6 = layers.Dense(16, name="densor6")
    self.dense7 = layers.Dense(8, name="densor7")
    self.dense8 = layers.Dense(1, name="output")

  def _flatten_last_dim(self, _tensor):
    """Creates a [batch_size, shape[1]*shape[2] dim tensor from one that is [batch_Size, shape[1], shape[2]."""
    shape = _tensor.shape
    if len(shape) < 3:
      raise ValueError(f"Tensor should be 3 dimensional. Received a tensor of rank {len(shape)}")
    return tf.reshape(_tensor, shape=(shape[0], shape[1]*shape[2]))

  def call(self, inputs, training=True):
    word_feats = self.word_embeddings(inputs["word_id"])
    pos_feats = self.pos_embeddings(inputs["pos_id"])
    hypothesized_label = self.label_embeddings(inputs["hypo_label_id"])
    # print("word features ", word_feats,
    #       "pos_features ", pos_feats,
    #       "label features ", hypothesized_label)
    # input()

    next_token_word_feats = self.word_embeddings(inputs["next_token_ids"])
    # this has shape (batch_size, 2, 300). so it is flattened to (batch_size, 600)
    next_token_word_feats = self._flatten_last_dim(next_token_word_feats)
    # print("next token feats reshaped ", next_token_word_feats)
    # input()
    prev_token_word_feats = self.word_embeddings(inputs["prev_token_ids"])
    prev_token_word_feats = self._flatten_last_dim(prev_token_word_feats)
    # print("prev token feats reshaped ", prev_token_word_feats)

    next_token_pos_feats = self.pos_embeddings(inputs["next_token_pos_ids"])
    next_token_pos_feats = self._flatten_last_dim(next_token_pos_feats)
    # print("next_token_pos_feats ", next_token_pos_feats)

    prev_token_pos_feats = self.pos_embeddings(inputs["prev_token_pos_ids"])
    prev_token_pos_feats = self._flatten_last_dim((prev_token_pos_feats))
    # print("prev token pos feats ", prev_token_pos_feats)
    # input()

    concat = self.concatenate([word_feats, pos_feats, inputs["case"], inputs["person"], inputs["voice"], inputs["verbform"],
                               prev_token_word_feats, next_token_word_feats, next_token_pos_feats, prev_token_pos_feats,
                               hypothesized_label])
    # print("concat ", concat)
    # print("hypothesized label ", hypothesized_label)
    # input()
    output = self.dense1(concat)
    output = self.dense2(output)
    output = self.dense3(output)
    output = self.dense4(output)
    output = self.dense5(output)
    output = self.dense6(output)
    output = self.dense7(output)
    output = self.dense8(output)

    return output

if __name__ == "__main__":
  data_path = "./ranker/data/tr_boun-ud-dev-ranker-data-rio.tfrecords"
  dataset =  ranker_preprocessor.read_dataset_from_tfrecords(data_path, batch_size=500)
  labeler_model_name="bilstm_labeler_topk"
  parser_model_name="label_first_gold_morph_and_labels"
  ranker = Ranker(labeler_model_name=labeler_model_name, parser_model_name=parser_model_name)
  ranker.train(dataset=dataset, epochs=1000)
  ranker.save_weights()
  # word_embeddings = load_models.load_word_embeddings()
  # prep = load_models.load_preprocessor(word_embeddings=word_embeddings)
  # label_feature = next(
  #   (f for f in prep.sequence_features_dict.values() if f.name == "dep_labels"), None)
  # n_output_classes=label_feature.n_values
  # labeler=data_extractor.load_labeler(prep, n_output_classes, labeler_model_name)
  # parser=data_extractor.load_parser(prep, n_output_classes, parser_model_name)
  # for layer in parser.model.layers:
  #   if layer.name == "label_embeddings":
  #     label_embedding_weights = layer.get_weights()

  # for layer in labeler.model.layers:
  #   if layer.name == "pos_embeddings":
  #     pos_embedding_weights = layer.get_weights()

  # ranker = LabelRanker(word_embeddings=word_embeddings,
  #                      pos_embedding_weights=pos_embedding_weights,
  #                     label_embedding_weights=label_embedding_weights)
  # print(ranker)
  # print("Ranker Layers")
  # for layer in ranker.layers:
  #   print(layer.name, layer.trainable)
  # for batch in dataset:
  #   output = ranker(batch)
  #   print("output score is ", output)
  # ranker.train(dataset),