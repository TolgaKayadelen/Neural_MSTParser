import logging
import argparse
import os

import tensorflow_ranking as tfr
import tensorflow_recommenders as tfrs
import tensorflow as tf
import numpy as np
from tensorflow.keras import layers, metrics, losses, optimizers

from input import embeddor
from ranker.preprocessing import ranker_preprocessor
from ranker.eval import rank_eval
from parser.utils import load_models, layer_utils
from proto import ranker_data_pb2
from util import writer, reader

Embeddings = embeddor.Embeddings
RankerDataset = ranker_data_pb2.RankerDataset

_MODEL_DIR = "./ranker/models"

class Ranker:

  def __init__(self, *,
               word_embeddings: Embeddings,
               labeler_model_name: str = None,
               parser_model_name:str = None,
               from_disk: bool = False,
               name:str = "label_ranker"):
    self.word_embeddings=word_embeddings
    if not from_disk:
      self.prep = load_models.load_preprocessor(word_embeddings=word_embeddings)
      self.label_feature = next((f for f in self.prep.sequence_features_dict.values() if f.name == "dep_labels"), None)
      self.n_output_classes=self.label_feature.n_values
      self.labeler=load_models.load_labeler(labeler_name=labeler_model_name, prep=self.prep)
      self.parser=load_models.load_parser(parser_name=parser_model_name, prep=self.prep)
    self.label_ranker = self._set_up_label_ranker(name=name, from_disk=from_disk)
    self._loss_fn = tfr.keras.losses.PairwiseHingeLoss()
    self._ranker_prep = ranker_preprocessor.RankerPreprocessor(word_embeddings=self.word_embeddings)

  @property
  def inputs(self):
    input_dict = {}
    input_dict["word_id"] = tf.keras.Input(shape=(), name="word_id")
    input_dict["hypo_label_id"] = tf.keras.Input(shape=(), name="hypo_label_id")
    input_dict["hypo_rank"] = tf.keras.Input(shape=(), name="hypo_rank")
    input_dict["hypo_reward"] = tf.keras.Input(shape=(), name="hypo_reward")
    input_dict["pos_id"] = tf.keras.Input(shape=(), name="pos_id")
    input_dict["case"] = tf.keras.Input(shape=(8), name="case")
    input_dict["person"] = tf.keras.Input(shape=(3), name="person")
    input_dict["voice"] = tf.keras.Input(shape=(4), name="voice")
    input_dict["verbform"] = tf.keras.Input(shape=(3), name="verbform")
    input_dict["word_string"] = tf.keras.Input(shape=(), name="word_string")
    input_dict["next_token_ids"] = tf.keras.Input(shape=(2, ), name="next_token_ids")
    input_dict["next_token_pos_ids"] = tf.keras.Input(shape=(2, ), name="next_token_pos_ids")
    input_dict["prev_token_ids"] = tf.keras.Input(shape=(2, ), name="prev_token_ids")
    input_dict["prev_token_pos_ids"] = tf.keras.Input(shape=(2, ), name="prev_token_pos_ids")
    return input_dict

  def _set_up_label_ranker(self, name, from_disk=False):
    if from_disk:
      logging.info("Uploading a pretrained ranker from disk.")
      return self.load_weights(name=name)
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

    label_ranker = LabelRankerModel(word_embeddings=self.word_embeddings, name=name)
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

  def train(self, dataset, epochs, test_data, test_every, save_every=None):
    logging.info("Ranker training started.")
    if test_data:
      baseline = self.get_labeler_baseline(test_data)
      logging.info(f"Baseline on test data: {baseline}")
    ada = tf.keras.optimizers.Adagrad(learning_rate=tf.Variable(0.1))
    self.label_ranker.compile(optimizer=ada)
    for epoch in range(epochs):
      history = self.label_ranker.fit(dataset, epochs=1)
      if test_data and test_every > 0 and epochs % test_every == 0:
        test_acc = self.test(test_data)
        logging.info("Test accuracy ", test_acc)
      # keep_training = input("Continue training: y/n ?")
      # if keep_training == "n":
      #   break
    return history

  def get_labeler_baseline(self, test_data):
    """Computes how many times the top predicted label by the labeler is the highest rewarded hypothesis."""
    baseline, _, _, _ = rank_eval.rank_accuracy(test_data)
    print("Baseline on test data: ", baseline)
    return baseline

  def test(self, test_data: RankerDataset):
    """Computes the performance of the labeler on a test set.
    How many times is the does the labeler manage to predict the top scoring hypothesis correctly.
    """
    logging.info("Testing on test dataset")
    test_dataset = self._ranker_prep.make_dataset_from_generator(datapoints=test_data)
    example_counter = 0
    test_rank_correct = 0
    for example in test_dataset:
      example_counter += 1
      # print("example ", example)
      ranks = example["hypo_reward"]
      top_ranking_hypothesis = np.argmax(ranks)
      # print(ranks, top_ranking_hypothesis)

      scores = self.label_ranker(example, training=False)
      # print("scores ", scores)
      top_scoring_hypothesis = np.argmax(scores)
      # print(top_scoring_hypothesis)
      if top_ranking_hypothesis == top_scoring_hypothesis:
        test_rank_correct += 1
    test_rank_acc = test_rank_correct / example_counter
    return test_rank_acc

  def load_weights(self, *, name: str, path=None):
    """Loads a pretrained model weights."""
    if path is None:
      path = os.path.join(_MODEL_DIR, name)
    else:
      path = os.path.join(path, name)
    label_ranker = LabelRankerModel(word_embeddings=self.word_embeddings, name=name)
    label_ranker(inputs=self.inputs)
    load_status = label_ranker.load_weights(os.path.join(path, name))
    logging.info(f"Loaded model from model named: {name} in: {_MODEL_DIR}")
    # load_status.assert_consumed()
    # logging.info(load_status)
    return label_ranker

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


class LabelRankerModel(tfrs.Model):
  def __init__(self, *,
               word_embeddings: Embeddings,
               name:str):
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
    self.reshape_words = tf.keras.layers.Reshape((600,), input_shape=(2,300))
    self.reshape_pos = tf.keras.layers.Reshape((64, ), input_shape=(2,32))
    self.concatenate = layers.Concatenate(name="concat")
    self.dense1 = layers.Dense(512, name="densor1",  kernel_initializer='glorot_uniform',
                               activation="relu")
    self.dense2 = layers.Dense(256, name="densor2",  kernel_initializer='glorot_uniform',
                               activation="relu")
    self.dense3 = layers.Dense(128, name="densor3", kernel_initializer='glorot_uniform',
                               activation="relu")
    self.dense4 = layers.Dense(64, name="densor4", kernel_initializer='glorot_uniform',
                               activation="relu")
    self.dense5 = layers.Dense(32, name="densor5", kernel_initializer='glorot_uniform',
                               activation="relu")
    self.dense6 = layers.Dense(16, name="densor6", kernel_initializer='glorot_uniform',
                               activation="relu")
    self.dense7 = layers.Dense(8, name="densor7", kernel_initializer='glorot_uniform',
                               activation="relu")
    self.dense8 = layers.Dense(1, name="output")
    self.task = tfrs.tasks.Ranking(
      loss=tfr.keras.losses.PairwiseHingeLoss(),
      metrics=[
        tfr.keras.metrics.NDCGMetric(name="ndcg_metric"),
        tf.keras.metrics.RootMeanSquaredError()
      ]
    )

  def compute_loss(self, inputs, training=False):
    labels = tf.expand_dims(inputs["hypo_reward"], 0)
    scores = self(inputs)
    scores = tf.expand_dims(scores, 0)
    return self.task(
      labels=labels,
      predictions=scores,
    )

  def call(self, inputs, training=True):
    word_feats = self.word_embeddings(inputs["word_id"])
    pos_feats = self.pos_embeddings(inputs["pos_id"])
    hypothesized_label = self.label_embeddings(inputs["hypo_label_id"])

    next_token_word_feats = self.word_embeddings(inputs["next_token_ids"])
    # this has shape (batch_size, 2, 300). so it is flattened to (batch_size, 600)
    next_token_word_feats = self.reshape_words(next_token_word_feats)
    # print("next token feats reshaped ", next_token_word_feats)
    prev_token_word_feats = self.word_embeddings(inputs["prev_token_ids"])
    prev_token_word_feats = self.reshape_words(prev_token_word_feats)
    # print("prev token feats reshaped ", prev_token_word_feats)

    next_token_pos_feats = self.pos_embeddings(inputs["next_token_pos_ids"])
    next_token_pos_feats = self.reshape_pos(next_token_pos_feats)
    # print("next_token_pos_feats ", next_token_pos_feats)

    prev_token_pos_feats = self.pos_embeddings(inputs["prev_token_pos_ids"])
    prev_token_pos_feats = self.reshape_pos((prev_token_pos_feats))
    # print("prev token pos feats ", prev_token_pos_feats)

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

    return tf.squeeze(output, -1)


def compare_labeler_to_ranker(ranker, test_data: RankerDataset):
  labeler_baseline = ranker.get_labeler_baseline(test_data)
  ranker_accuracy = ranker.test(test_data)
  print(f"Labeler Baseline: {labeler_baseline}")
  print(f"Ranker Accuracy: {ranker_accuracy}")


def main(args):
  train_data_path = "./ranker/data/tr_boun-ud-train-ranker-data-rio.tfrecords"
  # train_data_path = "./ranker/ranker_train_data_rio.tfrecords"
  test_data_path = "./ranker/data/tr_boun-ud-test-ranker-datapoint.pbtxt"
  test_data = reader.ReadRankerTextProto(path=test_data_path)
  dataset = ranker_preprocessor.read_dataset_from_tfrecords(train_data_path, batch_size=5)
  labeler_model_name="bilstm_labeler_topk"
  parser_model_name="label_first_gold_morph_and_labels"
  word_embeddings = load_models.load_word_embeddings()
  if args.load:
    ranker = Ranker(labeler_model_name=labeler_model_name,
                    parser_model_name=parser_model_name,
                    word_embeddings=word_embeddings,
                    from_disk=True,
                    name=args.name)
    # continue training
    history = ranker.train(dataset=dataset, epochs=args.epochs, test_data=test_data, test_every=args.test_every)
    # compare_labeler_to_ranker(ranker, test_data)
    ranker.save_weights()
  else:
    ranker = Ranker(labeler_model_name=labeler_model_name,
                    parser_model_name=parser_model_name,
                    word_embeddings=word_embeddings,
                    name=args.name)
    print(ranker.label_ranker)
    history = ranker.train(dataset=dataset, epochs=args.epochs , test_data=test_data, test_every=args.test_every)
    ranker.save_weights()
  # for k, v in history.history.items():
  #   print(k, v)
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

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--load",
                      type=bool,
                      default=False,
                      help="Whether to load a pretrained ranker from disk.")
  parser.add_argument("--name",
                      type=str,
                      required=True,
                      help="The name of the model.")
  parser.add_argument("--epochs",
                      type=int,
                      default=10,
                      help="Trains a new model.")
  parser.add_argument("--test_every",
                      type=int,
                      default=5,
                      help="Trains a new model.")
  args = parser.parse_args()
  main(args)

