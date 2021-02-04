import tensorflow as tf
import numpy as np
import argparse
import os
import time
from tensorflow.keras import layers, metrics, losses, optimizers
from typing import List, Dict
from input import preprocessor
from tagset.reader import LabelReader
from util.nn import nn_utils
import collections
import matplotlib.pyplot as plt

import logging


logging.basicConfig(format='%(levelname)s : %(message)s', level=logging.INFO)
np.set_printoptions(threshold=np.inf)


_DATA_DIR = "data/UDv23/Turkish/training"

Dataset = tf.data.Dataset
Embeddings = preprocessor.Embeddings
SequenceFeature = preprocessor.SequenceFeature

class NeuralMSTParser:
  """The neural mst parser."""
  
  def __init__(self, *, word_embeddings: Embeddings):
    self.word_embeddings = self._set_pretrained_embeddings(word_embeddings)
    self.loss_fn = losses.SparseCategoricalCrossentropy(from_logits=True)
    self.optimizer=tf.keras.optimizers.Adam(0.001, beta_1=0.9, beta_2=0.9)
    self.train_metrics = metrics.SparseCategoricalAccuracy()
    self.model = self._create_uncompiled_model()
    
  def __str__(self):
    return str(self.model.summary())
  
  def plot(self, name="neural_mstparser.png"):
     tf.keras.utils.plot_model(self.model, name, show_shapes=True)

  def _set_pretrained_embeddings(self, 
                                 embeddings: Embeddings) -> layers.Embedding:
    """Builds a pretrained keras embedding layer from an Embeddings object."""
    embed = tf.keras.layers.Embedding(embeddings.vocab_size,
                             embeddings.embedding_dim,
                             trainable=False,
                             name="word_embeddings")
    embed.build((None,))
    embed.set_weights([embeddings.index_to_vector])
    return embed
  
  def _create_uncompiled_model(self) -> tf.keras.Model:
    """Creates an NN model for edge factored dependency parsing."""
    
    word_inputs = tf.keras.Input(shape=(None,), name="words")
    word_features = self.word_embeddings(word_inputs)
    pos_inputs = tf.keras.Input(shape=([None]), name="pos")
    pos_features = tf.keras.layers.Embedding(input_dim=35,
                                             output_dim=32,
                                             name="pos_embeddings")(pos_inputs)
    concat = tf.keras.layers.concatenate([word_features, pos_features],
                                         name="concat")
    # encode the sentence with LSTM
    sentence_repr = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(
        units=256, return_sequences=True, name="encoded1"))(concat)
    sentence_repr = tf.keras.layers.Dropout(rate=0.5)(sentence_repr)
    sentence_repr = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(
        units=256, return_sequences=True, name="encoded2"))(sentence_repr)
    sentence_repr = tf.keras.layers.Dropout(rate=0.5)(sentence_repr)
    sentence_repr = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(
        units=256, return_sequences=True, name="encoded3"))(sentence_repr)
    
    # the edge scoring bit with the MLPs
    # Pass the sentence representation through two MLPs, for head and dep.
    head_mlp = tf.keras.layers.Dense(256, activation="relu", name="head_mlp")
    dep_mlp = tf.keras.layers.Dense(256, activation="relu", name="dep_mlp")
    h_arc_head = head_mlp(sentence_repr)
    h_arc_dep = dep_mlp(sentence_repr)
    
    # Biaffine part of the model. Computes the edge scores for all edges
    W_arc = tf.keras.layers.Dense(256, use_bias=False, name="W_arc")
    b_arc = tf.keras.layers.Dense(1, use_bias=False, name="b_arc")
    Hh_W = W_arc(h_arc_head)
    Hh_WT = tf.transpose(Hh_W, perm=[0,2,1])
    Hh_W_Ha = tf.linalg.matmul(h_arc_dep, Hh_WT)
    Hh_b = b_arc(h_arc_head)
    edge_scores = Hh_W_Ha + tf.transpose(Hh_b, perm=[0,2,1])
    
    model = tf.keras.Model(inputs={"words": word_inputs, "pos": pos_inputs},
                                    outputs=edge_scores)
    return model
  
  @tf.function
  def compute_loss(self, edge_scores, heads):
    n_sentences, n_words, _ = edge_scores.shape
    edge_scores = tf.reshape(edge_scores, shape=(n_sentences*n_words, n_words))
    heads = tf.reshape(heads, shape=(n_sentences*n_words, 1))
    predictions = tf.argmax(edge_scores, 1)
    loss_value = self.loss_fn(heads, edge_scores)
    return loss_value, predictions, heads
  
  
  @tf.function
  def train_step(self, words: tf.Tensor, pos: tf.Tensor, heads:tf.Tensor):
    with tf.GradientTape() as tape:
      edge_scores = self.model({"words": words, "pos": pos}, training=True)
      loss_value, predictions, h = self.compute_loss(edge_scores, heads)
      # n_sentences, n_words, _ = edge_scores.shape
      # edge_scores = tf.reshape(edge_scores, shape=(n_sentences*n_words, n_words))
      # heads = tf.reshape(heads, shape=(n_sentences*n_words, 1))
      # print(edge_scores)
      # print(heads)
      # input("press to cont")
      # loss_value = self.loss_fn(heads, edge_scores)
    grads = tape.gradient(loss_value, self.model.trainable_weights)
    self.optimizer.apply_gradients(zip(grads, self.model.trainable_weights))
    # Update train metrics
    self.train_metrics.update_state(heads, edge_scores)
    return loss_value, predictions, h

  def train_custom(self, dataset: Dataset, epochs: int=10):
    """Custom training method."""
    history = collections.defaultdict(list)
    for epoch in range(epochs):
      self.train_metrics.reset_states()
      logging.info(f"*********** Training epoch: {epoch} ***********\n")
      start_time = time.time()
      stats = collections.Counter()
      for step, batch in enumerate(dataset):
        words = batch["words"]
        pos = batch["pos"]
        heads = batch["heads"]
        loss_value, predictions, h = self.train_step(words, pos, heads)
        predictions = tf.reshape(predictions, shape=(predictions.shape[0], 1))
        n_correct = np.sum(predictions == h)
        stats["n_correct"] += n_correct
        stats["n_tokens"] += len(h)
      print(stats)
      uas = stats["n_correct"] / stats["n_tokens"]
      train_acc = self.train_metrics.result()
      l = tf.reduce_mean(loss_value)
      history["uas"].append(uas)
      history["loss"].append(l)
      logging.info(f"Training accuracy: {train_acc}")
      logging.info(f"UAS: {uas}")
      logging.info(f"Loss : {l}")
      # Log the time it takes for one epoch
      logging.info(f"Time for epoch: {time.time() - start_time}\n")
    return history


# TODO: the set up of the features should be done by the preprocessor class.
def _set_up_features(features: List[str], label=str) -> List[SequenceFeature]:
  sequence_features = []
  for feat in features:
    if not feat == label:
      sequence_features.append(preprocessor.SequenceFeature(name=feat))
    else:
      if feat == "heads":
        sequence_features.append(preprocessor.SequenceFeature(
          name=feat, is_label=True))
      else:
        label_dict = LabelReader.get_labels(label).labels
        label_indices = list(label_dict.values())
        label_feature = preprocessor.SequenceFeature(
          name=label, values=label_indices, n_values=len(label_indices),
          is_label=True)
        sequence_features.append(label_feature)
  return sequence_features


def plot(epochs, arc_scores, model_name):
  fig = plt.figure()
  ax = plt.axes()
  ax.plot(epochs,arc_scores, "-g", label="arcs", color="blue")
  plt.title("Performance on training data")
  plt.xlabel("epochs")
  plt.ylabel("accuracy")
  plt.legend()
  plt.savefig(f"{model_name}_plot")

def main(args):
  if args.train:
    embeddings = nn_utils.load_embeddings()
    word_embeddings = Embeddings(name= "word2vec", matrix=embeddings)
    prep = preprocessor.Preprocessor(word_embeddings=word_embeddings)
    
    sequence_features = _set_up_features(args.features, args.label)
    
    if args.dataset:
      dataset = prep.read_dataset_from_tfrecords(
                                 batch_size=50,
                                 features=sequence_features,
                                 records="./input/test501.tfrecords")
    else:
      dataset = prep.make_dataset_from_generator(
        path=os.path.join(_DATA_DIR, args.treebank),
        batch_size=10, 
        features=sequence_features
      )

  parser = NeuralMSTParser(word_embeddings=prep.word_embeddings)
  print(parser)
  parser.plot()
  scores = parser.train_custom(dataset, args.epochs)
  plot(np.arange(args.epochs), scores["uas"], "biaffine")
    
if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--train", type=bool, default=False,
                      help="Trains a new model.")
  parser.add_argument("--epochs", type=int, default=2,
                      help="Trains a new model.")
  parser.add_argument("--treebank", type=str,
                      default="treebank_train_0_50.pbtxt")
  parser.add_argument("--dataset",
                      help="path to a prepared tf.data.Dataset")
  parser.add_argument("--features", type=list,
                      default=["words", "pos", "dep_labels", "heads"],
                      help="features to use to train the model.")
  parser.add_argument("--label", type=str, default="heads",
                      help="labels to predict.")
  args = parser.parse_args()
  main(args)