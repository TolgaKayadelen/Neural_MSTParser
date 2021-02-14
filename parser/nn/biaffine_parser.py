import tensorflow as tf
import numpy as np
import argparse
import os
import time
from tensorflow.keras import layers, metrics, losses, optimizers
from typing import List, Dict, Tuple
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
    self.loss_fn = losses.SparseCategoricalCrossentropy(
      from_logits=True, reduction=tf.keras.losses.Reduction.NONE)
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
  def compute_loss(self, edge_scores, heads, pad_mask):
    n_sentences, n_words, _ = edge_scores.shape
    edge_scores = tf.reshape(edge_scores, shape=(n_sentences*n_words, n_words))
    heads = tf.reshape(heads, shape=(n_sentences*n_words, 1))
    pad_mask = tf.reshape(pad_mask, shape=(n_sentences*n_words, 1))
    
    predictions = tf.argmax(edge_scores, 1)
    predictions = tf.reshape(predictions, shape=(predictions.shape[0], 1))
    
    # Compute losses with and without pad
    loss_with_pad = self.loss_fn(heads, edge_scores)
    loss_with_pad = tf.reshape(loss_with_pad, shape=(loss_with_pad.shape[0], 1))
    loss_without_pad = tf.boolean_mask(loss_with_pad, pad_mask)
    
    return loss_with_pad, loss_without_pad, predictions, heads, pad_mask
  
  
  @tf.function
  def train_step(self, *, words: tf.Tensor, pos: tf.Tensor, heads:tf.Tensor
                 ) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
    with tf.GradientTape() as tape:
      edge_scores = self.model({"words": words, "pos": pos}, training=True)
      pad_mask = (words != 0)
      loss_with_pad, loss_without_pad, predictions, h, pad_mask = self.compute_loss(
        edge_scores, heads, pad_mask)
    
    # Even though we compute the loss with and without padding, we optimize
    # the gradient based on the loss value with padding. This makes the learning
    # more stable. You can uncomment the below line if you want to change this
    # (not advised.)
    grads = tape.gradient(loss_with_pad, self.model.trainable_weights)
    # grads = tape.gradient(loss_without_pad, self.model.trainable_weights)
    self.optimizer.apply_gradients(zip(grads, self.model.trainable_weights))
    
    # Update train metrics
    self.train_metrics.update_state(heads, edge_scores)
    
    return loss_with_pad, loss_without_pad, predictions, h, pad_mask

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
        loss_w_pad, loss_w_o_pad, predictions, h, pad_mask = self.train_step(
          words=words, pos=pos, heads=heads)
        
        # Get the total number of tokens without padding.
        words_reshaped = tf.reshape(words, shape=(pad_mask.shape))
        total_words = len(tf.boolean_mask(words_reshaped, pad_mask))
        
        # Calculate correct predictions with and without padding.
        correct_preds_with_pad = (predictions == h)
        correct_predictions = tf.boolean_mask(predictions == h, pad_mask)
        n_correct_with_pad = np.sum(correct_preds_with_pad) 
        n_correct = np.sum(correct_predictions)
        
        # Add to stats
        stats["n_correct_with_pad"] += n_correct_with_pad
        stats["n_correct"] += n_correct
        stats["n_tokens_with_pad"] += len(h)
        stats["n_tokens"] += total_words
    
      print(stats)
      train_acc = self.train_metrics.result()
      
      # Compute UAS with and without padding for this epoch
      uas_with_pad = stats["n_correct_with_pad"] / stats["n_tokens_with_pad"]
      uas_without_pad = stats["n_correct"] / stats["n_tokens"] 
      
      # Compute average loss with and without padding
      loss_with_pad = tf.reduce_mean(loss_w_pad)
      loss_without_pad = tf.reduce_mean(loss_w_o_pad)
      
      # Log all the stats
      logging.info(f"Training accuracy: {train_acc}")
      logging.info(f"UAS (with pad): {uas_with_pad}")
      logging.info(f"UAS (without pad): {uas_without_pad}")
      logging.info(f"Loss (with pad) : {loss_with_pad}")
      logging.info(f"Loss (without pad) : {loss_without_pad}")
      logging.info(f"Time for epoch: {time.time() - start_time}\n")
      
      # Populate history
      # Only the UAS without padding is considered.
      history["uas"].append(uas_without_pad)
      history["loss_with_pad"].append(loss_with_pad)
      history["loss_without_pad"].append(loss_without_pad)
      
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


def plot(epochs, arc_scores, loss_with_pad, loss_without_pad, model_name):
  fig = plt.figure()
  ax = plt.axes()
  ax.plot(epochs, arc_scores, "-g", label="uas", color="blue")
  ax.plot(epochs, loss_with_pad, "-g", label="loss_w_pading", color="red")
  ax.plot(epochs, loss_without_pad, "-g", label="loss_w_o_padding", color="green")
  plt.title("Performance on training data")
  plt.xlabel("epochs")
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
  plot(np.arange(args.epochs), scores["uas"], scores["loss_with_pad"],
       scores["loss_without_pad"], "training_performance_500")
    
if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--train", type=bool, default=False,
                      help="Trains a new model.")
  parser.add_argument("--epochs", type=int, default=10,
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