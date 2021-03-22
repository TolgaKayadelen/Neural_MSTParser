"""NN architectures."""

import tensorflow as tf
import numpy as np
import logging

from tensorflow.keras import layers, metrics, losses, optimizers
from input import embeddor
from typing import List, Dict, Tuple
from util.nn import nn_utils

logging.basicConfig(format='%(levelname)s : %(message)s', level=logging.INFO)

Embeddings = embeddor.Embeddings

class Input:
  """Defines properties of a keras input layer."""
  def __init__(self, name:str, shape=(None,), embed: bool = False,
               embed_in: int = None, embed_out: int = None,
              pretrained_embeddings: Embeddings = None):
    pass
              

class Attention(layers.Layer):
  """Implementation of an Attention layer."""
  def __init__(self, return_sequences=True, name="attention_layer"):
    super(Attention, self).__init__(name=name)
    self.return_sequences = return_sequences
  
  def build(self, input_shape):
    """Builds the attention layer.
    
    Args:
      Shape of a 3D tensor of (batch_size, seq_len, n_features).
    Returns:
      An Attention layer instance.
    """
    self.densor1=layers.Dense(1, activation="tanh")
    self.activator = layers.Activation("softmax", name="attention_weights")
    self.dotor = layers.Dot(axes=1)
  
  def call(self, inputs):
    """Computes the context vector for the inputs.
    
    Args:
      a: A 3D Tensor of (batch_size, seq_len, n_features)
    Returns:
      output: A 3D tensor of same shape as a, where the input is scaled
        according to the attention weights (i.e. context alphas)
    """
    energies = self.densor1(inputs)
    activations = self.activator(energies)
    output = inputs*activations
    return output


class EmbeddingLayer(layers.Layer):
  """Builds a keras embedding layer."""
  def __init__(self, *,
               pretrained: Embeddings = None, input_dim: int = None,
               output_dim: int = None, trainable: bool = False,
               name: str = "embedding") -> layers.Embedding:
    super(EmbeddingLayer, self).__init__(name=name)
    if pretrained is not None:
      logging.info(f"Setting up embedding layer from pretrained {name}")
      self.embedding = tf.keras.layers.Embedding(
        pretrained.vocab_size, pretrained.embedding_dim, trainable=trainable)
      self.embedding.build((None,))
      self.embedding.set_weights([pretrained.index_to_vector])
    elif input_dim and output_dim:
      logging.info(f"Setting up new embedding layer for {name}")
      self.embedding = tf.keras.layers.Embedding(
        input_dim=input_dim, output_dim=output_dim, trainable=trainable)
    else:
      raise FatalError("Can't set embeddings.")
  
  def call(self, input_tensor):
    """Passes input values through the embedding layer."""
    features = self.embedding(input_tensor)
    return features
    

class Perceptron(layers.Layer):
  """Single layered perceptron."""
  def __init__(self, *, n_units: int, activation: str,
               name: str) -> layers.Dense:
      super(Perceptron, self).__init__(name=name)
      self.densor = layers.Dense(n_units, activation=activation)
  
  def call(self, input_tensor):
    return self.densor(input_tensor)
    

class EdgeScorer(layers.Layer):
  """Biaffine layer for scoring edges."""
  def __init__(self, *, n_units: int, name:str, use_bias: bool = False):
    super(EdgeScorer, self).__init__(name=name)
    self.W_arc = layers.Dense(n_units, use_bias=use_bias, name="W_arc")
    self.b_arc = layers.Dense(1, use_bias=use_bias, name="b_arc")
  
  def call(self, h_arc_head, h_arc_dep):
    Hh_W = self.W_arc(h_arc_head)
    Hh_WT = tf.transpose(Hh_W, perm=[0,2,1])
    Hh_W_Ha = tf.linalg.matmul(h_arc_dep, Hh_WT)
    Hh_b = self.b_arc(h_arc_head)
    edge_scores = Hh_W_Ha + tf.transpose(Hh_b, perm=[0,2,1])
    return edge_scores
      

class LSTMBlock(layers.Layer):
  """A bidirectional LSTM block with 3 Birectional LSTM layers"""
  def __init__(self, *, n_units: int, return_sequences: bool = True,
               dropout_rate: float = 0.0, name="LSTMBlock"):
    super(LSTMBlock, self).__init__(name=name)
    self.dropout_rate = dropout_rate
    self.lstm1 = layers.Bidirectional(layers.LSTM(
      units=n_units, return_sequences=return_sequences, name="lstm1"))
    self.lstm2 = layers.Bidirectional(layers.LSTM(
      units=n_units, return_sequences=return_sequences, name="lstm2"))
    self.lstm3 = layers.Bidirectional(layers.LSTM(
      units=n_units, return_sequences=return_sequences, name="lstm3"))
    self.dropout1 = layers.Dropout(rate=dropout_rate, name="dropout1")
    self.dropout2 = layers.Dropout(rate=dropout_rate, name="dropout2")
    self.dropout3 = layers.Dropout(rate=dropout_rate, name="dropout3")
  
  def call(self, input_tensor):
    dropout = self.dropout_rate > 0
    if dropout:
      out = self.lstm1(input_tensor)
      out = self.dropout1(out)
      out = self.lstm2(out)
      out = self.dropout2(out)
      out = self.lstm3(out)
      out = self.dropout3(out)
    else:
      out = self.lstm1(input_tensor)
      out = self.lstm2(out)
      out = self.lstm3(out)
    return out    

class ParsingModel(tf.keras.Model):
  def __init__(self, *,
               n_dep_labels: int,
               word_embeddings: Embeddings, 
               config=None,
               name="ParsingModel"):
    super(ParsingModel, self).__init__(name=name)
    self.word_embeddings = EmbeddingLayer(pretrained=word_embeddings,
                                          name="word_embeddings")
    self.pos_embeddings = EmbeddingLayer(input_dim=35, output_dim=32,
                                         name="pos_embeddings",
                                         trainable=True)
    self.concatenate = layers.Concatenate(name="concat")
    self.encoder = LSTMBlock(n_units=256, dropout_rate=0.3, name="lstm_encoder")
    self.attention = Attention()
    self.dep_labels = layers.Dense(units=n_dep_labels, name="dep_labels")
    self.head_perceptron = Perceptron(n_units=256, activation="relu",
                                      name="head_mlp")
    self.dep_perceptron = Perceptron(n_units=256, activation="relu",
                                     name="dep_mlp")
    self.edge_scorer = EdgeScorer(n_units=256, name="edge_scorer")
  
  
  # TODO: this method should return dict.
  # TODO: branch this to parameterize returning dep_label predictios.
  def call(self, inputs: Dict[str, tf.keras.Input]):
    word_inputs = inputs["words"]
    word_features = self.word_embeddings(word_inputs)
    pos_inputs = inputs["pos"]
    pos_features = self.pos_embeddings(pos_inputs)
    morph_inputs = inputs["morph"]
    concat = self.concatenate([word_features, pos_features, morph_inputs])
    sentence_repr = self.encoder(concat)
    sentence_repr = self.attention(sentence_repr)
    dep_labels = self.dep_labels(sentence_repr)
    h_arc_head = self.head_perceptron(dep_labels)
    h_arc_dep = self.dep_perceptron(dep_labels)
    edge_scores = self.edge_scorer(h_arc_head, h_arc_dep)
    return edge_scores, dep_labels
    