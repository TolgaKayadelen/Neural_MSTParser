"""Custom layer configs go here.

All classes in this module inherit from keras.layers.Layer

"""

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
      logging.info(f"Setting up embedding layer for {name}")
      self.embedding = tf.keras.layers.Embedding(
        input_dim=input_dim, output_dim=output_dim, trainable=trainable)
    else:
      raise RuntimeError("Can't set embeddings.")
  
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
    

class DozatBiaffineScorer(layers.Layer):
  """Dozat style Biaffine Layer for scoring edges and labels.
  
  BiAffine Attention layer from https://arxiv.org/abs/1611.01734
  Expects inputs as batch_first sequences [batch_size, seq_len, seq_len].
  
  Returns score matrixes as [batch_size, seq_len, seq_len] for edge scores
  (out_channels=1), and scores as [batch_size, out_channels, seq_len, seq_]
  (where out_channels=#labels).
  """
  def __init__(self, *, out_channels: int = 1,
               name:str, use_bias: bool = False):
    super(DozatBiaffineScorer, self).__init__(name=name)
    self.out_channels = out_channels
    self.use_bias = use_bias
    print("Initialized the Biaffine Scorer.")
  
  def build(self, input_shape):
    # print("input shape is ", input_shape)
    self.W = self.add_weight(shape=(self.out_channels,
                                    input_shape[2],
                                    input_shape[2]),
                             initializer="random_normal",
                             trainable=True,
                             name="W")
    # print("weights shape is ", self.W.shape)  
  def call(self, x, y):
    if self.use_bias:
      x = tf.concat([x, tf.ones_like(x[..., :1])], axis=-1)
      y = tf.concat([y, tf.ones_like(y[..., :1])], axis=-1)
    
    s = tf.einsum('bxi,oij,byj->boxy', x, self.W, y)
    
    # If outchannels is 1, squeeze the matrix (remove dim=1)
    if s.shape[1] == 1:
      return tf.squeeze(s, 1)
    return s

class LSTMBlock(layers.Layer):
  """A bidirectional LSTM block with 3 Birectional LSTM layers"""
  def __init__(self, *,
              n_units: int,
              num_layers = 3,
              return_sequences: bool = True,
              return_state: bool = False,
              dropout_rate: float = 0.0,
              name="LSTMBlock"):
    super(LSTMBlock, self).__init__(name=name)
    total_layers = 0
    self.dropout_rate = dropout_rate
    if num_layers > 3:
      raise ValueError("More than 3 LSTM layers not supported.")
    self.lstm1 = layers.Bidirectional(layers.LSTM(
      units=n_units, return_sequences=return_sequences,
      name="lstm1"))
    self.dropout1 = layers.Dropout(rate=dropout_rate, name="dropout1")
    total_layers += 1
    num_layers -= 1
    if num_layers >= 1:
      self.lstm2 = layers.Bidirectional(layers.LSTM(
        units=n_units, return_sequences=return_sequences,
        name="lstm2"))
      self.dropout2 = layers.Dropout(rate=dropout_rate, name="dropout2")
      total_layers += 1
      num_layers -= 1
    else: self.lstm2 = None
    if num_layers >= 1:
      self.lstm3 = layers.Bidirectional(layers.LSTM(
        units=n_units, return_sequences=return_sequences,
        return_state=return_state,
        name="lstm3"))
      self.dropout3 = layers.Dropout(rate=dropout_rate, name="dropout3")
      num_layers -= 1
      total_layers += 1
    else: self.lstm3 = None
    logging.info(f"Total LSTM layers {total_layers}")
    input("Press to cont.")

  def call(self, input_tensor):
    dropout = self.dropout_rate > 0
    if dropout:
      out = self.lstm1(input_tensor)
      out = self.dropout1(out)
      if self.lstm2:
        out = self.lstm2(out)
        out = self.dropout2(out)
      if self.lstm3:
        out = self.lstm3(out)
        out = self.dropout3(out)
    else:
      out = self.lstm1(input_tensor)
      if self.lstm2 is not None:
        out = self.lstm2(out)
      if self.lstm3 is not None:
        out = self.lstm3(out)
    return out

class UnidirectionalLSTMBlock(layers.Layer):
  """An LSTM block with 3 LSTM layers"""
  def __init__(self, *, n_units: int,
              return_sequences: bool = True,
              return_state: bool = True,
              dropout_rate: float = 0.0, name="UnidirectionalLSTMBlock"):
    super(UnidirectionalLSTMBlock, self).__init__(name=name)
    self.dropout_rate = dropout_rate
    self.lstm1 = layers.LSTM(
      units=n_units,
      return_sequences=return_sequences,
      return_state=return_state,
      name="lstm1")
    self.lstm2 = layers.LSTM(
      units=n_units,
      return_sequences=return_sequences,
      return_state=return_state,
      name="lstm2")
    self.lstm3 = layers.LSTM(
      units=n_units,
      return_sequences=return_sequences,
      return_state=return_state, name="lstm3")
    self.dropout1 = layers.Dropout(rate=dropout_rate, name="dropout1")
    self.dropout2 = layers.Dropout(rate=dropout_rate, name="dropout2")
  
  def call(self, input_tensor):
    dropout = self.dropout_rate > 0
    if dropout:
      out, h, c = self.lstm1(input_tensor)
      out = self.dropout1(out)
      out, h, c = self.lstm2(out, initial_state=[h,c])
      out = self.dropout2(out)
      out, h, c = self.lstm3(out,initial_state=[h,c])
    else:
      out, h, c = self.lstm1(input_tensor, initial_state=[h,c])
      out, h, c = self.lstm2(out, initial_state=[h,c])
      out, h, c = self.lstm3(out, initial_state=[h,c])
    return out, h, c
