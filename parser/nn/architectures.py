"""Model architechtures go here.

All classes in this module inherit from keras.Model
"""

import logging
import tensorflow as tf
import tensorflow_addons as tfa

from tensorflow.keras import layers
from input import embeddor
from parser.nn import layer_utils

from typing import List

logging.basicConfig(format='%(levelname)s : %(message)s', level=logging.INFO)

Embeddings = embeddor.Embeddings

class LabelFirstParsingModel(tf.keras.Model):
  """Label first parsing model predicts labels before edges."""
  def __init__(self, *,
               n_dep_labels: int,
               word_embeddings: Embeddings, 
               config=None,
               name="Label_First_Parsing_Model",
               predict: List[str]):
    super(LabelFirstParsingModel, self).__init__(name=name)
    self.predict = predict
    self._null_label = tf.constant(0)
    self.word_embeddings = layer_utils.EmbeddingLayer(
                                          pretrained=word_embeddings,
                                          name="word_embeddings")
    self.pos_embeddings = layer_utils.EmbeddingLayer(
                                          input_dim=35, output_dim=32,
                                          name="pos_embeddings",
                                          trainable=True)
    # self.label_embeddings = layer_utils.EmbeddingLayer(input_dim=36,
    #                                                   output_dim=50,
    #                                                   name="label_embeddings",
    #                                                   trainable=True)
    self.concatenate = layers.Concatenate(name="concat")
    self.encoder = layer_utils.LSTMBlock(n_units=256, dropout_rate=0.3,
                                         name="lstm_encoder")
    self.attention = layer_utils.Attention()
    
    if "labels" in self.predict:
      self.dep_labels = layers.Dense(units=n_dep_labels, name="dep_labels")
    
    self.head_perceptron = layer_utils.Perceptron(n_units=256,
                                                  activation="relu",
                                                  name="head_mlp")
    self.dep_perceptron = layer_utils.Perceptron(n_units=256,
                                                 activation="relu",
                                                 name="dep_mlp")
    self.edge_scorer = layer_utils.EdgeScorer(n_units=256, name="edge_scorer")
    logging.info((f"Set up {name} to predict {predict}"))

  def call(self, inputs): # inputs = Dict[str, tf.keras.Input]
    """Forward pass.
    Args:
      inputs: Dict[str, tf.keras.Input]
    Returns:
      A dict which conteins:
        edge_scores: [batch_size, seq_len, seq_len] head preds for all tokens.
        label_scores: [batch_size, seq_len, n_labels] label preds for tokens.
    """
    # print("inputs ", inputs)
    word_inputs = inputs["words"]
    word_features = self.word_embeddings(word_inputs)
    pos_inputs = inputs["pos"]
    pos_features = self.pos_embeddings(pos_inputs)
    morph_inputs = inputs["morph"]
    # label_inputs = inputs["labels"]
    # label_features = self.label_embeddings(label_inputs)
    concat = self.concatenate([word_features,
                               pos_features,
                               morph_inputs,
                               # label_features
                              ])
    # concat = self.concatenate([word_features, label_features])
    sentence_repr = self.encoder(concat)
    sentence_repr = self.attention(sentence_repr)
    if "labels" in self.predict:
      dep_labels = self.dep_labels(sentence_repr)
      h_arc_head = self.head_perceptron(dep_labels)
      h_arc_dep = self.dep_perceptron(dep_labels)
      edge_scores = self.edge_scorer(h_arc_head, h_arc_dep)
      return {"edges": edge_scores,  "labels": dep_labels}
    else:
      h_arc_head = self.head_perceptron(sentence_repr)
      h_arc_dep = self.dep_perceptron(sentence_repr)
      edge_scores = self.edge_scorer(h_arc_head, h_arc_dep)
      return {"edges": edge_scores,  "labels": self._null_label}


class BiaffineParsingModel(tf.keras.Model):
  def __init__(self, *,
               n_dep_labels: int,
               word_embeddings: Embeddings, 
               config=None, 
               name="Biaffine_Parsing_Model",
               predict: List[str]):
    super(BiaffineParsingModel, self).__init__(name=name)
    self.predict = predict
    self._null_label = tf.constant(0)
    self.word_embeddings = layer_utils.EmbeddingLayer(
                                          pretrained=word_embeddings,
                                          name="word_embeddings")
    self.pos_embeddings = layer_utils.EmbeddingLayer(
                                         input_dim=35, output_dim=32,
                                         name="pos_embeddings",
                                         trainable=True)
    self.concatenate = layers.Concatenate(name="concat")
    self.encoder = layer_utils.LSTMBlock(n_units=256, dropout_rate=0.3,
                                         name="lstm_encoder")
    self.attention = layer_utils.Attention()
    
    if "labels" in self.predict:
      self.rel_perceptron_h = layer_utils.Perceptron(n_units=256,
                                                     activation="relu",
                                                     name="rel_mlp_h")
      self.rel_perceptron_d = layer_utils.Perceptron(n_units=256,
                                                     activation="relu",
                                                     name="rel_nlp_d")
      self.label_scorer = layer_utils.DozatBiaffineScorer(
                                                  out_channels=n_dep_labels,
                                                  name="biaffine_label_scorer")
    
    self.head_perceptron = layer_utils.Perceptron(n_units=256,
                                                  activation="relu",
                                                  name="head_mlp")
    self.dep_perceptron = layer_utils.Perceptron(n_units=256,
                                                 activation="relu",
                                                 name="dep_mlp")
    self.edge_scorer = layer_utils.DozatBiaffineScorer(
                                                 name="biaffine_edge_scorer")
    logging.info((f"Set up {name} to predict {predict}"))
  
  def call(self, inputs): # inputs = Dict[str, tf.keras.Input]
    """Forward pass.
    Args:
      inputs: Dict[str, tf.keras.Input]
    Returns:
      A dict which conteins:
        edge_scores: [batch_size, seq_len, seq_len] head preds for all tokens.
        label_scores: [batch_size, n_labels, seq_len, n_labels]. This tensor
          hold the probability score of seeing each label in n_labels when each
          token x is a dependent for each token y in the sentence.
    """
    word_inputs = inputs["words"]
    word_features = self.word_embeddings(word_inputs)
    pos_inputs = inputs["pos"]
    pos_features = self.pos_embeddings(pos_inputs)
    morph_inputs = inputs["morph"]
    concat = self.concatenate([word_features, pos_features, morph_inputs])
    sentence_repr = self.encoder(concat)
    sentence_repr = self.attention(sentence_repr)
    if "labels" in self.predict:
      h_arc_head = self.head_perceptron(sentence_repr)
      h_arc_dep = self.dep_perceptron(sentence_repr)
      edge_scores = self.edge_scorer(h_arc_head, h_arc_dep)
      
      h_rel_head = self.rel_perceptron_h(sentence_repr)
      h_rel_dep = self.rel_perceptron_d(sentence_repr)
      label_scores = self.label_scorer(h_rel_head, h_rel_dep)
      # print("edge scores shape: ", edge_scores.shape)
      # print("label scores shape: ", label_scores.shape)
      return {"edges": edge_scores,  "labels": label_scores}
    else:
      h_arc_head = self.head_perceptron(sentence_repr)
      h_arc_dep = self.dep_perceptron(sentence_repr)
      edge_scores = self.edge_scorer(h_arc_head, h_arc_dep)
      return {"edges": edge_scores,  "labels": self._null_label}
    

class GruEncoder(tf.keras.Model):
  def __init__(self, *,
               word_embeddings: Embeddings,
               encoder_dim: int,
               batch_size: int,
              name="GruEncoder"):
      super(GruEncoder, self).__init__(name=name)
      self.encoder_dim = encoder_dim
      self.word_embeddings = layer_utils.EmbeddingLayer(
                                      pretrained=word_embeddings,
                                      name="word_embeddings"
                                      )
      self.pos_embeddings = layer_utils.EmbeddingLayer(
                                      input_dim=35, output_dim=32,
                                      name="pos_embeddings",
                                      trainable=True
                                      )
      self.concatenate = layers.Concatenate(name="concat")
      self.rnn=layers.GRU(encoder_dim,
                          return_sequences=False,
                          return_state=True)
      # TODO: self.encoder_rnn_cell = tf.keras.layers.LSTMCell(self.dec_units)
    
  def call(self, inputs, state):
      word_inputs = inputs["words"]
      word_features = self.word_embeddings(word_inputs)
      pos_inputs = inputs["pos"]
      pos_features = self.pos_embeddings(pos_inputs)
      morph_inputs = inputs["morph"]
        
      concat = self.concatenate([word_features, pos_features, morph_inputs])
      
      thought, state = self.rnn(concat, initial_state=state)
      return thought, state
    
  def init_state(self, batch_size):
      return tf.zeros((batch_size, self.encoder_dim))

class GruDecoder(tf.keras.Model):
  def __init__(self, *,
               n_labels: int,
               decoder_dim: int,
               embedding_dim: int,
               name="GruDecoder"):
  
      super(GruDecoder, self).__init__(name=name)
      self.decoder_dim = decoder_dim
      self.embedding = tf.keras.layers.Embedding(n_labels, embedding_dim, trainable=True) 
      self.concatenate = layers.Concatenate(name="concat")
      self.rnn=layers.GRU(decoder_dim, return_sequences=True, return_state=True)
      # TODO: self.decoder_rnn_cell = tf.keras.layers.LSTMCell(self.dec_units)
      self.dense = layers.Dense(n_labels)
    
  def call(self, x, state):
    x = self.embedding(x)
    x, state = self.rnn (x, initial_state=state)
    label = self.dense(x)
    return label, state       
 
 
      