"""Model architechtures go here.

All classes in this module inherit from keras.Model
"""

import logging
import tensorflow as tf

from tensorflow.keras import layers
from input import embeddor
from parser.nn import layer_utils

from typing import List

logging.basicConfig(format='%(levelname)s : %(message)s', level=logging.INFO)

Embeddings = embeddor.Embeddings

class LSTMLabelingModel(tf.keras.Model):
  """A standalone bidirectional LSTM labeler."""
  def __init__(self, *,
               word_embeddings: Embeddings,
               n_units: int = 256,
               n_output_classes: int,
               use_pos=True,
               use_morph=True,
               return_lstm_output=True,
               name="LSTM_Labeler"):
    super(LSTMLabelingModel, self).__init__(name=name)
    self.use_pos = use_pos
    self.use_morph = use_morph
    self.word_embeddings = layer_utils.EmbeddingLayer(
      pretrained=word_embeddings, name="word_embeddings"
    )
    self.return_lstm_output = return_lstm_output
    if self.use_pos:
      self.pos_embeddings = layer_utils.EmbeddingLayer(
                                            input_dim=37, output_dim=32,
                                            name="pos_embeddings",
                                            trainable=True)
    self.concatenate = layers.Concatenate(name="concat")
    self.lstm_block = layer_utils.LSTMBlock(n_units=n_units,
                                            dropout_rate=0.3,
                                            name="lstm_block"
                                            )
    # Because in the loss function we have from_logits=True, we don't use the
    # param 'activation=softmax' in the layer. The loss function applies softmax to the
    # raw probabilites and then applies crossentropy.
    self.labels = layers.Dense(units=n_output_classes, name="labels")

  def call(self, inputs):
    # TODO: can we have the labeler use heads as a feature too?
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
    word_features = self.word_embeddings(word_inputs)
    concat_list = [word_features]
    if self.use_pos:
      pos_inputs = inputs["pos"]
      pos_features = self.pos_embeddings(pos_inputs)
      concat_list.append(pos_features)
    if self.use_morph:
      morph_inputs = inputs["morph"]
      concat_list.append(morph_inputs)
    
    if len(concat_list) > 1:
      concat = self.concatenate(concat_list)
      sentence_repr = self.lstm_block(concat)
      labels = self.labels(sentence_repr)
    else:
      sentence_repr = self.lstm_block(word_features)
      labels = self.labels(sentence_repr)
    if self.return_lstm_output:
      return {"labels": labels}, sentence_repr
    else:
      return {"labels": labels}
    

class LabelFirstParsingModel(tf.keras.Model):
  """Label first parsing model predicts labels before edges."""
  def __init__(self, *,
               n_dep_labels: int,
               word_embeddings: Embeddings,
               name="Label_First_Parsing_Model",
               predict: List[str],
               use_pos:bool = True,
               use_morph:bool=True,
               use_dep_labels:bool=False # for cases where we predict only edges using dep labels as gold.
               ):
    super(LabelFirstParsingModel, self).__init__(name=name)
    self.predict = predict
    self.use_pos = use_pos
    self.use_morph = use_morph
    self.use_dep_labels = use_dep_labels
    self._null_label = tf.constant(0)

    assert(not("labels" in self.predict and self.use_dep_labels)), "Can't use dep_labels both as feature and label!"

    self.word_embeddings = layer_utils.EmbeddingLayer(
                                          pretrained=word_embeddings,
                                          name="word_embeddings")

    if self.use_pos:
      self.pos_embeddings = layer_utils.EmbeddingLayer(
                                            input_dim=37, output_dim=32,
                                            name="pos_embeddings",
                                            trainable=True)

    self.concatenate = layers.Concatenate(name="concat")
    self.lstm_block = layer_utils.LSTMBlock(n_units=256,
                                            dropout_rate=0.3,
                                            name="lstm_block")
    # self.attention = layer_utils.Attention()
    
    if "labels" in self.predict:
      self.dep_labels = layers.Dense(units=n_dep_labels, name="labels")
    else:
      if self.use_dep_labels: # using dep labels as gold features.
        self.label_embeddings = layer_utils.EmbeddingLayer(input_dim=43,
                                                           output_dim=50,
                                                           name="label_embeddings",
                                                           trainable=True)
    
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
      inputs: Dict[str, tf.keras.Input]. This consist of
        words: Tensor of shape (batch_size, seq_len)
        pos: Tensor of shape (batch_size, seq_len)
        morph: Tensor of shape (batch_size, seq_len, 66)
        dep_labels: Tensor of shape (batch_size, seq_len)
      The boolean values set up during the initiation of the model determines
      which one of these features to use or not.
    Returns:
      A dict which conteins:
        edge_scores: [batch_size, seq_len, seq_len] head preds for all tokens (i.e. 10, 34, 34)
        label_scores: [batch_size, seq_len, n_labels] label preds for tokens (i.e. 10, 34, 36)
    """
    word_inputs = inputs["words"]
    word_features = self.word_embeddings(word_inputs)
    concat_list = [word_features]
    if self.use_pos:
      pos_inputs = inputs["pos"]
      pos_features = self.pos_embeddings(pos_inputs)
      concat_list.append(pos_features)
    if self.use_morph:
      morph_inputs = inputs["morph"]
      concat_list.append(morph_inputs)
    if self.use_dep_labels:
      label_inputs = inputs["labels"]
      label_features = self.label_embeddings(label_inputs)
      concat_list.append(label_features)
    if len(concat_list) > 1:
      sentence_repr = self.concatenate(concat_list)
    else:
      sentence_repr = word_features
    sentence_repr = self.lstm_block(sentence_repr)
    # sentence_repr = self.attention(sentence_repr)
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
               name="Biaffine_Parsing_Model",
               predict: List[str] = ["heads", "labels"],
               use_pos:bool=True,
               use_morph:bool=False):
    super(BiaffineParsingModel, self).__init__(name=name)
    self.predict = predict
    assert("labels" in self.predict) , "Biaffine parser has to predict labels."

    self._null_label = tf.constant(0)
    self._use_pos = use_pos
    self._use_morph = use_morph

    self.word_embeddings = layer_utils.EmbeddingLayer(
                                          pretrained=word_embeddings,
                                          name="word_embeddings")


    if self._use_pos:
      self.pos_embeddings = layer_utils.EmbeddingLayer(
                                          input_dim=37, output_dim=32,
                                          name="pos_embeddings",
                                          trainable=True)

    self.concatenate = layers.Concatenate(name="concat")
    self.encoder = layer_utils.LSTMBlock(n_units=256, dropout_rate=0.3, name="lstm_encoder")
    # self.attention = layer_utils.Attention()

    self.rel_perceptron_h = layer_utils.Perceptron(n_units=256, activation="relu", name="rel_mlp_h")
    self.rel_perceptron_d = layer_utils.Perceptron(n_units=256, activation="relu", name="rel_nlp_d")
    self.label_scorer = layer_utils.DozatBiaffineScorer(out_channels=n_dep_labels, name="biaffine_label_scorer")

    self.head_perceptron = layer_utils.Perceptron(n_units=256, activation="relu", name="head_mlp")
    self.dep_perceptron = layer_utils.Perceptron(n_units=256, activation="relu", name="dep_mlp")
    self.edge_scorer = layer_utils.DozatBiaffineScorer(name="biaffine_edge_scorer")

    logging.info((f"Set up {name} to predict {predict}"))
  
  def call(self, inputs): # inputs = Dict[str, tf.keras.Input]
    """Forward pass.
    Args:
      inputs: Dict[str, tf.keras.Input]
    Returns:
      A dict which conteins:
        edge_scores: [batch_size, seq_len, seq_len] head preds for all tokens.
        label_scores: [batch_size, n_labels, seq_len, seq_len]. This tensor
          holds the probability score of seeing each label in n_labels when each
          token x is a dependent for each token y in the sentence.
    """
    word_inputs = inputs["words"]
    word_features = self.word_embeddings(word_inputs)
    concat_list = [word_features]
    if self._use_pos:
      pos_inputs = inputs["pos"]
      pos_features = self.pos_embeddings(pos_inputs)
      concat_list.append(pos_features)
    if self._use_morph:
      concat_list.append(inputs["morph"])

    if len(concat_list) > 1:
      sentence_repr = self.concatenate(concat_list)
    else:
      sentence_repr = word_features

    sentence_repr = self.encoder(sentence_repr)
    # sentence_repr = self.attention(sentence_repr)

    h_arc_head = self.head_perceptron(sentence_repr)
    h_arc_dep = self.dep_perceptron(sentence_repr)
    edge_scores = self.edge_scorer(h_arc_head, h_arc_dep)
      
    h_rel_head = self.rel_perceptron_h(sentence_repr)
    h_rel_dep = self.rel_perceptron_d(sentence_repr)
    label_scores = self.label_scorer(h_rel_head, h_rel_dep)

    return {"edges": edge_scores,  "labels": label_scores}

    
class EncoderDecoderLabelFirstParser(tf.keras.Model):
  def __init__(self, *,
               n_dep_labels: int,
               word_embeddings: Embeddings,
               pos_embedding_dim: int = 37,
               character_embedding_dim: int = 32,
               encoder_dim: int,
               decoder_dim: int,
               batch_size = None,
               predict: List[str] = ["edges", "labels"],
               name="EncoderDecoderLabelFirstParser",
               config=None):
    super(EncoderDecoderLabelFirstParser, self).__init__(name=name)
    self.predict = predict
    self.encoder = LSTMEncoder(word_embeddings=word_embeddings,
                               pos_embedding_dim=pos_embedding_dim,
                               encoder_dim=encoder_dim,
                               batch_size=batch_size
                               )
    self.decoder = LSTMDecoder(n_labels=n_dep_labels,
                               decoder_dim=decoder_dim,
                               word_embeddings=word_embeddings,
                               pos_embedding_dim=pos_embedding_dim,
                               )
    self.head_perceptron = layer_utils.Perceptron(
       n_units=256, activation="relu", name="head_mlp")
    self.dep_perceptron = layer_utils.Perceptron(
        n_units=256, activation="relu", name="dep_mlp"
      )
    self.edge_scorer = layer_utils.EdgeScorer(n_units=256, name="edge_scorer")
    logging.info(f"Set up {name} to predict {predict}")
                                                   
  def call(self, inputs):
    encoder_out, enc_h, enc_c = self.encoder(inputs)
    dep_labels = self.decoder(inputs, initial_state=[enc_h, enc_c])
    h_arc_head = self.head_perceptron(dep_labels)
    h_arc_dep = self.dep_perceptron(dep_labels)
    edge_scores = self.edge_scorer(h_arc_head, h_arc_dep)
    return {"edges": edge_scores,  "labels": dep_labels}
      

class LSTMEncoder(tf.keras.Model):
  def __init__(self, *,
               word_embeddings: Embeddings,
               pos_embedding_dim: int = 37,
               encoder_dim: int,
               batch_size: int,
              name="LSTMEncoder"):
      super(LSTMEncoder, self).__init__(name=name)
      self.batch_size = batch_size
      self.encoder_dim = encoder_dim
      self.word_embeddings = layer_utils.EmbeddingLayer(
                                      pretrained=word_embeddings,
                                      name="word_embeddings"
                                      )
      self.pos_embeddings = layer_utils.EmbeddingLayer(
                                      input_dim=37,
                                      output_dim=pos_embedding_dim,
                                      name="pos_embeddings",
                                      trainable=True
                                      )
      self.concatenate = layers.Concatenate(name="concat")
      
      # LSTM layer
      self.lstm=layer_utils.UnidirectionalLSTMBlock(
                                      n_units=self.encoder_dim,
                                      return_sequences=True,
                                      return_state=True,
                                      dropout_rate=0.33)

    
  def call(self, inputs):
      word_inputs = inputs["words"]
      word_features = self.word_embeddings(word_inputs)
      pos_inputs = inputs["pos"]
      pos_features = self.pos_embeddings(pos_inputs)
      morph_inputs = inputs["morph"]
        
      # concat = self.concatenate([word_features, pos_features, morph_inputs])
      
      output, h, c = self.lstm(word_features)
      return output, h, c
  

class LSTMDecoder(tf.keras.Model):
  def __init__(self, *, 
               n_labels: int,
               decoder_dim: int,
               word_embeddings: Embeddings,
               pos_embedding_dim: int = 37,
               name="LSTMDecoder"
               ):
      super(LSTMDecoder, self).__init__(name=name)
      self.decoder_dim = decoder_dim
      super(LSTMDecoder, self).__init__(name=name)
      
      
      self.word_embeddings = layer_utils.EmbeddingLayer(
        pretrained=word_embeddings,
        name="word_embeddings"
      )
      
      self.pos_embeddings=layer_utils.EmbeddingLayer(
        input_dim=35, output_dim=pos_embedding_dim, name="pos_embeddings",
        trainable=True
      )
      
      self.concatenate=layers.Concatenate(name="concat")
      
      
      self.lstm_layer1 = layers.LSTM(self.decoder_dim,
                                    return_sequences=True,
                                    return_state=True,
                                    name="lstm1")
      self.lstm_layer2 = layers.LSTM(self.decoder_dim,
                                    return_sequences=True,
                                    return_state=True,
                                    name="lstm2")
      self.lstm_layer3 = layers.LSTM(self.decoder_dim,
                                     return_sequences=True,
                                     name="lstm3")
      self.dense = layers.Dense(units=n_labels, name="dense")
  
  def call(self, inputs, initial_state):
    word_inputs = inputs["words"]
    word_features = self.word_embeddings(word_inputs)
    pos_inputs=inputs["pos"]
    pos_features = self.pos_embeddings(pos_inputs)
    morph_inputs = inputs["morph"]
    
    concat = self.concatenate([
      word_features, pos_features, morph_inputs
    ])
    
    lstm_out1, h, c  = self.lstm_layer1(concat, initial_state=initial_state)
    lstm_out2, h, c = self.lstm_layer2(lstm_out1, initial_state=[h, c])
    lstm_out3 = self.lstm_layer3(lstm_out2, initial_state=[h, c])
    outputs = self.dense(lstm_out3)
    return outputs
    
    

class LSTMSeqEncoder(tf.keras.Model):
  def __init__(self, *,
               word_embeddings: Embeddings,
               encoder_dim: int,
               batch_size: int,
              name="LSTMSeqEncoder"):
      super(LSTMSeqEncoder, self).__init__(name=name)
      self.batch_size = batch_size
      self.encoder_dim = encoder_dim
      self.word_embeddings = layer_utils.EmbeddingLayer(
                                      pretrained=word_embeddings,
                                      name="word_embeddings"
                                      )
      self.pos_embeddings = layer_utils.EmbeddingLayer(
                                      input_dim=37, output_dim=32,
                                      name="pos_embeddings",
                                      trainable=True
                                      )
      self.concatenate = layers.Concatenate(name="concat")
      
      # LSTM layer
      self.lstm=layers.LSTM(self.encoder_dim,
                            return_sequences=True,
                            return_state=True,
                            recurrent_initializer='glorot_uniform')

    
  def call(self, inputs, state):
      word_inputs = inputs["words"]
      word_features = self.word_embeddings(word_inputs)
      pos_inputs = inputs["pos"]
      pos_features = self.pos_embeddings(pos_inputs)
      morph_inputs = inputs["morph"]
        
      concat = self.concatenate([word_features, pos_features, morph_inputs])
      
      output, h, c = self.lstm(concat, initial_state=state)
      return output, h, c
    
  def init_state(self, batch_size):
      # The reason to have two tensors here is that one is for the 
      # h (hidden state) and one is for c (cell state).
      return [tf.zeros((batch_size, self.encoder_dim)), 
              tf.zeros((batch_size, self.encoder_dim))
             ]