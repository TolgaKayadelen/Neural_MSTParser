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



class LSTMLabelingModel(tf.keras.Model):
  """A standalone bidirectional LSTM labeler."""
  def __init__(self, *,
               word_embeddings: Embeddings,
               n_units: int = 256,
               n_output_classes: int,
               use_pos=True,
               use_morph=True,
               name="LSTM_Labeler"):
    super(LSTMLabelingModel, self).__init__(name=name)
    self.use_pos = use_pos
    self.use_morph = use_morph
    
    self.word_embeddings = layer_utils.EmbeddingLayer(
      pretrained=word_embeddings, name="word_embeddings"
    )
    
    if self.use_pos:
      self.pos_embeddings = layer_utils.EmbeddingLayer(
                                            input_dim=35, output_dim=32,
                                            name="pos_embeddings",
                                            trainable=True)
    if self.use_pos or self.use_morph:
      self.concatenate = layers.Concatenate(name="concat")
    self.lstm_block = layer_utils.LSTMBlock(n_units=n_units,
                                            dropout_rate=0.3,
                                            name="lstm_block"
                                            )
    # self.attention = layer_utils.Attention()
    self.labels = layers.Dense(units=n_output_classes, activation="softmax",
                               name="labels")

  def call(self, inputs):
    """Forward pass."""
    
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
      # sentence_repr = self.attention(sentence_repr)
      labels = self.labels(sentence_repr)
    else:
      sentence_repr = self.lstm_block(word_features)
      # sentence_repr = self.attention(sentence_repr)
      labels = self.labels(sentence_repr)
    
    return {"labels": labels}
    
    

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
    
class EncoderDecoderLabelFirstParser(tf.keras.Model):
  def __init__(self, *,
               n_dep_labels: int,
               word_embeddings: Embeddings,
               pos_embedding_dim: int = 32,
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
               pos_embedding_dim: int = 32,
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
                                      input_dim=35,
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
               pos_embedding_dim: int = 32,
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
                                      input_dim=35, output_dim=32,
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


class LSTMSeqDecoder(tf.keras.Model):
  def __init__(self, *,
               n_labels: int,
               decoder_dim: int,
               embedding_dim: int = 32,
               batch_size: int,
               attention_type='luong',
               name="LSTMSeqDecoder"):
  
      super(LSTMSeqDecoder, self).__init__(name=name)
      self.batch_size = batch_size
      self.decoder_dim = decoder_dim
      self.attention_type = attention_type
 
      
      # Embedding layer.
      self.embedding = tf.keras.layers.Embedding(n_labels, embedding_dim,
                                                 trainable=True) 
      
      # The fundamental cell for decoder recurrent structure.
      self.decoder_lstm_cell=layers.LSTMCell(self.decoder_dim)
      
     
      
      # Create attention mechanism with memory = None
      self.attention_mechanism = self.build_attention_mechanism(
                                            decoder_dim=self.decoder_dim,
                                            memory=None,
                                            # batch_size=batch_size,
                                            # sequence_length=sequence_length,
                                            attention_type=self.attention_type
                                            )
      
      # Wrap the attention mechanism with the fundamental lstm cell of decoder.
      self.lstm_cell = self.build_lstm_with_attention(self.batch_size)
      
      # Dense layer
      self.dense = layers.Dense(n_labels, activation="softmax")
      
      # Sampler
      self.sampler = tfa.seq2seq.sampler.TrainingSampler()
      
      # Define the decoder
      # The sampler is responsible for sampling from the output distribution
      # and producing the input for th next decoding step. The decoding loop
      # is implemented in its call method.
      self.decoder = tfa.seq2seq.BasicDecoder(cell=self.lstm_cell,
                                              sampler=self.sampler,
                                              output_layer=self.dense)
      
     
    
    
  def build_lstm_with_attention(self, batch_size):
      lstm_cell = tfa.seq2seq.AttentionWrapper(
        cell=self.decoder_lstm_cell,
        attention_mechanism=self.attention_mechanism,
        attention_layer_size=self.decoder_dim,
      )
      return lstm_cell
  
  def build_attention_mechanism(self, *, 
                                decoder_dim,
                                memory,
                                # batch_size,
                                # sequence_length, 
                                attention_type='luong'):
      """Builds attention mechanism. 
      Args:
        memory: encoder hidden states of shape 
                                [batch_size, max_length_input, encoder_dim]
        memory_sequence_length: 1d array of shape (batch_size) with every 
                                element set to max_length input
      """
      # memory_sequence_length = batch_size * [sequence_length]
      if (attention_type == 'bahdanau'):
        return tfa.seq2seq.BahdanauAttention(
                units=decoder_dim,
                memory=memory,
                # memory_sequence_length=memory_sequence_length
                )
      else:
        return tfa.seq2seq.LuongAttention(
                  units=decoder_dim,
                  memory=memory, 
                  # memory_sequence_length=memory_sequence_length
                  )
  
  
  def build_initial_state(self, batch_size, encoder_state, dtype):
    decoder_initial_state = self.lstm_cell.get_initial_state(
              batch_size=batch_size,
              dtype=dtype)
    decoder_initial_state = decoder_initial_state.clone(cell_state=encoder_state)
    return decoder_initial_state
    
    
  def call(self, inputs, initial_state, sequence_length):
    x = self.embedding(inputs)
    outputs, _, _ = self.decoder(x,
                                initial_state=initial_state,
                                sequence_length=self.batch_size*[sequence_length])
    return outputs
 
 
      