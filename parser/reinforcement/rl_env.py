"""The RL environment is a model that shares the same word, pos embedding and lstm weights with the
pretrained parser.

The aim of this rl_env is to take input tokens and return their lstm encoded representations. We need such
a model separately so that the State vector is always the same and not updated within iterations.

The set_up_env() function in agent.py makes sure that the shared layers have the same weights and
the these layers are not trainable in the State model.

TODO: You'll probably deprecate this and replace with the state.py
"""


import tensorflow as tf
from input import embeddor

from parser.nn import layer_utils


# TODO: the state representation for DQN will change and will be more feature/context dependent.
class State(tf.keras.Model):
  def __init__(self, *, word_embeddings: embeddor.Embeddings,
               name="State",
               use_pos: bool = True,
               use_morph: bool = True,
               use_dep_labels=False,
               use_label_embeddings=False,
               ):
    super(State, self).__init__(name=name)
    self.use_pos = use_pos
    self.use_morph = use_morph
    self.use_dep_labels=use_dep_labels
    self.use_label_embeddings = use_label_embeddings
    self.label_embedding_dim_size = 50
    self.word_embeddings = layer_utils.EmbeddingLayer(
      pretrained=word_embeddings, trainable=False,
      name="word_embeddings"
    )

    if self.use_pos:
      self.pos_embeddings = layer_utils.EmbeddingLayer(
        input_dim=37, output_dim=32,
        name="pos_embeddings",
        trainable=True
      )

    if self.use_dep_labels and self.use_label_embeddings:
      self.label_embeddings = layer_utils.EmbeddingLayer(
        input_dim=self.dep_label_size, output_dim=self.label_embedding_dim_size,
        name="label_embeddings", trainable=True
      )

    self.concatenate = tf.keras.layers.Concatenate(name="concat")
    self.encoder = layer_utils.LSTMBlock(n_units=256,
                                         num_layers=2,
                                         name="lstm_encoder")

  def call(self, inputs):
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
      sentence_repr = self.concatenate(concat_list)
    else:
      sentence_repr = word_features
    state = self.encoder(sentence_repr)
    return state













'''
class RLLabeler(tf.keras.Model):

  def __init__(self, *,
               n_units: int = 256,
               n_output_classes: int,
               model_name="dep_labeler"):
    super(RLLabeler, self).__init__(name=model_name)
    self.dqn = tf.keras.Sequantial([
      tf.keras.Input(shape=(None, None, 388)),
      tf.keras.layers.Dense(512, activation="relu"),
      tf.keras.layers.Dense(256, activation="relu"),
      tf.keras.layers.Dense(128, activation="relu"),
      tf.keras.layers.Dense(n_output_classes)
    ],
      name="dqn")

  def compile(self, optimizer, loss_fn):
    super(RLLabeler, self).compile()
    self.optimizer = optimizer
    self.loss_fn = loss_fn

  def train_step(self, states, actions):
    print("states ", states)
    input("press ---- states in train step is printed")
    with tf.GradientTape() as tape:
      predictions = self(states)
      print("prediction ", predictions)
      print("actions ", actions)
      input("press")
      loss = self.loss_fn(actions, predictions)
      print("loss is ", loss)
    trainable_weights = self.trainable_weights
    gradients = tape.gradient(loss, trainable_weights)
    print("grads ", gradients)
    input("press")
    self.optimizer.apply_gradients(zip(gradients, trainable_weights))
    return loss
'''
