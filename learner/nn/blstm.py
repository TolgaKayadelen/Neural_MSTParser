import tensorflow as tf
import numpy as np
np.set_printoptions(threshold=np.inf)
import argparse
import os
import time
from tensorflow.keras import layers, metrics, losses, optimizers
from typing import List, Dict
from input import preprocessor
from tagset.reader import LabelReader
from util.nn import nn_utils

import logging
logging.basicConfig(format='%(levelname)s : %(message)s', level=logging.INFO)


_DATA_DIR = "data/UDv23/Turkish/training"

Dataset = tf.data.Dataset
Embeddings = preprocessor.Embeddings
SequenceFeature = preprocessor.SequenceFeature

class BLSTM:
  
  def __init__(self, *, word_embeddings: Embeddings,
               n_output_classes: int, output_name: str):
    self.word_embeddings_layer = self._set_pretrained_embeddings(
                                                                word_embeddings)
    self.model = self._create_uncompiled_model(n_output_classes, output_name)                                                            
    self.loss_fn = losses.CategoricalCrossentropy(from_logits=True)
    self.optimizer = optimizers.Adam(0.01, amsgrad=True)
    self.train_metrics = metrics.CategoricalAccuracy()
    self.val_metrics = metrics.CategoricalAccuracy()
    self._compiled = False
    self._n_output_classes = n_output_classes
  
  def __str__(self):
    return str(self.model.summary())

  def _set_pretrained_embeddings(self, 
                                 embeddings: Embeddings) -> layers.Embedding:
    """Builds a pretrained keras embedding layer from an Embeddings object."""
    embed = tf.keras.layers.Embedding(embeddings.vocab_size,
                             embeddings.embedding_dim,
                             trainable=False,
                             name="word_embeddings")
    embed.build((None,))
    # print(embeddings.index_to_vector[493047])
    # input("press to cont.")
    embed.set_weights([embeddings.index_to_vector])
    return embed
  
  def _plot(self, name="multi_input_and_output_model.png"):
    tf.keras.utils.plot_model(self.model, name, show_shapes=True)
  
  def _create_uncompiled_model(self, n_classes, output_name) -> tf.keras.Model:
    word_inputs = tf.keras.Input(shape=(None,), name="words")
    # cat_inputs = tf.keras.Input(shape=([None]), name="cat")
    # cat_features = layers.Embedding(32, 32, name="cat_embeddings")(cat_inputs)
    word_features = self.word_embeddings_layer(word_inputs)
    # concat = layers.concatenate([cat_features, word_features])
    X = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(units=128,
      return_sequences=True, name="encoded"))(word_features)
    X = tf.keras.layers.Dropout(rate=0.5)(X)
    X = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(units=128,
                                         return_sequences=True,
                                         name="encoded2"))(X)
    X = tf.keras.layers.Dropout(rate=0.5)(X)
    X = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(units=32,
                                         return_sequences=True,
                                         name="encoded3"))(X)
    X = tf.keras.layers.Dense(units=n_classes, name=output_name)(X)
    # X = tf.keras.layers.Activation("softmax")(X)
    model = tf.keras.Model(inputs={"words": word_inputs}, outputs=X)
    return model
  
  def _get_compiled_model(self, optimizer=optimizers.Adam(0.01),
                          loss="categorical_crossentropy",
                          metrics=["accuracy"]):
    logging.info("Compiling model..")
    self._compiled = True
    self.model.compile(loss=loss,
                       optimizer=optimizer,
                       metrics=metrics)
  
  @tf.function
  def train_step(self, words: tf.Tensor, y:tf.Tensor):
    with tf.GradientTape() as tape:
      logits = self.model({"words": words}, training=True)
      # Send the **LOGITS** to the loss function.
      loss_value = self.loss_fn(y, logits)
    grads = tape.gradient(loss_value, self.model.trainable_weights)
    self.optimizer.apply_gradients(zip(grads, self.model.trainable_weights))
    # Update train metrics
    self.train_metrics.update_state(y, logits)
    return loss_value
  
  @tf.function
  def test_step(self, x, y):
    val_logits = self.model(x["words"], x["pos"], training=False)
    self.val_metrics.update_state(y, val_logits)
  
  # TODO finish this method. You need to pad all the tensors in all the batches
  # to the same value for this method to work.
  def train_fit(self, dataset, epochs=30):
    """Training using model.fit"""
    if not self._compiled:
      self._get_compiled_model()
    self.model.compile(optimizer="adam", loss="categorical_crossentropy",
                       metrics=["accuracy"])
    counter = 0
    for batch in dataset:
      counter += 1
      words = batch["words"]
      pos = batch["pos"]
      if counter > 1:
        raise RuntimeError(
           "Cannot have more than one batch if using model.fit")
    # print(words)
    # print(pos)
    labels = tf.one_hot(pos, self._n_output_classes)
    # print(labels)
    # input("press to cont")
    self.model.fit(x={"words":words}, y=labels, batch_size=10, epochs=epochs)
    
    """
    words = tf.concat(words, 0)
    pos = tf.concat(pos, 0)
    targets = tf.concat(targets, 0)
    """    
    # NOTE
    # a = tf.constant([1,2], [3,4]) # a is of shape (2,2)
    # b = tf.constant([1,2,3], [4,5,6]) # b is of shape (2,3)
    # paddings = tf.constat([0,0], [0,1]) # pad along columns with length 1. 
    # a = tf.pad(a, paddings) # now a also has shape (2,3)
    # c  = tf.concat([a,b], 0) # c is of shape (4,3)
    
  def train_custom(self, dataset: Dataset, epochs: int=10):
    """Custom training method."""
    for epoch in range(epochs):
      # Reset training metrics at the end of each epoch
      self.train_metrics.reset_states()
      logging.info(f"*********** Training epoch: {epoch} ***********\n")
      start_time = time.time()
      for step, batch in enumerate(dataset):
        words = batch["words"]
        # cat = batch["category"]
        y = tf.one_hot(batch["pos"], self._n_output_classes)
        loss_value = self.train_step(words, y)
      # Display metrics at the end of each epoch
      train_acc = self.train_metrics.result()
      logging.info(f"Training accuracy: {train_acc}")
          
      # Log the time it takes for one epoch
      logging.info(f"Time for epoch: {time.time() - start_time}\n")
  

# TODO: the set up of the features should be done by the preprocessor class.
def _set_up_features(features: List[str], label=str) -> List[SequenceFeature]:
  sequence_features = []
  for feat in features:
    if not feat == label:
      sequence_features.append(preprocessor.SequenceFeature(name=feat))
    else:
      label_dict = LabelReader.get_labels(label).labels
      label_indices = list(label_dict.values())
      label_feature = preprocessor.SequenceFeature(
        name=label, values=label_indices, n_values=len(label_indices),
        is_label=True)
      sequence_features.append(label_feature)
  return sequence_features


def main(args):
  if args.train:
    embeddings = nn_utils.load_embeddings()
    word_embeddings = Embeddings(name= "word2vec", matrix=embeddings)
    prep = preprocessor.Preprocessor(word_embeddings=word_embeddings)
    sequence_features = _set_up_features(args.features, args.label)
    
    if args.dataset:
      dataset = prep.read_dataset_from_tfrecords(
                                 batch_size=10,
                                 features=sequence_features,
                                 records="./input/test50.tfrecords")
    else:
      dataset = prep.make_dataset_from_generator(
        path=os.path.join(_DATA_DIR, args.treebank),
        batch_size=50, 
        features=sequence_features
      )

  label_feature = next((f for f in sequence_features if f.is_label), None)
  mylstm = BLSTM(word_embeddings=prep.word_embeddings,
                 n_output_classes=label_feature.n_values,
                 output_name=args.label)
  print(mylstm)
  # mylstm.train_custom(dataset, 20)
  mylstm.train_fit(dataset, 30)

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--train", type=bool, default=False,
                      help="Trains a new model.")
  parser.add_argument("--treebank", type=str,
                      default="treebank_train_0_50.pbtxt")
  parser.add_argument("--dataset",
                      help="path to a prepared tf.data.Dataset")
  parser.add_argument("--features", type=list, default=["words", "pos"],
                      help="features to use to train the model.")
  parser.add_argument("--label", type=str, default="pos",
                      help="labels to predict.")
  args = parser.parse_args()
  main(args)
  