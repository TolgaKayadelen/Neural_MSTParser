import tensorflow as tf
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
    self.loss_fn = losses.CategoricalCrossentropy(from_logits=True)
    self.model = self._create_umcompiled_model(n_output_classes, output_name)
    self.optimizer = optimizers.Adam(0.001, amsgrad=True)
    self.train_metrics = metrics.CategoricalAccuracy()
    self.val_metrics = metrics.CategoricalAccuracy()
    self._compiled = False

  def _set_pretrained_embeddings(self, 
                                 embeddings: Embeddings) -> layers.Embedding:
    """Builds a pretrained keras embedding layer from an Embeddings object."""
    embed = layers.Embedding(embeddings.vocab_size,
                             embeddings.embedding_dim,
                             weights=[embeddings.index_to_vector],
                             trainable=False,
                             name="word_embeddings")
    embed.build((None,))
    return embed
  
  def __str__(self):
    return str(self.model.summary())
  
  def _plot(self, name="multi_input_and_output_model.png"):
    tf.keras.utils.plot_model(self.model, name, show_shapes=True)
  
  def _create_umcompiled_model(self, n_classes, output_name) -> tf.keras.Model:
    word_inputs = tf.keras.Input(shape=([None]), name="words")
    cat_inputs = tf.keras.Input(shape=([None]), name="cat")
    cat_features = layers.Embedding(32, 32, name="cat_embeddings")(cat_inputs)
    word_features = self.word_embeddings_layer(word_inputs)
    concat = layers.concatenate([cat_features, word_features])
    encoded1 = layers.Bidirectional(layers.LSTM(units=128,
                                               return_sequences=True,
                                               name="encoded"))(concat)
    encoded1 = layers.Dropout(rate=0.5)(encoded1)
    encoded2 = layers.Bidirectional(layers.LSTM(units=128,
                                               return_sequences=True,
                                               name="encoded2"))(encoded1)
    encoded2 = layers.Dropout(rate=0.5)(encoded2)
    X = layers.Dense(units=n_classes, name=output_name)(encoded2)
    model = tf.keras.Model(inputs={"words": word_inputs, "cat": cat_inputs},
                           outputs=X)
    return model
  
  def _get_compiled_model(self, optimizer="adam",
                          loss="categorical_crossentropy",
                          metrics=["accuracy"]):
    logging.info("Compiling model..")
    self._compiled = True
    self.model.compile(loss=loss, optimizer=optimizer, metrics=metrics)
  
  @tf.function
  def train_step(self, words: tf.Tensor, cat:tf.Tensor, y:tf.Tensor):
    with tf.GradientTape() as tape:
      logits = self.model({"words": words, "cat": cat}, training=True)
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
  def train_fit(self, dataset):
    """Training using model.fit"""
    if not self._compiled:
      self._get_compiled_model()
  
    words = []
    pos = []
    targets = []
    for batch in dataset:
      print(batch)
      words.append(batch["words"])
      pos.append(batch["pos"])
      targets.append(batch["targets"])
    print("===")
    
    words = tf.concat(words, 0)
    pos = tf.concat(pos, 0)
    targets = tf.concat(targets, 0)
    print(words)
    print(pos)
    print(targets)
    
    # NOTE
    # a = tf.constant([1,2], [3,4]) # a is of shape (2,2)
    # b = tf.constant([1,2,3], [4,5,6]) # b is of shape (2,3)
    # paddings = tf.constat([0,0], [0,1]) # pad along columns with length 1. 
    # a = tf.pad(a, paddings) # now a also has shape (2,3)
    # c  = tf.concat([a,b], 0) # c is of shape (4,3)
    
  def train_custom(self, dataset: Dataset, epochs: int = 10):
    """Custom training method."""
    for epoch in range(epochs):
      # Reset training metrics at the end of each epoch
      self.train_metrics.reset_states()
      logging.info(f"*********** Training epoch: {epoch} ***********\n")
      start_time = time.time()
      for step, batch in enumerate(dataset):
        words = batch["words"]
        cat = batch["category"]
        y = batch["pos"]
        loss_value = self.train_step(words, cat, y)
      # Display metrics at the end of each epoch
      train_acc = self.train_metrics.result()
      logging.info(f"Training accuracy: {train_acc}")
          
      # Log the time it takes for one epoch
      logging.info(f"Time for epoch: {time.time() - start_time}\n")
  

def _set_up_features(train_features: List[str],
                     label=str) -> List[SequenceFeature]:
  sequence_features = []
  for feat in train_features:
    sequence_features.append(preprocessor.SequenceFeature(name=feat))
  label_dict = LabelReader.get_labels(label).labels
  label_indices = list(label_dict.values())
  label_feature = preprocessor.SequenceFeature(
    name=label, values=label_indices, n_values=len(label_indices),
    is_label=True, one_hot=True)
  print(label_feature)
  sequence_features.append(label_feature)
  return sequence_features


def main(args):
  if args.train:
    embeddings = nn_utils.load_embeddings()
    word_embeddings = Embeddings(name= "word2vec", matrix=embeddings)
    prep = preprocessor.Preprocessor(word_embeddings=word_embeddings)
    sequence_features = _set_up_features(args.train_features, args.label)
    
    if args.dataset:
      dataset = prep.read_dataset_from_tfrecords(
                                 features=sequence_features,
                                 records="./input/test11.tfrecords")
    else:
      dataset = prep.make_dataset_from_generator(
        path=os.path.join(_DATA_DIR, args.treebank),
        features=sequence_features
      )
  label_feature = next((f for f in sequence_features if f.is_label), None)
  mylstm = BLSTM(word_embeddings=prep.word_embeddings,
                 n_output_classes=label_feature.n_values,
                 output_name=args.label)
  mylstm.train_custom(dataset, 30)

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--train", type=bool, default=False,
                      help="Trains a new model.")
  parser.add_argument("--treebank", type=str,
                      default="treebank_train_0_10.pbtxt")
  parser.add_argument("--dataset",
                      help="path to a prepared tf.data.Dataset")
  parser.add_argument("--train_features", type=list, default=["words", "category"],
                      help="features to use to train the model.")
  parser.add_argument("--label", type=str, default="pos",
                      help="labels to predict.")
  args = parser.parse_args()
  main(args)
  