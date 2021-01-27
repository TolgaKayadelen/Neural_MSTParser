import tensorflow as tf
import time
from tensorflow.keras import layers, metrics, losses, optimizers
from input import preprocessor
from util.nn import nn_utils

import logging
logging.basicConfig(format='%(levelname)s : %(message)s', level=logging.INFO)

Embeddings = preprocessor.Embeddings




class BLSTM:
  
  def __init__(self, *, word_embeddings: Embeddings):
    self.word_embeddings_layer = self._set_pretrained_embeddings(
                                                                word_embeddings)
    self.loss_fn = losses.CategoricalCrossentropy(from_logits=True)
    self.model = self._create_umcompiled_model()
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
  
  def _create_umcompiled_model(self) -> tf.keras.Model:
    word_inputs = tf.keras.Input(shape=([None]), name="words")
    pos_inputs = tf.keras.Input(shape=([None]), name="pos")
    pos_features = layers.Embedding(32, 32, name="pos_embeddings")(pos_inputs)
    word_features = self.word_embeddings_layer(word_inputs)
    concat = layers.concatenate([pos_features, word_features])
    encoded = layers.Bidirectional(layers.LSTM(units=128,
                                               return_sequences=True,
                                               name="encoded"))(concat)
    encoded = layers.Dropout(rate=0.5)(encoded)
    encoded2 = layers.Bidirectional(layers.LSTM(units=128,
                                               return_sequences=True,
                                               name="encoded2"))(encoded)
    encoded2 = layers.Dropout(rate=0.5)(encoded2)
    X = layers.Dense(units=2)(encoded2)
    # NOTE: as you have from_logits=True in your loss function. You don't use the activation layer.
    # X = tf.keras.layers.Activation("softmax", name="result")(X)
    
    model = tf.keras.Model(inputs={"words": word_inputs, "pos": pos_inputs},
                           outputs=X)
    return model
  
  def _get_compiled_model(self, optimizer="adam",
                          loss="categorical_crossentropy",
                          metrics=["accuracy"]):
    logging.info("Compiling model..")
    self._compiled = True
    self.model.compile(loss=loss, optimizer=optimizer, metrics=metrics)
  
  @tf.function
  def train_step(self, words: tf.Tensor, pos:tf.Tensor, y:tf.Tensor):
    # print("y shape ", y.shape)
    # input("press to cont")
    with tf.GradientTape() as tape:
      logits = self.model({"words": words, "pos": pos}, training=True)
      # print("logits shape ", logits.shape)
      # input("press to cont")
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
    
  def train_custom(self, dataset, epochs=30):
    """Custom training method."""
    for epoch in range(epochs):
      # Reset training metrics at the end of each epoch
      self.train_metrics.reset_states()
      logging.info(f"*********** Training epoch: {epoch} ***********")
      start_time = time.time()
      for step, batch in enumerate(dataset):
        words = batch["words"]
        pos = batch["pos"]
        y = batch["targets"]
        loss_value = self.train_step(words, pos, y)
        
        # Log
        # if step % 10 == 0: 
        #  logging.info(f"Training loss for one batch at step {step} :  {float(loss_value)}")
      
      # Display metrics at the end of each epoch
      train_acc = self.train_metrics.result()
      logging.info(f"Training accuracy: {train_acc}")
          
      # Log the time it takes for one epoch
      logging.info(f"Time for epoch: {time.time() - start_time}")
  

if __name__== "__main__":
  embeddings = nn_utils.load_embeddings()
  word_embeddings = Embeddings(name= "word2vec", matrix=embeddings)
  
  prep = preprocessor.Preprocessor(word_embeddings=word_embeddings)
  
  dataset = prep.make_dataset_from_generator(
    path="data/UDv23/Turkish/training/treebank_0_3.pbtxt",
    features=[preprocessor.SequenceFeature(name="words"),
              preprocessor.SequenceFeature(name="pos"),
              preprocessor.SequenceFeature(name="targets")]
  )
  for batch in dataset:
    print(batch)
  mylstm = BLSTM(word_embeddings=prep.word_embeddings)
  print(mylstm)
  mylstm.train_custom(dataset)
  