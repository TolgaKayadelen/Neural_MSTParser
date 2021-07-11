import collections
import logging
import os
import sys
import time

from input import embeddor
import tensorflow as tf
import tensorflow_addons as tfa
import matplotlib as mpl
import numpy as np

from parser.nn import architectures
from proto import metrics_pb2
from tagset.reader import LabelReader
from tensorflow.keras import layers, metrics, losses, optimizers
from typing import List, Dict, Tuple
from util.nn import nn_utils

# Set up basic configurations
logging.basicConfig(format='%(levelname)s : %(message)s', level=logging.INFO)
np.set_printoptions(threshold=np.inf)
mpl.style.use("seaborn")

# Set up type aliases
Dataset = tf.data.Dataset
Embeddings = embeddor.Embeddings

# Path for saving or loading pretrained models.
_MODEL_DIR = "./model/nn/pretrained/labeler"

class Seq2SeqLabeler:
  """A sequence to sequence dependency labeler."""
  
  def __init__(self, *, word_embeddings: Embeddings,
              n_output_classes: int,
              encoder_dim: int,
              decoder_dim: int,
              batch_size: int,
              model_name: str = "seq2seq_labeler"):
    # Embeddings
    self.word_embeddings = word_embeddings
    
    self.batch_size = batch_size
    
    self.encoder_dim = encoder_dim
    
    self.decoder_dim = decoder_dim
        
    # Optimizer
    self.optimizer=tf.keras.optimizers.Adam(0.001, beta_1=0.9, beta_2=0.9)

    self.metric = metrics.SparseCategoricalAccuracy()
    
    # Number of output labels. Used for dependency label prediction only.
    self._n_output_classes = n_output_classes
    
    # The encoder 
    self.encoder = architectures.LSTMEncoder(word_embeddings=word_embeddings,
                                            encoder_dim=encoder_dim,
                                            batch_size=batch_size)
    
    # The decoder 
    self.decoder = architectures.LSTMDecoder(n_labels=n_output_classes,
                                             decoder_dim=self.decoder_dim,
                                             batch_size=self.batch_size,
                                             embedding_dim=256
                                             )
    self.scce = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)    
    
  
  def test_encoder_fn(self, dataset):
    encoder = architectures.LSTMEncoder(word_embeddings=self.word_embeddings,
                                        encoder_dim=self.encoder_dim,
                                        batch_size=self.batch_size,
                                        )
    sample_hidden = encoder.init_state(self.batch_size)
    for step, batch in enumerate(dataset):
      example_input_batch = {
        "words": batch["words"],
        "pos": batch["pos"],
        "morph": tf.dtypes.cast(batch["morph"], tf.float32)
      }
    sample_output, sample_h, sample_c = encoder(example_input_batch, sample_hidden)
    print ('Encoder output shape: (batch size, sequence length, units) {}'.format(sample_output.shape))
    print ('Encoder h vector shape: (batch size, units) {}'.format(sample_h.shape))
    print ('Encoder c vector shape: (batch size, units) {}'.format(sample_c.shape))
    input("press to cont.")
    return sample_output, sample_h, sample_c
    
  
  def test_decoder_fn(self, sample_output, sample_h, sample_c):
    decoder = architectures.LSTMDecoder(n_labels=2,
                                        decoder_dim=self.decoder_dim,
                                        embedding_dim=32,
                                        batch_size=self.batch_size)
    # some random decoder input.
    sample_x = tf.random.uniform(shape=(4, 15)) # (batch_size, max_length_output)
    decoder.attention_mechanism.setup_memory(sample_output)
    initial_state = decoder.build_initial_state(4, [sample_h, sample_c], tf.float32)
    print("initial state ", initial_state)
    input("press to cont.")
    
    sample_decoder_outputs=decoder(sample_x, initial_state, sequence_length=15)
    # print(tf.argmax(sample_decoder_outputs, axis=-1))
    print("Decoder outputs shape ", sample_decoder_outputs.rnn_output.shape)
    print(sample_decoder_outputs)
    input("press to cont.")
  
  def label_loss(self, dep_labels, label_preds):
    """Computes label loss and label predictions
    Args:
      dep_labels: tf.Tensor of shape (batch_size, seq_len, n_labels). Holding
        correct labels for each token as a one hot vector.
      label_preds: tf.Tensor of shape (batch_size, seq_len, n_labels). Holding
        probability scores for each token's label prediction.
    Returns:
      loss: cross entropy loss.
      n_correct_labels: number of correct labels in the batch.
    """
    
    mask = tf.math.logical_not(tf.math.equal(dep_labels, 0))
    # mask = tf.cast(mask, dtype=tf.int64)
    # true_labels_ragged = tf.ragged.boolean_mask(dep_labels, mask)
    # predictions_ragged = tf.ragged.boolean_mask(tf.argmax(label_preds, 2), mask)
    true_labels = tf.boolean_mask(dep_labels, mask)
    predictions = tf.boolean_mask(tf.argmax(label_preds, 2), mask)
    
    self.metric.update_state(dep_labels, label_preds, sample_weight=mask)
  
    n_correct_labels = np.sum(true_labels == predictions)
    n_tokens = len(true_labels)
    loss = self.scce(dep_labels, label_preds, sample_weight=mask)

    return loss, n_correct_labels, n_tokens


  def train_step(self, *,
                 encoder_in:  Dict[str, tf.Tensor],
                 decoder_in: tf.Tensor,
                 decoder_out: tf.Tensor,
                 encoder_hidden: tf.Tensor
                 ):
    """Runs one training step.
    
    Args:
      encoder_in: a Dict of tf.Tensor objects consisting of:
          words: A tf.Tensor of word indexes of shape (batch_size, seq_len)
                where the seq_len is padded with 0s on the right.
          pos: A tf.Tensor of pos indexes of shape (batch_size, seq_len), of the
                where same shape as words.
          morph: A tf.Tensor of (batch_size, seq_len, n_morph)
      decoder_in:  A tf.Tensor dependency label indexes representing dep_labels
                for each token, of shape (batch_size, seq_len).
      decoder_out: A tf.Tensor of dependency label indexes, offset from decoder
                in by one timestep. These are the true labels
    Returns:
      loss.
    """
    with tf.GradientTape() as tape:
      encoder_out, enc_h, enc_c = self.encoder(encoder_in, encoder_hidden,
                                               training=True)
      # print("encoder out ", encoder_out)
      # input("press to cont.")
      
      # Set the AttentionMechanism object with encoder_outputs.
      self.decoder.attention_mechanism.setup_memory(encoder_out)
      
      
      # Create AttentionWrapperState as initial state for the decoder
      decoder_initial_state = self.decoder.build_initial_state(
        batch_size=self.batch_size, encoder_state=[enc_h, enc_c],
        dtype=tf.float32 
      )
      # print("decoder initial state ", decoder_initial_state)
      # input("press to cont.")
      
      
      pred = self.decoder(decoder_in, decoder_initial_state,
                          sequence_length=decoder_out.shape[1])
      
      logits = pred.rnn_output
      # print("pred ", pred)
      # print("logits ", logits)
      # input("press to cont.")
      print(f"target ", decoder_out)
      print(f"predictions ", pred.sample_id)
      # input("press to cont.")
      
      loss, n_correct_labels, n_tokens = self.label_loss(decoder_out, logits)
    
    variables = self.encoder.trainable_variables + self.decoder.trainable_variables
    gradients = tape.gradient(loss, variables)
    self.optimizer.apply_gradients(zip(gradients, variables))
    
    return loss, n_correct_labels, n_tokens
  

  def train(self, dataset: Dataset, epochs: int=10, test_data: Dataset=None):
    """Custom training method.
    
    Args:
      dataset: Dataset containing features and labels.
      epochs: number of training epochs
      test_data: Dataset containing features and labels for test set.
    Returns:
      history: logs of scores and losses at the end of training.
    """
    
    for epoch in range(1, epochs+1):
      # Reset the states before starting the new epoch.
      self.metric.reset_states()
      total_loss = 0
      total_correct = 0
      total_tokens = 0
      
      encoder_hidden = self.encoder.init_state(self.batch_size)
      # print("encoder hidden ", encoder_hidden)
      # input("press to cont.")
      
      logging.info(f"\n\n{'->' * 12} Training Epoch: {epoch} {'<-' * 12}\n\n")
      
      start_time = time.time()
      
      # Start the training loop for this epoch.
      for step, batch in enumerate(dataset):
       
        # Read the data and set up the encoder and decoder input/output.
        encoder_in = {
          "words": batch["words"],
          "pos": batch["pos"],
          "morph": tf.dtypes.cast(batch["morph"], tf.float32)
        }
        
        decoder_in = batch["dep_labels"] 
        
        decoder_out = tf.keras.preprocessing.sequence.pad_sequences(
          decoder_in[:, 1:], maxlen=decoder_in.shape[1],
          padding="post", dtype="int64")
        
        # print("decoder in ", decoder_in)
        # print("decoder out ", decoder_out)
        
        # Run forward propagation to get losses and predictions for this step.
        loss, n_correct_labels, n_tokens = self.train_step(
                                              encoder_in=encoder_in,
                                              decoder_in=decoder_in,
                                              decoder_out=decoder_out,
                                              encoder_hidden=encoder_hidden
                                            )
        total_correct += n_correct_labels
        total_tokens += n_tokens
        # logging.info(f"loss: {loss}")
        # logging.info(f"n_correct labels: {n_correct_labels}")
       
          
      # print accuracy at the end of epoch
      print("total correct ", total_correct)
      print("total tokens ", total_tokens)
      accuracy = total_correct / total_tokens
      logging.info(f"accuracy: {accuracy}")
      logging.info(f"metric: {self.metric.result().numpy()}")
      
      # if epoch % 1 == 0 and test_data:
      #  accuracy = self.predict(dataset=test_data)
      #  logging.info(f"Test accuracy after epoch {epoch}: {accuracy}")
        
      logging.info(f"Time for epoch: {time.time() - start_time}\n")
   
    

  def predict(self, *, dataset: Dataset, heads: bool=True, labels: bool=True):
    """Tests the performance on a test dataset."""
    
    label_reader = LabelReader.get_labels("dep_labels")
    label_dict = label_reader.labels
    # print("label dict ", label_dict)
    
    accuracy = metrics.Accuracy()
    accuracy.reset_states()
    
    for example in dataset:
      # accuracy.reset_states()
      n_tokens = example["words"].shape[1]
      # logging.info(f"Number of words: {n_tokens}")
      # logging.info(f"Correct labels: {example['dep_labels']}")
      
      
      encoder_in = {
        "words": example["words"],
        "pos": example["pos"],
        "morph": tf.dtypes.cast(example["morph"], tf.float32)
      }
      
      encoder_state = self.encoder.init_state(batch_size=1)
      encoder_out, encoder_state = self.encoder(encoder_in, encoder_state)
      decoder_state = encoder_state
      
      # start the decoder with the start character
      decoder_in = tf.expand_dims(tf.constant([label_dict["TOP"]]), axis=0)
   
      # Add the corresponding token to the predictions.
      predicted_label_seq = [label_reader.itov(decoder_in.numpy()[0][0])]
      predicted_label_idx = [decoder_in.numpy()[0][0]]
      # print("predicted_label_idx ", predicted_label_idx)
      
      # Sequence generation loop
      for i in range(1, int(n_tokens)):
        decoder_pred, decoder_state = self.decoder(decoder_in, decoder_state)
        # print("decoder pred ", decoder_pred)
        predicted_label_index = tf.argmax(decoder_pred, axis=-1)
        # print(predicted_label_index.numpy()[0][0])
        predicted_label = label_reader.itov(predicted_label_index.numpy()[0][0])
        # print("predicted label ", predicted_label)
        predicted_label_seq.append(predicted_label)
        predicted_label_idx.append(predicted_label_index.numpy()[0][0])
        # print("labels so far ", predicted_label_seq)
        decoder_in = predicted_label_index
      
      # print(example["dep_labels"])
      original_labels = [label_reader.itov(label) for label
                         in example["dep_labels"][0, :].numpy()]
      print("original labels ", original_labels)
      print("predicted labels ", predicted_label_seq)
      # print(example["dep_labels"])
      # print(predicted_label_idx)
      if not len(original_labels) == len(predicted_label_seq):
        raise RuntimeError("Fatal: Original and Predicted Label lengths should match!")
      
      accuracy.update_state(example["dep_labels"], 
                            tf.convert_to_tensor([predicted_label_idx])
                            )
      logging.info(f"Accuracy on test example: {accuracy.result().numpy()}")
    
    
    return accuracy.result().numpy()
      
  def save(self, *, suffix=0):
    """Saves the model to path."""
    model_name = self.model.name
    try:
      path = os.path.join(_MODEL_DIR, self.model.name)
      if suffix > 0:
        path += str(suffix)
        model_name = self.model.name+str(suffix)
      os.mkdir(path)
      self.model.save_weights(os.path.join(path, model_name),
                                          save_format="tf")
      logging.info(f"Saved model to {path}")
    except FileExistsError:
      logging.warning(f"A model with the same name exists, suffixing {suffix+1}")
      self.save(suffix=suffix+1)
    
  def load(self, *, name: str, suffix=None, path=None):
    """Loads a pretrained model."""
    if path is None:
      path = os.path.join(_MODEL_DIR, name)
    self.model.load_weights(os.path.join(path, name))
    logging.info(f"Loaded model from model named: {name} in: {_MODEL_DIR}")
    