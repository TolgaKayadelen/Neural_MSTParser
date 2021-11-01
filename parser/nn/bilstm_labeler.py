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


"""
Usage example:
bazel-bin/parser_main/nn/main --train --parser_type=bilstm_labeler \
--train_treebank=treebank_train_0_50.pbtxt \
--test_treebank=treebank_0_3_gold.pbtxt \
--epochs=5 --model_name=test --predict=labels --labels=dep_labels
"""

# Set up basic configurations
logging.basicConfig(format='%(levelname)s : %(message)s', level=logging.INFO)
np.set_printoptions(threshold=np.inf)
mpl.style.use("seaborn")

# Set up type aliases
Dataset = tf.data.Dataset
Embeddings = embeddor.Embeddings
Metrics = metrics_pb2.Metrics

# Path for saving or loading pretrained models.
_MODEL_DIR = "./model/nn/pretrained/labeler"

class BiLSTMLabeler:
  """A bilstm labeler.
  Can be used for any kind of sequence labeling tasks.
  """
  
  def __init__(self, *, word_embeddings: Embeddings,
              n_output_classes: int,
              n_units: int,
              model_name: str = "bilstm_labeler"):
    # Embeddings
    self.word_embeddings = word_embeddings
    
    self.lstm_units = n_units
        
    # Optimizer
    self.optimizer=tf.keras.optimizers.Adam(0.001, beta_1=0.9, beta_2=0.9)

    self.metric = metrics.SparseCategoricalAccuracy()
    
    # Number of output labels. Used for dependency label prediction only.
    self._n_output_classes = n_output_classes
    
    self.model = self._labeling_model(model_name=model_name)
    
    self.loss_fn = losses.SparseCategoricalCrossentropy()
    
    self.model_name = model_name

    self.metrics = self._metrics()

  def _labeling_model(self, *, model_name: str) -> tf.keras.Model:
    """Creates a labeling model."""
    
    word_inputs = tf.keras.Input(shape=(None,), name="words")
    pos_inputs = tf.keras.Input(shape=(None,), name="pos")
    morph_inputs = tf.keras.Input(shape=(None, 66), name="morph")
    # label_inputs = tf.keras.Input(shape=(None,), name="labels")
    # inputs = {"words": word_inputs,
    #          "pos": pos_inputs,
    #          "morph": morph_inputs
    #         }
    inputs = {"words": word_inputs,
              "morph": morph_inputs}

    model = architectures.LSTMLabelingModel(
                                  word_embeddings=self.word_embeddings,
                                  n_units=256,
                                  n_output_classes=self._n_output_classes,
                                  use_pos=False,
                                  use_morph=True,
                                  name=model_name)
    model(inputs=inputs)
    return model
  
  def __str__(self):
    return str(self.model.summary())
  
  def plot(self, name=None):
    if name is None:
      name = self.model_name
    tf.keras.utils.plot_model(self.model, name, show_shapes=True)

  def _update_metrics(self, train_metrics, test_metrics, loss_metrics):
    all_metrics = {**train_metrics, **test_metrics, **loss_metrics}
    for key, value in all_metrics.items():
      if not self.metrics.metric[key].tracked:
        logging.info(f"Metric {key} is not tracked!")
      else:
        self.metrics.metric[key].value_list.value.append(value)

  def _metrics(self) -> Metrics:
    metrics_list = ("ls", "ls_test", "label_loss")
    metrics = nn_utils.set_up_metrics(*metrics_list)
    return metrics

  def _compute_metrics(self, *, stats: Dict, padded=False):
    """Computes Label Score."""
    return_dict = {}
    label_score = stats["n_correct_labels"] / stats["n_tokens"]
    return_dict["ls"] = label_score
    if padded:
      label_score_padded = stats["n_label_correct_padded"] / stats["n_tokens_padded"]
      return_dict["ls"] = label_score
    return return_dict
  
  def label_loss(self, dep_labels, label_preds):
    """Computes label loss and label predictions
    Args:
      dep_labels: tf.Tensor of shape (batch_size, seq_len, n_labels). Holding
        correct labels for each token as a one hot vector.
      label_preds: tf.Tensor of shape (batch_size, seq_len, n_labels). Holding
        probability scores (logits) for each token's label prediction.
    Returns:
      loss: cross entropy loss.
      n_correct_labels: number of correct labels in the batch.
    """
    
    mask = tf.math.logical_not(tf.math.equal(dep_labels, 0))

    true_labels = tf.boolean_mask(dep_labels, mask)
    predictions = tf.boolean_mask(tf.argmax(label_preds, 2), mask)

    self.metric.update_state(dep_labels, label_preds, sample_weight=mask)
  
    n_correct_labels = np.sum(true_labels == predictions)
    n_tokens = len(true_labels)
    
    loss = self.loss_fn(dep_labels, label_preds, sample_weight=mask)

    return loss, n_correct_labels, n_tokens


  def train_step(self, *,
                 inputs:  Dict[str, tf.Tensor],
                 target: tf.Tensor,
                 ) -> Dict:
    """Runs one training step.
    
    Args:
      inputs: a Dict of tf.Tensor objects consisting of:
          words: A tf.Tensor of word indexes of shape (batch_size, seq_len)
                where the seq_len is padded with 0s on the right.
          pos: A tf.Tensor of pos indexes of shape (batch_size, seq_len), of the
                where same shape as words.
          morph: A tf.Tensor of (batch_size, seq_len, n_morph)
      decoder_target: True labels.
    Returns:
      loss.
    """
    with tf.GradientTape() as tape:
      pred = self.model(inputs, training=True)
      loss, n_correct_labels, n_tokens = self.label_loss(target, pred["labels"])
    
    variables = self.model.trainable_weights
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
    ls_test = 0
    for epoch in range(1, epochs+1):
      # Reset the states before starting the new epoch.
      self.metric.reset_states()
      total_loss = 0
      total_correct = 0
      total_tokens = 0
      
      logging.info(f"\n\n{'->' * 12} Training Epoch: {epoch} {'<-' * 12}\n\n")
      
      start_time = time.time()
      stats = collections.Counter()
      
      # Start the training loop for this epoch.
      for step, batch in enumerate(dataset):
       
        # Read the data and set up the encoder and decoder input/output.
        inputs = {
          "words": batch["words"],
          "pos": batch["pos"],
          "morph": tf.dtypes.cast(batch["morph"], tf.float32)
        }
        
        target = batch["dep_labels"]
        
        # Run forward propagation to get losses and predictions for this step.
        loss, n_correct_labels, n_tokens = self.train_step(inputs=inputs, target=target)
        stats["n_correct_labels"] += n_correct_labels
        stats["n_tokens"] += n_tokens
        stats["total_loss"] += loss

      # Get trainin stats and metrics for epoch.
      print(f"Training stats {stats}")
      training_metrics = self._compute_metrics(stats=stats)
          
      # print accuracy at the end of epoch
      print("total correct ", stats["n_correct_labels"])
      print("total tokens ", stats["n_tokens"])
      accuracy = stats["n_correct_labels"] / stats["n_tokens"]
      logging.info(f"Training accuracy: {accuracy}")
      logging.info(f"Training metric: {self.metric.result().numpy()}")
      logging.info(f"Label loss (training): {stats['total_loss'].numpy()}")
      
      if epoch % 5 == 0 and test_data:
        ls_test = self.predict(dataset=test_data)
        logging.info(f"Test accuracy {ls_test}")
      self._update_metrics(
        train_metrics=training_metrics,
        test_metrics={"ls_test" : ls_test},
        loss_metrics={"label_loss": stats["total_loss"].numpy()}
      )
      logging.info(f"Time for epoch: {time.time() - start_time}\n")
    return self.metrics

  def predict(self, *, dataset: Dataset, heads: bool=True, labels: bool=True):
    """Tests the performance on a test dataset."""
    
    label_reader = LabelReader.get_labels("dep_labels")
    label_dict = label_reader.labels
    # print("label dict ", label_dict)
    
    accuracy = metrics.SparseCategoricalAccuracy()
    accuracy.reset_states()
    
    for example in dataset:
      # accuracy.reset_states()
      n_tokens = example["words"].shape[1]
      # logging.info(f"Number of words: {n_tokens}")
      
      inputs = {
        "words": example["words"],
        "pos": example["pos"],
        "morph": tf.dtypes.cast(example["morph"], tf.float32)
      }
      
      pred = self.model(inputs)
      
      true_labels = example["dep_labels"]
      predictions = tf.argmax(pred["labels"], axis=2)
      # print("true labels ", true_labels)
      # print("predictions ", predictions)
      
      accuracy.update_state(true_labels, pred["labels"])
      # print("accuracy on test ", accuracy.result().numpy())
     

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