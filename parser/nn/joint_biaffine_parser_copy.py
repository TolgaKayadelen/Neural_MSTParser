import argparse
import collections
import logging
import os
import sys
import time

from input import preprocessor, embeddor
import tensorflow as tf
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from parser.nn import model_builder as builder
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
SequenceFeature = preprocessor.SequenceFeature

# Set up global constants
_DATA_DIR = "data/UDv23/Turkish/training"
_TEST_DATA_DIR = "data/UDv23/Turkish/test"

class NeuralMSTParser:
  """The neural mst parser."""
  
  def __init__(self, *, word_embeddings: Embeddings,
              n_output_classes: int,
              predict: List[str] = ["edges"]):
    self.word_embeddings = word_embeddings
    self.edge_loss_fn = losses.SparseCategoricalCrossentropy(
      from_logits=True, reduction=tf.keras.losses.Reduction.NONE)
    self.label_loss_fn = losses.CategoricalCrossentropy(
      from_logits=True, reduction=tf.keras.losses.Reduction.NONE)
    self.optimizer=tf.keras.optimizers.Adam(0.001, beta_1=0.9, beta_2=0.9)
    self.edge_train_metrics = metrics.SparseCategoricalAccuracy()
    self.label_train_metrics = metrics.CategoricalAccuracy()
    self._n_output_classes = n_output_classes
    self._predict = predict
    self.model = self._parsing_model()
    assert(self._predict == self.model.predict), "Inconsistent configuration!"

  def __str__(self):
    return str(self.model.summary())
  
  def plot(self, name="neural_joint_mstparser.png"):
     tf.keras.utils.plot_model(self.model, name, show_shapes=True)
  
  def _parsing_model(self) -> tf.keras.Model:
    """Creates an NN model for edge factored dependency parsing."""
    
    word_inputs = tf.keras.Input(shape=(None,), name="words")
    pos_inputs = tf.keras.Input(shape=(None,), name="pos")
    morph_inputs = tf.keras.Input(shape=(None, 66), name="morph")
    inputs = {"words": word_inputs, "pos": pos_inputs, "morph": morph_inputs}
    
    model = builder.ParsingModel(n_dep_labels=self._n_output_classes,
                                 word_embeddings=self.word_embeddings,
                                 predict=self._predict)
    model(inputs=inputs)
    return model
  

  def _compute_metrics(self, *, stats: Dict):
    """Computes UAS, LS, and LAS metrics."""
    # Compute UAS with and without padding for this epoch.
    uas_with_pad = stats["n_edge_correct_with_pad"] / stats["n_tokens_with_pad"]
    uas_without_pad = stats["n_edge_correct"] / stats["n_tokens"]
    
    return_dict = {"uas_with_pad": uas_with_pad,
                   "uas_without_pad": uas_without_pad}
    
    # Compute Label Score (LS) with and without padding for this epoch
    if "n_label_correct_with_pad" in stats:
      label_score_pad = stats["n_label_correct_with_pad"] / stats["n_tokens_with_pad"]
      label_score_without_pad = stats["n_label_correct"] / stats["n_tokens"]
      return_dict["label_score_pad"] = label_score_pad
      return_dict["label_score_without_pad"] = label_score_without_pad
    
    # Compute Labeled Attachment Score
    if "n_las" in stats:
      las_train = stats["n_las"] / stats["n_tokens"]
      return_dict["las_train"] = las_train
    
    return return_dict
  
  def _compute_n_correct(self, *, preds: Dict):
    """Computes n correct edge and label predictions from system predictons."""
    
    # Calculate correct edge predictions
    edge_pred = preds["edge_pred"]
    # edge_correct_with_pad = preds["edge_correct_with_pad"]
    correct_heads = preds["h"]
    
    # n_correct edge predictions with padding
    edge_correct_with_pad = (edge_pred == correct_heads)
    n_edge_correct_with_pad = np.sum(edge_correct_with_pad)

    # n_correct edge predictions without padding
    edge_correct = tf.boolean_mask(edge_pred == correct_heads, preds["pad_mask"])
    n_edge_correct = np.sum(edge_correct)
    
    return_dict = {"edge_correct": edge_correct,
                   "n_edge_correct_with_pad":  n_edge_correct_with_pad, 
                   "n_edge_correct": n_edge_correct,
                   "h": preds["h"]}
    
    # Calculate correct label predictions with padding
    if "label_loss" in preds:
      label_preds = preds["label_preds"]
      correct_labels = preds["correct_labels"]
      label_correct_with_pad = (label_preds == correct_labels)
      n_label_correct_with_pad = np.sum(label_correct_with_pad)
    
      # Calculate correct label predictions without padding
      label_correct = tf.boolean_mask(label_correct_with_pad, preds["pad_mask"])
      n_label_correct = np.sum(label_correct)
      
      # Add to return dict
      return_dict["label_correct"] = label_correct
      return_dict["n_label_correct_with_pad"] = n_label_correct_with_pad
      return_dict["n_label_correct"] = n_label_correct
    
    return return_dict

  def _labeled_attachment_score(self, edges, labels, test=False):
    """Computes the number of tokens that have correct Labeled Attachment.
    
    Args: 
      edges: tf.Tensor, a boolean tensor representing edge predictions for
        tokens. Tokens whose edges are correctly predicted are represented
        with the value True and the others with the value False.
      labels: similar to the edges tensor, but for label predictions.
    
    Returns:
      n_las: int, the number of tokens that have the correct head and
        label assignment. That is, the number of tokens where the indexes in 
        the edges and labels tensors are both True.
    """

    n_token = float(len(edges))
    if not len(edges) == len(labels):
      sys.exit("FATAL ERROR: Mismatch in the number of tokens!")
    n_las = np.sum(
      [1 for tok in zip(edges, labels) if tok[0] == True and tok[1] == True])
    return n_las

  @tf.function
  def edge_loss(self, edge_scores, heads, pad_mask):
    """Computes loss for edge predictions.
    Args:
      edge_scores: A 3D tensor of (batch_size, seq_len, seq_len). This holds
        the edge prediction for each token in a sentence, for the whole batch.
        The outer dimension (second seq_len) is where the head probabilities
        for a token are represented.
      heads: A 2D tensor of (batch_size, seq_len)
      pad_mask: A 2D tensor of (batch_size, seq_len)
    Returns:
      edge_loss_with_pad: 2D tensor of shape (batch_size*seq_len, seq_len).
        Holds loss values for each of the predictions.
      edge_loss_w_o_pad: Loss values where the pad tokens are not considered.
      heads: 2D tensor of (batch_size*seq_len, 1)
      pad_mask: 2D tensor of (bath_size*se_len, 1)
    """
    n_sentences, n_words, _ = edge_scores.shape
    edge_scores = tf.reshape(edge_scores, shape=(n_sentences*n_words, n_words))
    heads = tf.reshape(heads, shape=(n_sentences*n_words, 1))
    # Pad mask is being reshaped here.
    pad_mask = tf.reshape(pad_mask, shape=(n_sentences*n_words, 1))
    
    predictions = tf.argmax(edge_scores, 1)
    predictions = tf.reshape(predictions, shape=(predictions.shape[0], 1))
    
    # Compute losses with and without pad
    edge_loss_with_pad = self.edge_loss_fn(heads, edge_scores)
    edge_loss_with_pad = tf.reshape(edge_loss_with_pad,
                                    shape=(edge_loss_with_pad.shape[0], 1))
    edge_loss_w_o_pad = tf.boolean_mask(edge_loss_with_pad, pad_mask)
        
    return edge_loss_with_pad, edge_loss_w_o_pad, predictions, heads, pad_mask

  @tf.function
  def label_loss(self, dep_labels, label_scores, pad_mask):
    """Computes label loss and label predictions
    Args:
      dep_labels: tf.Tensor of shape (batch_size, seq_len, n_labels). Holding
        correct labels for each token as a one hot vector.
      label_scores: tf.Tensor of shape (batch_size, seq_len, n_labels). Holding
        probability scores for each token's label prediction.
      pad_mask: tf.Tensor of shape (batch_size*seq_len, 1)
    Returns:
      label_loss: the label loss associated with each token.
      correct_labels: tf.Tensor of shape (batch_size*seq_len, 1). The correct
        dependency labels.
      label_preds: tf.Tensor of shape (batch_size*seq_len, 1). The predicted
        dependency labels.
    """
    label_preds = tf.reshape(tf.argmax(label_scores,
                                       axis=2),
                             shape=(pad_mask.shape)
                             )
    correct_labels = tf.reshape(tf.argmax(dep_labels,
                                          axis=2),
                                shape=(pad_mask.shape))
  
    label_loss = self.label_loss_fn(dep_labels, label_scores)
    return label_loss, correct_labels, label_preds

  @tf.function
  def train_step(self, *, words: tf.Tensor, pos: tf.Tensor, morph: tf.Tensor,
                 dep_labels: tf.Tensor, heads:tf.Tensor
                 ) -> Tuple[tf.Tensor, ...]:
    """Runs one training step.
    
    Args:
      words: A tf.Tensor of word indexes of shape (batch_size, seq_len) where
          the seq_len is padded with 0s on the right.
      pos: A tf.Tensor of pos indexes of shape (batch_size, seq_len), of the
          same shape as words.
      dep_labels: A tf.Tensor of one hot vectors representing dep_labels for
          each token, of shape (batch_size, seq_len, n_classes).
    Returns:
      edge_loss_pad: edge loss computed with padding tokens.
      edge_loss_w_o_pad: edge loss computed without padding tokens.
      edge_pred: edge predictions.
      h: correct heads.
      pad_mask: pad mask to identify padded tokens.
      label_loss: the depdendeny label loss associated with each token.
      correct_labels: tf.Tensor of shape (batch_size*seq_len, 1). The correct
          dependency labels.
      label_preds: tf.Tensor of shape (batch_size*seq_len, 1). The predicted
          dependency labels
    """
    with tf.GradientTape() as tape:
      scores = self.model({"words": words, "pos": pos, "morph": morph},
                           training=True)
      pad_mask = (words != 0)
      
      # Get edge and label scores.
      edge_scores, label_scores = scores["edges"], scores["labels"]
      edge_loss_pad, edge_loss_w_o_pad, edge_pred, h, pad_mask = self.edge_loss(
        edge_scores, heads, pad_mask)
      if "labels" in self._predict:
        # input("executing this..")
        label_loss, correct_labels, label_preds = self.label_loss(dep_labels,
                                                                  label_scores,
                                                                  pad_mask)
    
    # Compute gradients.
    if "labels" in self._predict:
      grads = tape.gradient(
        [edge_loss_pad, label_loss], self.model.trainable_weights
      )
    else:
      grads = tape.gradient(edge_loss_pad, self.model.trainable_weights)
    
    # Update the optimizer
    self.optimizer.apply_gradients(zip(grads, self.model.trainable_weights))
    
    # Update train metrics and populate return dict.
    self.edge_train_metrics.update_state(heads, edge_scores)
    return_dict = {"edge_loss_pad": edge_loss_pad,
                  "edge_loss_w_o_pad": edge_loss_w_o_pad,
                  "edge_pred": edge_pred, "h": h, "pad_mask": pad_mask}
                  
    if "labels" in self._predict:
      self.label_train_metrics.update_state(dep_labels, label_scores)
      return_dict["label_loss"] = label_loss
      return_dict["correct_labels"] = correct_labels
      return_dict["label_preds"] = label_preds

    return return_dict

  def train(self, dataset: Dataset, epochs: int=10, test_data: Dataset=None):
    """Custom training method.
    
    Args:
      dataset: Dataset containing features and labels.
      epochs: number of training epochs
      test_data: Dataset containing features and labels for test set.
    Returns:
      history: logs of scores and losses at the end of training.
    """
    history = collections.defaultdict(list)
    uas_test, label_acc_test, las_test = [0] * 3
    
    for epoch in range(1, epochs+1):
      # Reset the states before starting the new epoch.
      self.edge_train_metrics.reset_states()
      self.label_train_metrics.reset_states()
      
      logging.info(f"{'->' * 12} Training Epoch: {epoch} {'<-' * 12}\n\n")
      
      start_time = time.time()
      stats = collections.Counter()
      
      # Start the training loop for this epoch.
      for step, batch in enumerate(dataset):
        
        # Read the data and labels from the dataset.
        words, pos, heads = batch["words"], batch["pos"], batch["heads"]
        dep_labels = tf.one_hot(batch["dep_labels"], self._n_output_classes)
        # We cast the type of this tensor to float32 because in the model
        # pos and word features are passed through an embedding layer, which
        # converts them into float values implicitly.
        # TODO: maybe do this as you read in the morph values in preprocessor.
        morph = tf.dtypes.cast(batch["morph"], tf.float32)
        
        # Run forward propagation to get losses and predictions for this step.
        losses_and_preds = self.train_step(words=words, pos=pos, morph=morph,
                                           dep_labels=dep_labels, heads=heads)
    
        # Get the total number of tokens without padding for this step.
        words_reshaped = tf.reshape(words, 
                                    shape=(losses_and_preds["pad_mask"].shape))
        total_words = len(tf.boolean_mask(words_reshaped,
                                          losses_and_preds["pad_mask"]))
        
        # Get the number of correct predictions for this step.
        n_correct = self._compute_n_correct(preds=losses_and_preds)
        
        # Accumulate the stats for all steps in the epoch.
        stats["n_edge_correct_with_pad"] += n_correct["n_edge_correct_with_pad"]
        stats["n_edge_correct"] += n_correct["n_edge_correct"]
        stats["n_tokens_with_pad"] += len(n_correct["h"])
        stats["n_tokens"] += total_words
        
        # Calculate the number of tokens which has correct las for this step
        # and add it to the accumulator.
        if "labels" in self._predict:
          stats["n_label_correct"] += n_correct["n_label_correct"]
          stats["n_label_correct_with_pad"] += n_correct["n_label_correct_with_pad"]
          las_correct = self._labeled_attachment_score(
            n_correct["edge_correct"], n_correct["label_correct"])
          stats["n_las"] += las_correct     

      print(f"Training Stats: {stats}")
      
      # Get UAS, LS, and LAS after the epoch.
      train_metrics = self._compute_metrics(stats=stats)
      
      # Average the loss scores.
      edge_train_acc = self.edge_train_metrics.result()
      label_train_acc = self.label_train_metrics.result()
      
      # Compute average edge losses with and without padding
      avg_edge_loss_pad = tf.reduce_mean(losses_and_preds["edge_loss_pad"])
      avg_edge_loss_w_o_pad = tf.reduce_mean(losses_and_preds["edge_loss_w_o_pad"])
      
      # Compute average label loss (only with pad)
      if "labels" in self._predict:
        avg_label_loss = tf.reduce_mean(losses_and_preds["label_loss"])    
      
      logging.info(f"""
        UAS train (without pad): {train_metrics['uas_without_pad']}
        Edge loss (with pad): {avg_edge_loss_pad}\n""")
      
      if "labels" in self._predict:
        logging.info(f"""
          LS train (without pad) {train_metrics['label_score_without_pad']}
          LAS train (without pad) {train_metrics['las_train']}
          Label loss (with pad) {avg_label_loss}\n""")
      
      logging.info(f"Time for epoch: {time.time() - start_time}\n")
      
      # Update scores on test data at the end of every X epoch.
      if epoch % 2 == 0 and test_data:
        uas_test, label_acc_test, las_test = self.test(dataset=test_data)
        logging.info(f"UAS test: {uas_test}")
        logging.info(f"LS test: {label_acc_test}")
        logging.info(f"LAS test: {las_test}\n")
      
      # Populate stats history
      history["uas_train"].append(train_metrics['uas_without_pad'])
      history["uas_test"].append(uas_test)
      history["edge_loss_pad"].append(avg_edge_loss_pad)
      
      if "labels" in self._predict:
        history["ls_train"].append(train_metrics['label_score_without_pad']) 
        history["las_train"].append(train_metrics['las_train'])
        history["ls_test"].append(label_acc_test)
        history["las_test"].append(las_test)
        history["label_loss_pad"].append(avg_label_loss)
      
    return history

  def test(self, *, dataset: Dataset, heads: bool=True, labels: bool=True):
    """Tests the performance on a test dataset."""
    
    head_accuracy = tf.keras.metrics.Accuracy()
    label_accuracy = tf.keras.metrics.Accuracy()
    
    n_tokens = 0.0
    n_las = 0.0
    las_test = 0.0
    
    for example in dataset:
      words, pos = example["words"], example["pos"]
      morph = tf.dtypes.cast(example["morph"], tf.float32)
      n_tokens += words.shape[1]
      
      scores = self.model({"words": words, "pos": pos, "morph": morph},
                          training=False)
      
      edge_scores, label_scores = scores["edges"], scores["labels"]
     
      # TODO: in the below computation of scores, you should leave out
      # the 0th token, which is the dummy token.
      heads, dep_labels = example["heads"], example["dep_labels"]
      
      head_preds = tf.argmax(edge_scores, 2)
      te = (heads == head_preds)
      correct_edges = tf.reshape(te, shape=(te.shape[1],))
      head_accuracy.update_state(heads, head_preds)
      
      if "labels" in self._predict:
        label_preds = tf.argmax(label_scores, 2)
        tl = (dep_labels == label_preds)
        correct_labels = tf.reshape(tl, shape=(tl.shape[1],))
        n_las += self._labeled_attachment_score(correct_edges, correct_labels,
                                                test=True)
        # las_test = n_las / n_tokens
        print(f"n_tokens: {n_tokens}, n_las: {n_las}")
        label_accuracy.update_state(dep_labels, label_preds)
        
    las_test = n_las / n_tokens
    print(f"las test {las_test}")
    return head_accuracy.result(), label_accuracy.result(), las_test

# TODO make this plot function accept **kwargs.
def plot(epochs, uas_train, label_acc_train, uas_test, label_acc_test,
         las_train, las_test, model_name):
  fig = plt.figure()
  ax = plt.axes()
  ax.plot(epochs, uas_train, "-g", label="uas_train", color="blue")
  ax.plot(epochs, label_acc_train, "-g", label="labels_train", color="green")
  ax.plot(epochs, uas_test, "-g", label="uas_test", color="orchid")
  ax.plot(epochs, label_acc_test, "-g", label="labels_test", color="sienna")
  ax.plot(epochs, las_train, "-g", label="las_train", color="yellow")
  ax.plot(epochs, las_test, "-g", label="las_test", color="brown")
  
  plt.title("Performance on train and test data")
  plt.xlabel("epochs")
  plt.ylabel("accuracy")
  plt.legend()
  plt.savefig(f"{model_name}_plot")


def main(args):
  if args.train:
    embeddings = nn_utils.load_embeddings()
    word_embeddings = embeddor.Embeddings(name= "word2vec", matrix=embeddings)
    prep = preprocessor.Preprocessor(
      word_embeddings=word_embeddings, features=args.features,
      labels=args.labels)
    label_feature = next((f for f in prep.sequence_features if f.name == "dep_labels"),
                          None)
    # print("label feature ", label_feature)
    # print("n output classes", label_feature.n_values)
    # input("press to cont.")
    
    if args.dataset:
      logging.info(f"Reading from tfrecords {args.dataset}")
      dataset = prep.read_dataset_from_tfrecords(
                                 batch_size=args.batchsize,
                                 records="./input/treebank_train_0_50.tfrecords")
    else:
      logging.info(f"Generating dataset from {args.treebank}")
      dataset = prep.make_dataset_from_generator(
        path=os.path.join(_DATA_DIR, args.treebank),
        batch_size=args.batchsize)
    if args.test:
      if not args.train:
        sys.exit("Testing with a pretrained model is not supported yet.")
      test_dataset = prep.make_dataset_from_generator(
        path=os.path.join(_TEST_DATA_DIR, args.test_treebank),
        batch_size=1)
  parser = NeuralMSTParser(word_embeddings=prep.word_embeddings,
                           n_output_classes=label_feature.n_values,
                           predict=args.predict)
  print(parser)
  parser.plot()
  scores = parser.train(dataset, args.epochs, test_data=test_dataset)
  plot(np.arange(args.epochs), scores["uas_train"], scores["ls_train"],
                 scores["uas_test"], scores["ls_test"], scores["las_train"],
                 scores["las_test"], args.model_name)

    
if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--train",
                      type=bool,
                      default=True,
                      help="Trains a new model.")
  parser.add_argument("--test",
                      type=bool,
                      default=True,
                      help="Whether to test the trained model on test data.")
  parser.add_argument("--epochs",
                      type=int,
                      default=10,
                      help="Trains a new model.")
  parser.add_argument("--treebank",
                      type=str,
                      default="treebank_train_0_10.pbtxt")
  parser.add_argument("--test_treebank",
                      type=str,
                      default="treebank_0_3_gold.pbtxt")
  parser.add_argument("--dataset",
                      help="path to a prepared tf.data.Dataset")
  parser.add_argument("--features",
                      nargs="+",
                      type=str,
                      default=["words", "pos", "morph", "dep_labels", "heads"],
                      help="features to use to train the model.")
  # TODO: make sure --labels and --predict follow consistent naming.
  parser.add_argument("--labels",
                      nargs="+",
                      type=str,
                      default=["heads", "dep_labels"],
                      help="labels to predict.")
  parser.add_argument("--predict",
                      nargs="+",
                      type=str,
                      default=["edges", "labels"],
            					choices=["edges", "labels"],
                      help="which features to predict")
  parser.add_argument("--batchsize",
                      type=int, 
                      default=2,
                      help="Size of training and test data batches")
  parser.add_argument("--model_name",
                      type=str,
                      default="test",
                      help="Name of the model to save.")

  args = parser.parse_args()
  main(args)