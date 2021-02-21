import tensorflow as tf
import numpy as np
import argparse
import os
import time
from tensorflow.keras import layers, metrics, losses, optimizers
from typing import List, Dict, Tuple
from input import preprocessor
from tagset.reader import LabelReader
from util.nn import nn_utils
import collections
import matplotlib as mpl
import matplotlib.pyplot as plt

import logging


logging.basicConfig(format='%(levelname)s : %(message)s', level=logging.INFO)
np.set_printoptions(threshold=np.inf)
mpl.style.use("seaborn")

_DATA_DIR = "data/UDv23/Turkish/training"
_TEST_DATA_DIR = "data/UDv23/Turkish/test"

Dataset = tf.data.Dataset
Embeddings = preprocessor.Embeddings
SequenceFeature = preprocessor.SequenceFeature

class NeuralMSTParser:
  """The neural mst parser."""
  
  def __init__(self, *, word_embeddings: Embeddings, n_output_classes: int):
    self.word_embeddings = self._set_pretrained_embeddings(word_embeddings)
    self.edge_loss_fn = losses.SparseCategoricalCrossentropy(
      from_logits=True, reduction=tf.keras.losses.Reduction.NONE)
    self.label_loss_fn = losses.CategoricalCrossentropy(
      from_logits=True, reduction=tf.keras.losses.Reduction.NONE)
    # self.label_loss_fn = losses.CategoricalCrossentropy(
    #  from_logits=True)
    self.optimizer=tf.keras.optimizers.Adam(0.001, beta_1=0.9, beta_2=0.9)
    self.edge_train_metrics = metrics.SparseCategoricalAccuracy()
    self.label_train_metrics = metrics.CategoricalAccuracy()
    self.model = self._create_uncompiled_model(n_output_classes)
    self._n_output_classes = n_output_classes
    
  def __str__(self):
    return str(self.model.summary())
  
  def plot(self, name="neural_joint_mstparser.png"):
     tf.keras.utils.plot_model(self.model, name, show_shapes=True)

  def _set_pretrained_embeddings(self, 
                                 embeddings: Embeddings) -> layers.Embedding:
    """Builds a pretrained keras embedding layer from an Embeddings object."""
    embed = tf.keras.layers.Embedding(embeddings.vocab_size,
                             embeddings.embedding_dim,
                             trainable=False,
                             name="word_embeddings")
    embed.build((None,))
    embed.set_weights([embeddings.index_to_vector])
    return embed
  
  def _create_uncompiled_model(self, n_classes) -> tf.keras.Model:
    """Creates an NN model for edge factored dependency parsing."""
    
    word_inputs = tf.keras.Input(shape=(None,), name="words")
    word_features = self.word_embeddings(word_inputs)
    pos_inputs = tf.keras.Input(shape=([None]), name="pos")
    pos_features = tf.keras.layers.Embedding(input_dim=35,
                                             output_dim=32,
                                             name="pos_embeddings")(pos_inputs)
    concat = tf.keras.layers.concatenate([word_features, pos_features],
                                         name="concat")
    # encode the sentence with LSTM
    sentence_repr = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(
        units=256, return_sequences=True, name="encoded1"))(concat)
    sentence_repr = tf.keras.layers.Dropout(rate=0.3)(sentence_repr)
    sentence_repr = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(
        units=256, return_sequences=True, name="encoded2"))(sentence_repr)
    sentence_repr = tf.keras.layers.Dropout(rate=0.3)(sentence_repr)
    sentence_repr = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(
        units=256, return_sequences=True, name="encoded3"))(sentence_repr)
    # Get the dependency label predictions
    dep_labels = tf.keras.layers.Dense(units=n_classes,
                                       name="dep_labels")(sentence_repr)
    
    # the edge scoring bit with the MLPs
    # Pass the sentence representation through two MLPs, for head and dep.
    head_mlp = tf.keras.layers.Dense(256, activation="relu", name="head_mlp")
    dep_mlp = tf.keras.layers.Dense(256, activation="relu", name="dep_mlp")
    h_arc_head = head_mlp(dep_labels)
    h_arc_dep = dep_mlp(dep_labels)
    
    # Biaffine part of the model. Computes the edge scores for all edges
    W_arc = tf.keras.layers.Dense(256, use_bias=False, name="W_arc")
    b_arc = tf.keras.layers.Dense(1, use_bias=False, name="b_arc")
    Hh_W = W_arc(h_arc_head)
    Hh_WT = tf.transpose(Hh_W, perm=[0,2,1])
    Hh_W_Ha = tf.linalg.matmul(h_arc_dep, Hh_WT)
    Hh_b = b_arc(h_arc_head)
    edge_scores = Hh_W_Ha + tf.transpose(Hh_b, perm=[0,2,1])
    
    model = tf.keras.Model(inputs={"words": word_inputs, "pos": pos_inputs},
                                    outputs=[edge_scores, dep_labels])
    return model
  
  
  def test(self, *, dataset: Dataset, heads: bool=True, labels: bool=True):
    """Tests the performance on a test dataset."""
    head_accuracy = tf.keras.metrics.Accuracy()
    label_accuracy = tf.keras.metrics.Accuracy()
    # counter = collections.Counter()
    for example in dataset:
      words = example["words"]
      pos = example["pos"]
      edge_scores, label_scores = self.model({"words": words, "pos": pos},
                                             training=False)
      heads = example["heads"]
      dep_labels = example["dep_labels"]
      head_preds = tf.argmax(edge_scores, 2)
      label_preds = tf.argmax(label_scores, 2)                               
      # counter["total_tokens"] += heads.shape[1]
      # counter["head_correct"] += np.sum(heads == head_preds)
      # counter["label_correct"] += np.sum(dep_labels == label_preds)
      head_accuracy.update_state(heads, head_preds)
      label_accuracy.update_state(dep_labels, label_preds)

    return head_accuracy.result(), label_accuracy.result()
    

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
  def train_step(self, *, words: tf.Tensor, pos: tf.Tensor,
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
      label_loss: the depdendeny label loss associated with each token.
      correct_labels: tf.Tensor of shape (batch_size*seq_len, 1). The correct
          dependency labels.
      label_preds: tf.Tensor of shape (batch_size*seq_len, 1). The predicted
          dependency labels
    ..."""
    with tf.GradientTape() as tape:
      edge_scores, label_scores = self.model({"words": words, "pos": pos},
                                             training=True)
      pad_mask = (words != 0)
      edge_loss_pad, edge_loss_w_o_pad, edge_pred, h, pad_mask = self.edge_loss(
        edge_scores, heads, pad_mask)
      
      label_loss, correct_labels, label_preds = self.label_loss(
          dep_labels, label_scores, pad_mask)

    grads = tape.gradient([edge_loss_pad, label_loss],
                          self.model.trainable_weights)
    self.optimizer.apply_gradients(zip(grads, self.model.trainable_weights))
    
    # Update train metrics
    self.edge_train_metrics.update_state(heads, edge_scores)
    self.label_train_metrics.update_state(dep_labels, label_scores)
    
    # TODO: don't return the edge scores. 
    return (edge_loss_pad, edge_loss_w_o_pad, edge_pred, h, pad_mask,
            label_loss, correct_labels, label_preds, edge_scores)

  def train_custom(self, dataset: Dataset, epochs: int=10,
                   test_data: Dataset=None):
    """Custom training method.
    
    Args:
      dataset: Dataset containing features and labels.
      epochs: number of training epochs
      test_data: Dataset containing features and labels for test set.
    Returns:
      history: logs of scores and losses at the end of training.
    """
    history = collections.defaultdict(list)
    for epoch in range(epochs):
      # Reset the states before starting the new epoch.
      self.edge_train_metrics.reset_states()
      self.label_train_metrics.reset_states()
      logging.info(f"*********** Training epoch: {epoch} ***********\n\n")
      start_time = time.time()
      stats = collections.Counter()
      
      # Start the training loop for this epoch.
      for step, batch in enumerate(dataset):
        
        # Read the data and labels from the dataset.
        words = batch["words"]
        # print(words)
        pos = batch["pos"]
        # print(pos)
        dep_labels = tf.one_hot(batch["dep_labels"], self._n_output_classes)
        heads = batch["heads"]
        # print(heads)
        
        # Get the losses and predictions
        (edge_loss_pad, edge_loss_w_o_pad, edge_pred, h, pad_mask, label_loss,
        correct_labels, label_preds, edge_scores) = self.train_step(words=words, pos=pos,
                                                       dep_labels=dep_labels,
                                                       heads=heads)
        print("edge scores ", edge_scores)
        print("edge pred ", edge_pred)
        print("heads ", h)
        input("press to cont..")
        # Get the total number of tokens without padding
        words_reshaped = tf.reshape(words, shape=(pad_mask.shape))
        total_words = len(tf.boolean_mask(words_reshaped, pad_mask))
        
        # Calculate correct label predictions with padding
        label_correct_with_pad = (label_preds == correct_labels)
        n_label_correct_with_pad = np.sum(label_correct_with_pad)
        
        # Calculate correct label predictions without padding
        label_correct = tf.boolean_mask(label_correct_with_pad, pad_mask)
        n_label_correct= np.sum(label_correct)
        
        # Calculate correct edge predictions with padding
        edge_correct_with_pad = (edge_pred == h)
        n_edge_correct_with_pad = np.sum(edge_correct_with_pad)

        # Calculate correct edge predictions without padding
        edge_correct = tf.boolean_mask(edge_pred == h, pad_mask)
        n_edge_correct = np.sum(edge_correct)
        
        # Add to stats
        stats["n_edge_correct_with_pad"] += n_edge_correct_with_pad
        stats["n_edge_correct"] += n_edge_correct
        stats["n_label_correct_with_pad"] += n_label_correct_with_pad
        stats["n_label_correct"] += n_label_correct
        stats["n_tokens_with_pad"] += len(h)
        stats["n_tokens"] += total_words
        
      # Evaluate on test data at the end of epoch.
      if test_data:
        uas_test, label_acc_test = self.test(dataset=test_data)

      # Get training accuracy metrics 
      print(stats)
      edge_train_acc = self.edge_train_metrics.result()
      label_train_acc = self.label_train_metrics.result()
      
      # Compute average edge losses with and without padding
      avg_edge_loss_pad = tf.reduce_mean(edge_loss_pad)
      avg_edge_loss_w_o_pad = tf.reduce_mean(edge_loss_w_o_pad)
      
      # Compute average label loss (only with pad)
      avg_label_loss = tf.reduce_mean(label_loss)
      
      # Compute UAS with and without padding for this epoch
      uas_with_pad = stats["n_edge_correct_with_pad"] / stats["n_tokens_with_pad"]
      uas_without_pad = stats["n_edge_correct"] / stats["n_tokens"]
      
      # Compute Label Score (LS) with and without padding for this epoch
      label_score_pad = stats["n_label_correct_with_pad"] / stats["n_tokens_with_pad"]
      label_score_without_pad = stats["n_label_correct"] / stats["n_tokens"]
      
      # TODO: implement method that computes labeled attachement score
      # from UAS and LS.
      
      # Log all the stats
      # logging.info(f"Training accuracy (heads): {edge_train_acc}")
      # logging.info(f"Training accuracy (labels) : {label_train_acc}\n")
      
      # logging.info(f"UAS (with pad): {uas_with_pad}")
      logging.info(f"UAS (without pad): {uas_without_pad}")
      
      # logging.info(f"LS (with pad) {label_score_pad}")
      logging.info(f"LS (without pad) {label_score_without_pad}\n")
      
      logging.info(f"UAS on test data: {uas_test}")
      logging.info(f"LS on test data: {label_acc_test}\n")
      
      logging.info(f"Edge loss (with pad): {avg_edge_loss_pad}")
      # logging.info(f"Edge loss without pad: {avg_edge_loss_w_o_pad}")
      logging.info(f"Label loss (with pad) {avg_label_loss}\n")
      
      logging.info(f"Time for epoch: {time.time() - start_time}\n")
      
      # Populate stats history
      history["uas_train"].append(uas_without_pad) # Unlabeled Attachment Score
      history["ls_train"].append(label_score_without_pad) # Label Score
      history["uas_test"].append(uas_test)
      history["ls_test"].append(label_acc_test)
      history["edge_loss_pad"].append(avg_edge_loss_pad)
      # history["edge_loss_without_pad"].append(avg_edge_loss_w_o_pad)
      history["label_loss_pad"].append(avg_label_loss)
      # history["label_accuracy"].append(label_train_acc)
      
    return history

# TODO: the set up of the features should be done by the preprocessor class.
def _set_up_features(features: List[str],
                     label: List[str]) -> List[SequenceFeature]:
  sequence_features = []
  for feat in features:
    if not feat in label:
      sequence_features.append(preprocessor.SequenceFeature(name=feat))
    # We don't need to get the label_indices for "heads" as we don't do
    # label prediction on them in a typical sense. 
    else:
      if feat == "heads":
        sequence_features.append(preprocessor.SequenceFeature(
          name=feat, is_label=True))
      else:
        label_dict = LabelReader.get_labels(feat).labels
        label_indices = list(label_dict.values())
        label_feature = preprocessor.SequenceFeature(
          name=feat, values=label_indices, n_values=len(label_indices),
          is_label=True)
        sequence_features.append(label_feature)
  return sequence_features


def plot(epochs, uas_train, label_acc_train, uas_test, label_acc_test,
         model_name):
  fig = plt.figure()
  ax = plt.axes()
  ax.plot(epochs, uas_train, "-g", label="uas_train", color="blue")
  ax.plot(epochs, label_acc_train, "-g", label="labels_train", color="green")
  ax.plot(epochs, uas_test, "-g", label="uas_test", color="orchid")
  ax.plot(epochs, label_acc_test, "-g", label="labels_test", color="sienna")
  
  plt.title("Performance on training data")
  plt.xlabel("epochs")
  plt.ylabel("accuracy")
  plt.legend()
  plt.savefig(f"{model_name}_plot")



def main(args):
  sequence_features = _set_up_features(args.features, args.label)
  label_feature = next((f for f in sequence_features if f.name == "dep_labels"),
                        None)
  
  
  if args.train:
    embeddings = nn_utils.load_embeddings()
    word_embeddings = Embeddings(name= "word2vec", matrix=embeddings)
    prep = preprocessor.Preprocessor(word_embeddings=word_embeddings)
    
    if args.dataset:
      dataset = prep.read_dataset_from_tfrecords(
                                 batch_size=args.batchsize,
                                 features=sequence_features,
                                 records="./input/test501.tfrecords")
    else:
      dataset = prep.make_dataset_from_generator(
        path=os.path.join(_DATA_DIR, args.treebank),
        batch_size=args.batchsize, 
        features=sequence_features
      )
    if args.test:
      if not args.train:
        sys.exit("Testing with a pretrained model is not supported yet.")
      test_dataset = prep.make_dataset_from_generator(
        path=os.path.join(_TEST_DATA_DIR, args.test_treebank),
        batch_size=1,
        features=sequence_features
      )
  
  parser = NeuralMSTParser(word_embeddings=prep.word_embeddings,
                           n_output_classes=label_feature.n_values)
  print(parser)
  parser.plot()
  scores = parser.train_custom(dataset, args.epochs, test_data=test_dataset)
  plot(np.arange(args.epochs), scores["uas_train"], scores["ls_train"],
                 scores["uas_test"], scores["ls_test"], args.model_name)

    
if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--train", type=bool, default=True,
                      help="Trains a new model.")
  parser.add_argument("--test", type=bool, default=True,
                      help="Whether to test the trained model on test data.")
  parser.add_argument("--epochs", type=int, default=50,
                      help="Trains a new model.")
  parser.add_argument("--treebank", type=str,
                      default="treebank_0_3.pbtxt")
  parser.add_argument("--test_treebank", type=str,
                      default="treebank_0_3_gold.pbtxt")
  parser.add_argument("--dataset",
                      help="path to a prepared tf.data.Dataset")
  parser.add_argument("--features", type=list,
                      default=["words", "pos", "dep_labels", "heads"],
                      help="features to use to train the model.")
  parser.add_argument("--label", type=list, default=["heads", "dep_labels"],
                      help="labels to predict.")
  parser.add_argument("--batchsize", type=int, default=250,
                      help="Size of training and test data batches")
  parser.add_argument("--model_name", type=str, default="test_model",
                      help="Name of the model to save.")
  args = parser.parse_args()
  main(args)