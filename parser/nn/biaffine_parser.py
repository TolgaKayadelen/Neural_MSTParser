"""

NEXT: 
https://github.com/EelcovdW/Biaffine-Parser/blob/e372f5731bf00ff8cd013fdcd1e243d223c19117/train.py#L157

See the notebook playground to understand how to apply biaffine scoring that is impleemted
in here.
"""

import collections
import logging
import os
import sys
import time

from input import embeddor
import tensorflow as tf
import matplotlib as mpl
import numpy as np

from parser.nn import architectures
from proto import metrics_pb2
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
Metrics = metrics_pb2.Metrics

# Path for saving or loading pretrained models.
_MODEL_DIR = "./model/nn/pretrained"

class BiaffineMSTParser:
  """An MST Parser that parses the dependency labels before arcs."""
  
  def __init__(self, *, word_embeddings: Embeddings,
              n_output_classes: int,
              predict: List[str] = ["edges"],
              model_name: str = "biaffine_mst_parser"):
    self.word_embeddings = word_embeddings
    self.edge_loss_fn = losses.SparseCategoricalCrossentropy(
      from_logits=True, reduction=tf.keras.losses.Reduction.NONE)
    self.label_loss_fn = losses.SparseCategoricalCrossentropy(from_logits=True)
    self.optimizer=tf.keras.optimizers.Adam(0.001, beta_1=0.9, beta_2=0.9)
    self.edge_train_metrics = metrics.SparseCategoricalAccuracy()
    self.label_train_metrics = metrics.CategoricalAccuracy()
    self._n_output_classes = n_output_classes
    self._predict = predict
    self.model = self._parsing_model(model_name=model_name)
    self.metrics = self._metrics()
    assert(self._predict == self.model.predict), "Inconsistent configuration!"

  def __str__(self):
    return str(self.model.summary())
  
  def plot(self, name="neural_joint_mstparser.png"):
     tf.keras.utils.plot_model(self.model, name, show_shapes=True)
  
  def _metrics(self) -> Metrics:
    metric_list = ("uas", "uas_test", "edge_loss_padded")
    if "labels" in self._predict:
      metric_list += ("ls", "ls_test", "las", "las_test", "label_loss_padded")
    metrics = nn_utils.set_up_metrics(*metric_list)
    return metrics
  
  def _parsing_model(self, *, model_name: str) -> tf.keras.Model:
    """Creates an NN model for edge factored dependency parsing."""
    
    word_inputs = tf.keras.Input(shape=(None,), name="words")
    pos_inputs = tf.keras.Input(shape=(None,), name="pos")
    morph_inputs = tf.keras.Input(shape=(None, 66), name="morph")
    inputs = {"words": word_inputs, "pos": pos_inputs, "morph": morph_inputs}
    
    model = architectures.BiaffineParsingModel(
                              n_dep_labels=self._n_output_classes,
                              word_embeddings=self.word_embeddings,
                              predict=self._predict,
                              name=model_name)
    model(inputs=inputs)
    return model
  

  def _compute_metrics(self, *, stats: Dict, padded=False):
    """Computes UAS, LS, and LAS metrics.
    
    If padded, returns padded metrics as well as the unpadded ones.
    """
    # Compute UAS with and without padding for this epoch.
    uas = stats["n_edge_correct"] / stats["n_tokens"]
    return_dict = {"uas": uas}
    
    if padded:
      uas_padded = stats["n_edge_correct_padded"] / stats["n_tokens_padded"]
      return_dict["uas_padded"] = uas_padded
    
    # Compute Label Score (LS) with and without padding for this epoch
    if "labels" in self._predict:
      
      label_score = stats["n_label_correct"] / stats["n_tokens"]
      return_dict["ls"] = label_score
      
      if padded:
        label_score_padded = stats["n_label_correct_padded"] / stats["n_tokens_padded"]
        return_dict["ls_padded"] = label_score_padded
      
      # Compute Labeled Attachment Score
      if "n_las" in stats:
        las = stats["n_las"] / stats["n_tokens"]
        return_dict["las"] = las

    return return_dict
  
  def _update_metrics(self, train_metrics, test_metrics, loss_metrics):
    all_metrics = {**train_metrics, **test_metrics, **loss_metrics}
    
    for key, value in all_metrics.items():
      if not self.metrics.metric[key].tracked:
        logging.info(f"Metric {key} is not tracked!")
      else:
        self.metrics.metric[key].value_list.value.append(value)

  def _compute_n_correct(self, *, preds: Dict):
    """Computes n correct edge and label predictions from system predictons."""
    
    # Calculate correct edge predictions
    edge_pred = preds["edge_pred"]
    # edge_correct_with_pad = preds["edge_correct_with_pad"]
    correct_heads = preds["h"]
    
    # n_correct edge predictions with padding
    edge_correct_padded = (edge_pred == correct_heads)
    n_edge_correct_padded = np.sum(edge_correct_padded)

    # n_correct edge predictions without padding
    edge_correct = tf.boolean_mask(edge_pred == correct_heads, preds["pad_mask"])
    n_edge_correct = np.sum(edge_correct)
    
    return_dict = {"edge_correct": edge_correct,
                   "n_edge_correct_padded":  n_edge_correct_padded, 
                   "n_edge_correct": n_edge_correct,
                   "h": preds["h"]}
    
    # Calculate correct label predictions with padding
    if "labels" in self._predict:
      label_preds = preds["label_preds"]
      correct_labels = preds["correct_labels"]
      label_correct_padded = (label_preds == correct_labels)
      n_label_correct_padded = np.sum(label_correct_padded)
    
      # Calculate correct label predictions without padding
      label_correct = tf.boolean_mask(label_correct_padded, preds["pad_mask"])
      n_label_correct = np.sum(label_correct)
      
      # Add to return dict
      return_dict["label_correct"] = label_correct
      return_dict["n_label_correct"] = n_label_correct
      return_dict["n_label_correct_padded"] = n_label_correct_padded

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
  
  def _arc_maps(self, heads:tf.Tensor, labels:tf.Tensor):
    """"Returns a list of tuples mapping heads, dependents and labels. 
    
    For each sentence in a training batch, this function creates a list of
    tuples with the values [batch_idx, head_idx, dep_idx, label_idx].
    
    Next: this will be used in scoring label loss.
    """
    arc_maps = []
    for sentence_idx, batch in enumerate(heads):
      for token_idx, head in enumerate(batch):
        arc_map = []
        arc_map.append(sentence_idx) # batch index
        arc_map.append(head.numpy()) # head index
        arc_map.append(token_idx) # dependent index
        arc_map.append(labels[sentence_idx, token_idx].numpy()) # label index
        arc_maps.append(arc_map)
    return arc_maps

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
    # print("edge_scores shape is, ", edge_scores.shape)
    # input("press to cont.")
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
  def label_loss(self, dep_labels, label_scores, arc_maps, pad_mask):
    """Computes label loss and label predictions
    Args:
      dep_labels: tf.Tensor of shape (batch_size, seq_len, n_labels). Holding
        correct labels for each token as a one hot vector.
      label_scores: tf.Tensor of shape (batch_size, seq_len, seq_len, n_labels).
        Holding probability scores for for each label for all possible head<->dep
        relations.
      arc_maps: A List of lists, each list holds info as 
        [batch_idx, head_idx, dep_idx, label_idx] for each sentence/token.
      pad_mask: tf.Tensor of shape (batch_size*seq_len, 1)
    Returns:
      label_loss: the label loss associated with each token. This is always
        computed with padding.
      correct_labels: tf.Tensor of shape (batch_size*seq_len, 1). The correct
        dependency labels.
      label_preds: tf.Tensor of shape (batch_size*seq_len, 1). The predicted
        dependency labels.
    """
    # Transpose the label scores to [batch_size, seq_len, seq_len, n_classes]
    label_scores = tf.transpose(label_scores, perm=[0,2,3,1])
    print("label scores shape now ", label_scores.shape)
    arc_maps = np.array(arc_maps)
    print(arc_maps, arc_maps.shape)
   
    input("press to cont.")
    logits = tf.gather_nd(label_scores, indices=arc_maps[:, :3])
    labels = arc_maps[:, 3]
    print("logits ", logits)
    print("labels ", labels)
    input("press to cont.")
    
    label_loss = self.label_loss_fn(labels, logits)
    print("label loss: ", label_loss)
    input("press to cont.")
    
    # TODO: continue debugging from here. Check whteher the label loss function
    # should return mean reduced label loss or not (compare with label first
    # parser.)
    label_preds = tf.reshape(tf.argmax(label_scores,
                                       axis=2),
                             shape=(pad_mask.shape)
                             )
    correct_labels = tf.reshape(tf.argmax(dep_labels,
                                          axis=2),
                                shape=(pad_mask.shape))
  
    return label_loss, correct_labels, label_preds

  @tf.function
  def train_step(self, *, words: tf.Tensor, pos: tf.Tensor, morph: tf.Tensor,
                 dep_labels: tf.Tensor, heads:tf.Tensor, arc_maps: List
                 ) -> Tuple[tf.Tensor, ...]:
    """Runs one training step.
    
    Args:
      words: A tf.Tensor of word indexes of shape (batch_size, seq_len) where
          the seq_len is padded with 0s on the right.
      pos: A tf.Tensor of pos indexes of shape (batch_size, seq_len), of the
          same shape as words.
      dep_labels: A tf.Tensor of one hot vectors representing dep_labels for
          each token, of shape (batch_size, seq_len, n_classes).
      morph: A tf.Tensor of (batch_size, seq_len, n_morph)
      heads: Correct heads, a tf.Tensor of shape (batch_size, seq_len).
      arc_maps: A List of lists: [batch_idx, head_idx, dep_idx, label_idx]
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
        label_loss, correct_labels, label_preds = self.label_loss(dep_labels,
                                                                  label_scores,
                                                                  arc_maps,
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
    uas_test, ls_test, las_test = [0] * 3
    avg_label_loss_padded, avg_edge_loss_padded = [0] * 2
    
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
        
        
        arc_maps = self._arc_maps(heads, batch["dep_labels"])
        # print(arc_maps)
     
        # We cast the type of this tensor to float32 because in the model
        # pos and word features are passed through an embedding layer, which
        # converts them into float values implicitly.
        # TODO: maybe do this as you read in the morph values in preprocessor.
        morph = tf.dtypes.cast(batch["morph"], tf.float32)
        
        # Run forward propagation to get losses and predictions for this step.
        losses_and_preds = self.train_step(words=words, pos=pos, morph=morph,
                                           dep_labels=dep_labels, heads=heads,
                                           arc_maps=arc_maps)
    
        # Get the total number of tokens without padding for this step.
        words_reshaped = tf.reshape(words, 
                                    shape=(losses_and_preds["pad_mask"].shape))
        total_words = len(tf.boolean_mask(words_reshaped,
                                          losses_and_preds["pad_mask"]))
        
        # Get the number of correct predictions for this step.
        n_correct = self._compute_n_correct(preds=losses_and_preds)
        
        # Accumulate the stats for all steps in the epoch.
        stats["n_edge_correct"] += n_correct["n_edge_correct"]
        stats["n_edge_correct_padded"] += n_correct["n_edge_correct_padded"]
        stats["n_tokens"] += total_words
        stats["n_tokens_padded"] += len(n_correct["h"])
       
        
        # Calculate the number of tokens which has correct las for this step
        # and add it to the accumulator.
        if "labels" in self._predict:
          stats["n_label_correct"] += n_correct["n_label_correct"]
          stats["n_label_correct_padded"] += n_correct["n_label_correct_padded"]
          las_correct = self._labeled_attachment_score(
            n_correct["edge_correct"], n_correct["label_correct"])
          stats["n_las"] += las_correct     

      print(f"Training Stats: {stats}")
      
      # Get UAS, LS, and LAS after the epoch.
      training_metrics = self._compute_metrics(stats=stats)
      
      # Get accuracy metrics.
      edge_train_acc = self.edge_train_metrics.result()
      label_train_acc = self.label_train_metrics.result()
      
      # Compute average edge losses with and without padding
      avg_edge_loss_padded = tf.reduce_mean(
        losses_and_preds["edge_loss_pad"]).numpy()
      avg_edge_loss_unpadded = tf.reduce_mean(
        losses_and_preds["edge_loss_w_o_pad"]).numpy()
      
      # Compute average label loss (only with pad)
      if "labels" in self._predict:
        avg_label_loss_padded = tf.reduce_mean(
          losses_and_preds["label_loss"]).numpy()
      
      logging.info(f"""
        UAS train: {training_metrics['uas']}
        Edge loss (padded): {avg_edge_loss_padded}\n""")
      
      if "labels" in self._predict:
        logging.info(f"""
          LS train: {training_metrics['ls']}
          LAS train: {training_metrics['las']}
          Label loss (padded) {avg_label_loss_padded}\n""")
      
      logging.info(f"Time for epoch: {time.time() - start_time}\n")
      
      # Update scores on test data at the end of every X epoch.
      if epoch % 5 == 0 and test_data:
        uas_test, ls_test, las_test = self.test(dataset=test_data)
        logging.info(f"UAS test: {uas_test}")
        logging.info(f"LS test: {ls_test}")
        logging.info(f"LAS test: {las_test}\n")
      

      # Update metrics
      self._update_metrics(
        train_metrics=training_metrics,
        test_metrics={"uas_test": uas_test,
                      "ls_test": ls_test,
                      "las_test": las_test},
        loss_metrics={"edge_loss_padded": avg_edge_loss_padded,
                      "label_loss_padded": avg_label_loss_padded})

    return self.metrics
    

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
        label_accuracy.update_state(dep_labels, label_preds)
        
    las_test = n_las / n_tokens
    return (head_accuracy.result().numpy(),
            label_accuracy.result().numpy(),
            las_test)
  
  def parse(self, example: Dict):
    """Parse an example with this parser."""
    
    words, pos = example["words"], example["pos"]
    morph = tf.dtypes.cast(example["morph"], tf.float32)
    n_tokens = words.shape[1]
      
    scores = self.model({"words": words, "pos": pos, "morph": morph},
                        training=False)
    return scores["edges"], scores["labels"]

  def save(self, *, suffix=0):
    """Saves the model to path."""
    model_name = self.model.name
    print(model_name)
    try:
      path = os.path.join(_MODEL_DIR, self.model.name)
      print("path ", path)
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
    