"""This is a seq2seq transformer label first parsing model based on pretrained BERT."""
import os
import collections
import datasets
import datetime
import time
import tensorflow as tf
import numpy as np
import transformers

from datasets import load_dataset, disable_caching
from tensorflow.keras import layers, metrics, losses, optimizers
from transformers import AutoConfig, AutoTokenizer, TFAutoModelForTokenClassification, DataCollatorForTokenClassification
from transformers import create_optimizer, set_seed
from transformers import DataCollatorWithPadding
from parser import base_parser
from parser.utils import layer_utils
from tagset.reader import LabelReader as label_reader
from util import reader


import logging
logging.basicConfig(format='%(levelname)s : %(message)s', level=logging.INFO)

from typing import List

# disable_caching()

tokenizer=AutoTokenizer.from_pretrained("bert-base-multilingual-cased", use_fast=True)
raw_dataset = load_dataset("./transformer/hf/dataset/boun_hf_data_loader.py")
features = raw_dataset["train"].features
label_list = features["dep_labels"].feature.names
label_to_id = {i: i for i in range(len(label_list))}
print("label to id ", label_to_id)


def tokenize_and_align_labels(examples):
  tokenized_inputs = tokenizer(examples["tokens"],
                               max_length=False,
                               padding=False,
                               truncation=True,
                               is_split_into_words=True)
  labels = []
  words = []
  for i, label in enumerate(examples["dep_labels"]):
    # print("i ", i, "label ", label)
    # print("words ", examples["tokens"][i])
    # input()
    word_ids = tokenized_inputs.word_ids(batch_index=i)
    previous_word_idx = None
    label_ids = []
    for word_idx in word_ids:
      # Special tokens have a word id that is None. We set the label to -100 so they are automatically
      # ignored in the loss function.
      if word_idx is None:
        label_ids.append(-100)
        # print("label ids ", label_ids)
        # input()
      # We set the label for the first token of each word.
      elif word_idx != previous_word_idx:
        # print("label[word_idx] ", label[word_idx])
        # input()
        label_ids.append(label_to_id[label[word_idx]])
      # For the other tokens in a word, we set the label to either the current label or -100, depending on
      # the label_all_tokens flag.
      else:
        # label_ids.append(label_to_id[label[word_idx]])
        label_ids.append(-100)
      previous_word_idx = word_idx
    word_ids = [i if i != None else -100 for i in word_ids]
    labels.append(label_ids)
    words.append(word_ids)
  tokenized_inputs["labels"] = labels
  tokenized_inputs["words"] = words
  # print("tokenized inputs ", tokenized_inputs)
  # input()
  return tokenized_inputs


class BertLabelFirstParser:
  """A label first parser predicts labels before predicting heads."""
  def __init__(self, *,
               name:str = "bert_label_first_parser",
               word_embeddings,
               num_labels: int,
               language="tr",
               features=["pos", "morph", "dep_labels"],
               bert_model_name="bert-base-multilingual-cased",
               pretrained_bert_model_path=None,
               log_dir=None,
               test_every: int = 5,
               label_embedding_weights = None,
               pos_embeddings_weights = None,
               train_batch_size: int = 50,
               load_pretrained=None):
    if pretrained_bert_model_path is not None:
      self.bert_model=transformers.TFAutoModelForTokenClassification.from_pretrained(pretrained_bert_model_path)
    else:
      self.bert_config = AutoConfig.from_pretrained(bert_model_name, num_labels=num_labels)
      self.bert_model = transformers.TFAutoModelForTokenClassification.from_pretrained(
        bert_model_name, config=self.bert_config)
    self.word_embeddings=word_embeddings
    self.load_pretrained=load_pretrained
    self._use_pos = "pos" in features
    self._use_morph = "morph" in features
    self._use_dep_labels = "dep_labels" in features
    self.metrics = collections.Counter()
    self.output_dir = "./transformer/hf/pretrained/bert-lfp-parser"
    self.train_batch_size=train_batch_size
    self.training_stats = collections.Counter()
    self.test_stats = collections.Counter()
    self.pos_reader = label_reader.get_labels("pos", "tr")
    # Setting up the model.
    self.model = self._parsing_model(model_name=name)
    if self.load_pretrained is not None:
      self.load(name=name, path=load_pretrained)
    if label_embedding_weights is not None:
      self.model.label_embeddings.set_weights(label_embedding_weights)
      self.model.label_embeddings.trainable = False
      logging.info(f"Installed pretrained label embeddings, trainable {self.model.label_embeddings.trainable}")
    if pos_embeddings_weights is not None:
      self.model.pos_embeddings.set_weights(pos_embeddings_weights)
      self.model.pos_embeddings.trainable = False
      logging.info(f"Installed pretrained pos embeddings, trainable {self.model.pos_embeddings.trainable}")
    # input()
    set_seed(42)

  # Region set up
  @property
  def inputs(self):
    input_dict = {}
    tokens = tf.keras.Input(shape=(None, ), name="tokens", dtype=tf.string)
    pos = tf.keras.Input(shape=(None, ), name="pos", dtype=tf.int64)
    heads = tf.keras.Input(shape=(None, ), name="heads", dtype=tf.int64)
    dep_labels = tf.keras.Input(shape=(None, ), name="dep_labels", dtype=tf.int64)
    input_ids = tf.keras.Input(shape=(None, ), name="input_ids", dtype=tf.int64)
    token_type_ids = tf.keras.Input(shape=(None, ), name="token_type_ids", dtype=tf.int64)
    attention_mask = tf.keras.Input(shape=(None, ), name="attention_mask", dtype=tf.int64)
    words = tf.keras.Input(shape=(None, ), name="words", dtype=tf.int64)
    labels = tf.keras.Input(shape=(None, ), name="labels", dtype=tf.int64)
    input_dict["tokens"] = tokens
    input_dict["pos"] = pos
    input_dict["heads"] =  heads
    input_dict["dep_labels"] = dep_labels
    input_dict["input_ids"] =  input_ids
    input_dict["token_type_ids"] =  token_type_ids
    input_dict["attention_mask"] =  attention_mask
    input_dict["words"] =  words
    input_dict["labels"] =  labels
    return input_dict

  def _parsing_model(self, model_name):
    print(f"""Using features pos: {self._use_pos}, morph: {self._use_morph},
              dep_labels: {self._use_dep_labels}""")
    model = BertLabelFirstParsingModel(
      name=model_name,
      word_embeddings=self.word_embeddings,
      use_pos=self._use_pos,
      use_morph=self._use_morph,
      bert_model=self.bert_model
    )
    if self.load_pretrained is not None:
      # We need this to initialize the variables of these layers.
      model.lstm_block(tf.random.uniform(shape=(2,14,382)))
      model.dep_perceptron(tf.random.uniform(shape=(2,14,512)))
      model.head_perceptron(tf.random.uniform(shape=(2,14,512)))
      model.edge_scorer(tf.random.uniform(shape=(2,14,256)), tf.random.uniform(shape=(2,14,256)))
    return model

  # Region data processing.
  def _flatten(self, _tensor):
    """The tensor should be a 2D tensor of shape batch_size, n_dim.
      Returns a tensor of shape (batch_size*n_dim, 1)"""
    reshaped = tf.reshape(_tensor, shape=(_tensor.shape[0]*_tensor.shape[1]))
    return reshaped

  def _process_dataset(self):
    processed_dataset = raw_dataset.map(
      tokenize_and_align_labels,
      batched=True,
      desc="Running tokenizer on dataset")
    return processed_dataset

  def make_dataset_from_generator(self, processed_dataset, batch_size):
    generator = lambda: self._example_generator(processed_dataset)
    output_shapes = {
      "sent_id" : [None],
      "tokens": [None],
      "token_ids": [None],
      "pos": [None],
      "heads": [None],
      "dep_labels": [None],
      "input_ids": [None],
      "token_type_ids": [None],
      "attention_mask": [None],
      "words": [None],
      "labels": [None]
    }
    padded_shapes = {
      "sent_id" : [None],
      "tokens": [None],
      "token_ids": [None],
      "pos": [None],
      "heads": [None],
      "dep_labels": [None],
      "input_ids": [None],
      "token_type_ids": [None],
      "attention_mask": [None],
      "words": [None],
      "labels": [None]
    }
    output_types = {
      "sent_id" : tf.string,
      "tokens": tf.string,
      "token_ids": tf.int64,
      "pos": tf.int64,
      "heads": tf.int64,
      "dep_labels": tf.int64,
      "input_ids": tf.int64,
      "token_type_ids": tf.int64,
      "attention_mask": tf.int64,
      "words": tf.int64,
      "labels": tf.int64
    }

    _padding_values = {
      "sent_id" : tf.constant("-pad-", dtype=tf.string),
      "tokens": tf.constant("-pad-", dtype=tf.string),
      "token_ids": tf.constant(0, dtype=tf.int64),
      "pos": tf.constant(0, dtype=tf.int64),
      "heads": tf.constant(-100, dtype=tf.int64),
      "dep_labels": tf.constant(0, dtype=tf.int64),
      "input_ids": tf.constant(0, dtype=tf.int64),
      "token_type_ids": tf.constant(0, dtype=tf.int64),
      "attention_mask": tf.constant(0, dtype=tf.int64),
      "words": tf.constant(-100, dtype=tf.int64),
      "labels": tf.constant(-100, dtype=tf.int64)
    }
    dataset = tf.data.Dataset.from_generator(generator,
                                             output_shapes=output_shapes,
                                             output_types=output_types)
    dataset = dataset.padded_batch(batch_size,
                                   padded_shapes=padded_shapes,
                                   padding_values=_padding_values)
    return dataset

  def _example_generator(self, processed_dataset):
    yield_dict = {}
    for example in processed_dataset:
      # making sure sent_id, original words and original labels match the size
      # of the tokenized ones.
      yield_dict["sent_id"] = [example["sent_id"]] * len(example["input_ids"])
      diff_len = len(example["input_ids"]) - len(example["tokens"])
      example["tokens"].extend(["-pad-"] * diff_len)
      example["token_ids"].extend([0] * diff_len)
      example["dep_labels"].extend([0] * diff_len)
      example["pos"].extend(["-pad-"] * diff_len)
      try:
        example["pos"] = [self.pos_reader.vtoi(pos) for pos in example["pos"]]
      except KeyError as e:
        print("Key error in ", e.args[0])
      example["heads"].extend([-100] * diff_len)
      yield_dict["tokens"] = example["tokens"]
      yield_dict["token_ids"] = example["token_ids"]
      yield_dict["dep_labels"] = example["dep_labels"]
      yield_dict["pos"] = example["pos"]
      yield_dict["heads"] = example["heads"]
      yield_dict["input_ids"] = example["input_ids"]
      yield_dict["token_type_ids"] = example["token_type_ids"]
      yield_dict["attention_mask"] = example["attention_mask"]
      yield_dict["words"] = example["words"]
      yield_dict["labels"] = example["labels"]

      yield yield_dict

  ######## Region train and test loop.
  def train(self, epochs):
    processed_dataset = self._process_dataset()
    dataset_length = len(processed_dataset["train"])
    print("Train Dataset Size: ", dataset_length)
    # for example in processed_dataset["train"]:
    #  print("example ", example)
    #  c = input()
    #  if c == "b":
    #    break
    tf_train_dataset = self.make_dataset_from_generator(processed_dataset["train"],
                                                        batch_size=self.train_batch_size)
    tf_val_dataset = self.make_dataset_from_generator(processed_dataset["validation"],
                                                      batch_size=10)
    # for example in tf_train_dataset:
    #  print(example)
    #  c = input()
    #  if c == "b":
    #    break
    label_loss = CustomNonPaddingTokenLoss(name="lfp_label_loss", ignore_index=-100)
    head_loss = CustomNonPaddingTokenLoss(name="lfp_head_loss", ignore_index=-100)
    num_train_steps = dataset_length * epochs
    optimizer, lr_schedule = create_optimizer(
      init_lr=5e-05,
      num_train_steps=num_train_steps,
      num_warmup_steps=0,
      adam_beta1=0.9,
      adam_beta2=0.999,
      adam_epsilon=1e-08,
      weight_decay_rate=0.0,
      adam_global_clipnorm=1.0)
    for epoch in range(epochs):
      for key in self.training_stats:
        self.training_stats[key] = 0.0
      logging.info(f"\n\n{'->' * 12} Training Epoch: {epoch} {'<-' * 12}\n\n")
      start_time = time.time()
      epoch_loss = 0.0
      for step, batch in enumerate(tf_train_dataset):
        # Take one forward step
        with tf.GradientTape() as tape:
          scores = self.model(batch)
          edge_scores, label_scores = scores["edges"], scores["labels"]
          head_preds, label_preds = scores["head_preds"], scores["label_preds"]
          # print("head preds ", head_preds)
          # print("label preds ", label_preds)
          correct_labels = batch["dep_labels"]
          # print("correct labels ", correct_labels)
          correct_heads = batch["heads"]
          # print("correct heads ", correct_heads)
          pad_mask = (correct_heads != -100)
          # print("pad mask ", pad_mask)
          # input()
          # This is the tokenized version of the labels used for loss calculation
          labels = batch["labels"]
          # print("tokenized labels ", labels)
          print("step ", step)
          label_loss_value = label_loss(labels, label_scores)
          # print("label loss value ", label_loss_value)
          head_loss_value = head_loss(correct_heads, edge_scores)
          print("head loss value ", head_loss_value)
          # input()
          epoch_loss += (label_loss_value + head_loss_value)
          # print("epoch loss ", epoch_loss)
          if step > 0:
            loss_avg = epoch_loss / step
            print(f"Running Loss ", loss_avg)
        # Back propagate errors at the end of each step
        joint_loss = label_loss_value + head_loss_value
        grads = tape.gradient(joint_loss,
                              self.model.trainable_weights)
        optimizer.apply_gradients(zip(grads, self.model.trainable_weights))

        # Update stats
        correct_predictions_dict = self.correct_predictions(
          head_predictions=head_preds,
          correct_heads=correct_heads,
          label_predictions=label_preds,
          correct_labels=correct_labels,
          pad_mask=pad_mask
        )
        n_words_in_batch = np.sum(pad_mask)
        # print("n words in batch ", n_words_in_batch)
        # input()
        self._update_stats(correct_predictions_dict, n_words_in_batch)
        if step % 5 == 0:
          logging.info(f"Training stats: {self.training_stats}")
          training_metrics = self._compute_metrics(training=True)
          print(f"Traning metrics {training_metrics}")

      # Log stats at the end of each epoch
      training_metrics = self._compute_metrics(training=True)
      print(f"Traning metrics {training_metrics}")
      logging.info(f"Time for epoch {time.time() - start_time}")
    self.test(test_set=tf_val_dataset)
    logging.info(f"End of training. Saving model")
    self.save()

  def test(self, test_set):
    logging.info("Testing on the validation set.")
    for key in self.test_stats:
      self.test_stats[key] = 0.0
    for example in test_set:
      scores = self.model(example, training=False)
      edge_scores, label_scores = scores["edges"], scores["labels"]
      head_preds, label_preds = scores["head_preds"], scores["label_preds"]
      # print("head preds ", head_preds, "label preds ", label_preds)
      correct_labels = example["dep_labels"]
      # print("correct labels ", correct_labels)
      correct_heads = example["heads"]
      # print("correct heads ", correct_heads)
      pad_mask = (correct_heads != -100)
      # print("pad mask ", pad_mask)
      correct_predictions_dict = self.correct_predictions(
        head_predictions=head_preds,
        correct_heads=correct_heads,
        label_predictions=label_preds,
        correct_labels=correct_labels,
        pad_mask=pad_mask
      )
      # print("correct predictions dict ", correct_predictions_dict)
      n_words = np.sum(pad_mask)
      # print("n words ", n_words)
      # input()
      self._update_stats(correct_predictions_dict,
                         n_words, stats="test")
    logging.info(f"Test stats: {self.test_stats}")
    test_results = self._compute_metrics(test=True)
    print(f"Test results: ", test_results)

  ######## Region Eval.
  def _update_stats(self, correct_predictions_dict, n_words_in_batch,
                    stats="training"):
    if stats == "training":
      stats = self.training_stats
    else:
      stats = self.test_stats

    stats["n_tokens"] += n_words_in_batch
    # Correct head predictions
    stats["n_chp"] += correct_predictions_dict["n_chp"]
    h = correct_predictions_dict["chp"]

    stats["n_clp"] += correct_predictions_dict["n_clp"]
    l = correct_predictions_dict["clp"]

    stats["n_chlp"] += np.sum(
      [1 for tok in zip(h, l) if tok[0] == True and tok[1] == True])

  def correct_predictions(self, *,
                          head_predictions=None,
                          correct_heads=None,
                          label_predictions=None,
                          correct_labels=None,
                          pad_mask=None):
    """Computes correctly predicted edges and labels and relevant stats for them."""
    correct_predictions_dict = {"chp": None, "n_chp": None, "clp": None, "n_clp": None}

    if pad_mask is None:
      pad_mask = np.full(shape=head_predictions.shape, fill_value=True, dtype=bool)

    # print("pad mask ", pad_mask)

    correct_head_preds =tf.boolean_mask(head_predictions == correct_heads, pad_mask)
    # print("corr haed preds ", correct_head_preds)
    n_correct_head_preds = tf.math.reduce_sum(tf.cast(correct_head_preds, tf.int32))
    # print("n corr head preds ", n_correct_head_preds)
    correct_predictions_dict["chp"] = correct_head_preds
    correct_predictions_dict["n_chp"] = n_correct_head_preds.numpy()
    correct_label_preds = tf.boolean_mask(label_predictions == correct_labels, pad_mask)
    # print("corr label preds ", correct_label_preds)
    n_correct_label_preds = tf.math.reduce_sum(tf.cast(correct_label_preds, tf.int32))
    # print("n corr label preds ", n_correct_label_preds)
    correct_predictions_dict["clp"] = correct_label_preds
    correct_predictions_dict["n_clp"] = n_correct_label_preds.numpy()
    # print("corr predictions dict ", correct_predictions_dict)
    # input()
    return correct_predictions_dict

  def _compute_metrics(self, training=False, test=False):
    """Computes metrics for uas, ls, and las."""
    if not training and not test:
      raise ValueError("Neither training nor test metrics are requested!")
    metrics = {}
    if test:
      _test_metrics = {}
      _test_metrics["uas"] = (self.test_stats["n_chp"] / self.test_stats["n_tokens"])
      _test_metrics["ls"] = (self.test_stats["n_clp"] / self.test_stats["n_tokens"])
      _test_metrics["las"] = (self.test_stats["n_chlp"] / self.test_stats["n_tokens"])
      metrics["test"] = _test_metrics

    if training:
      _train_metrics = {}
      _train_metrics["uas"] = (self.training_stats["n_chp"] / self.training_stats["n_tokens"])
      _train_metrics["ls"] = (self.training_stats["n_clp"] / self.training_stats["n_tokens"])
      _train_metrics["las"] = (self.training_stats["n_chlp"] / self.training_stats["n_tokens"])
      metrics["training"] = _train_metrics
    return metrics

  # Region load and save
  def load(self, *, name: str, path):
    """Loads a pretrained model weights."""
    path = os.path.join(path, name)
    load_status = self.model.load_weights(os.path.join(path, name))
    logging.info(f"Loaded model from model named: {name} in: {path}")
    load_status.assert_consumed()

  def save(self):
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    save_dir = os.path.join(self.output_dir, current_time)
    os.mkdir(save_dir)
    bert_dir = os.path.join(save_dir, "bert")
    os.mkdir(bert_dir)
    model_dir = os.path.join(save_dir, self.model.name)
    os.mkdir(model_dir)
    self.model.bert_model.save_pretrained(bert_dir)
    self.model.save_weights(os.path.join(model_dir, self.model.name), save_format="tf")
    logging.info(f"Saved model to  {save_dir}")

################# Parsing Model ################
class BertLabelFirstParsingModel(tf.keras.Model):
  """Bert based transformer model predicting labels before edges."""
  def __init__(self, *,
               name="Bert_Based_Label_First_Parsing_Model",
               word_embeddings,
               bert_model,
               use_pos:bool = True,
               use_morph:bool=False,
               ):
    super(BertLabelFirstParsingModel, self).__init__(name=name)

    self.bert_model = bert_model
    self.use_pos = use_pos
    self.use_morph = use_morph

    self.word_embeddings = layer_utils.EmbeddingLayer(
      pretrained=word_embeddings,
      name="word_embeddings",
      trainable=False)

    self.pos_embeddings = layer_utils.EmbeddingLayer(input_dim=37,
                                                     output_dim=32,
                                                     name="pos_embeddings",
                                                     trainable=True)

    self.label_embeddings = layer_utils.EmbeddingLayer(input_dim=43,
                                                       output_dim=50,
                                                       name="label_embeddings",
                                                       trainable=True)

    self.concatenate = layers.Concatenate(name="concat")

    self.lstm_block = LSTMBlock(n_units=256,
                                dropout_rate=0.3,
                                name="lstm_block")

    self.head_perceptron = layer_utils.Perceptron(n_units=256,
                                                  activation="relu",
                                                  name="head_mlp")
    self.dep_perceptron = layer_utils.Perceptron(n_units=256,
                                                 activation="relu",
                                                 name="dep_mlp")
    self.edge_scorer = layer_utils.EdgeScorer(n_units=256, name="edge_scorer")


  def call(self, inputs, training=True):
    """Forward pass."""
    # bert model inputs
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]
    token_type_ids=inputs["token_type_ids"]

    # these are bert tokenized labels with -100 for word piece tokens.
    labels = inputs["labels"]

    label_predictions = self.bert_model(input_ids=input_ids,
                                        attention_mask=attention_mask,
                                        token_type_ids=token_type_ids)
    label_scores = label_predictions.logits
    # print("label_scores ", label_scores)
    # input()
    label_preds = tf.argmax(label_scores, axis=2)
    # print("labels ", labels)
    # print("label preds ", label_preds)
    # input()
    true_preds = [[
      p.numpy() for (p, l) in zip(pred, label) if l != -100] for (pred, label) in zip(label_preds, labels)]
    # print("true preds ",  true_preds)
    # input()
    padded_len = inputs["dep_labels"].shape[1]
    # print("padded length ", padded_len)
    true_preds = tf.convert_to_tensor([l+(padded_len-len(l))*[0] for l in true_preds], dtype=tf.int64)
    # §print("true preds ", true_preds)
    # print("dep labels ", inputs["dep_labels"])
    # input()
    pos = inputs["pos"]
    token_ids = inputs["token_ids"]

    embedded_preds = self.label_embeddings(true_preds)
    embedded_pos = self.pos_embeddings(pos)
    embedded_tokens = self.word_embeddings(token_ids)
    # print("embedded preds ", embedded_preds)
    # print("embedded pos ", embedded_pos)
    concat = self.concatenate([embedded_tokens, embedded_pos, embedded_preds])
    sentence_repr = self.lstm_block(concat, training=training)

    h_arc_head = self.head_perceptron(sentence_repr)
    h_arc_dep = self.dep_perceptron(sentence_repr)
    edge_scores = self.edge_scorer(h_arc_head, h_arc_dep)
    head_preds = tf.argmax(edge_scores, axis=2)
    # print("head preds ", head_preds)
    # print("heads ", inputs["heads"])
    # input()
    return {"edges": edge_scores, "labels": label_scores,
            "label_preds": true_preds, "head_preds": head_preds}


class LSTMBlock(layers.Layer):
  """A bidirectional LSTM block with 3 Birectional LSTM layers"""
  def __init__(self, *, n_units: int,
               return_sequences: bool = True,
               return_state: bool = False,
               dropout_rate: float = 0.0,
               name="LSTMBlock"):
    super(LSTMBlock, self).__init__(name=name)
    print("Setting up LSTM block with dropout rate ", dropout_rate)
    self.dropout_rate = dropout_rate
    self.lstm1 = layers.Bidirectional(layers.LSTM(
      units=n_units, return_sequences=return_sequences,
      # stateful=True,
      name="lstm1"))
    self.lstm2 = layers.Bidirectional(layers.LSTM(
      units=n_units, return_sequences=return_sequences,
      # stateful=True,
      name="lstm2"))
    self.lstm3 = layers.Bidirectional(layers.LSTM(
      units=n_units, return_sequences=return_sequences,
      return_state=return_state,
      # stateful=True,
      name="lstm3"))
    self.dropout1 = layers.Dropout(rate=dropout_rate, name="dropout1")
    self.dropout2 = layers.Dropout(rate=dropout_rate, name="dropout2")
    # self.dropout3 = layers.Dropout(rate=dropout_rate, name="dropout3")

  def call(self, input_tensor, training=True):
    dropout = self.dropout_rate > 0 and training
    # print("dropout is ", dropout)
    if dropout:
      out = self.lstm1(input_tensor)
      out = self.dropout1(out)
      out = self.lstm2(out)
      out = self.dropout2(out)
      out = self.lstm3(out)
      # out = self.dropout3(out)
    else:
      out = self.lstm1(input_tensor)
      out = self.lstm2(out)
      out = self.lstm3(out)
    return out


class CustomNonPaddingTokenLoss(tf.keras.losses.Loss):
  def __init__(self, name="custom_lfp_loss", ignore_index=-100):
    super().__init__(name=name)
    self.ignore_index = ignore_index

  def call(self, y_true, y_pred):
    # print("labels before ", y_true)
    mask = tf.cast((y_true != self.ignore_index), dtype=tf.int64)
    # print("mask ", mask)
    y_true = y_true * mask
    # print("labels after ", y_true)
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(
      from_logits=True, reduction=tf.keras.losses.Reduction.NONE
    )
    loss = loss_fn(y_true, y_pred)
    # print("loss before ", loss)
    mask = tf.cast(mask, tf.float32)
    loss = loss * mask
    # print("loss after ", loss)
    loss_value = tf.reduce_sum(loss) / tf.reduce_sum(mask)
    # print("loss value ", loss_value)
    # input()
    return loss_value