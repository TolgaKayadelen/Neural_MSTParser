"""This is a seq2seq transformer label first parsing model based on pretrained BERT."""
import os
import collections
import datasets
import datetime
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
from util import reader

import logging
logging.basicConfig(format='%(levelname)s : %(message)s', level=logging.INFO)

from typing import List

disable_caching()


tokenizer=AutoTokenizer.from_pretrained("bert-base-multilingual-cased", use_fast=True)
collate_fn = DataCollatorForTokenClassification(tokenizer=tokenizer, return_tensors='tf')


raw_dataset = load_dataset("./transformer/hf/dataset/hf_data_loader.py")
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
  # print("tokenized inputs ", tokenized_inputs)
  # input()
  labels = []
  words = []
  for i, label in enumerate(examples["dep_labels"]):
    # print("i ", i, "label ", label)
    # print("words ", examples["tokens"][i])
    # input()
    word_ids = tokenized_inputs.word_ids(batch_index=i)
    # print("word ids ", word_ids)

    # print("word indexes ", word_indexes)
    # input()
    previous_word_idx = None
    label_ids = []
    for word_idx in word_ids:
      # Special tokens have a word id that is None. We set the label to -100 so they are automatically
      # ignored in the loss function.
      # print("word idx ", word_idx)
      # input()
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
    # print("labels ", labels)
    # print("words ", words)
    # input()
  tokenized_inputs["labels"] = labels
  tokenized_inputs["words"] = words
  # print("tokenized inputs ", tokenized_inputs)
  # input()
  return tokenized_inputs


class BertLabelFirstParser:
  """A label first parser predicts labels before predicting heads."""
  def __init__(self, *,
               name:str = "bert_lfp_parser",
               num_labels: int,
               language="tr",
               features=["pos", "morph", "dep_labels"],
               bert_model_name="bert-base-multilingual-cased",
               pretrained_bert_model_path=None,
               log_dir=None,
               test_every: int = 5,
               train_batch_size: int = 1):
    if pretrained_bert_model_path is not None:
      self.bert_model=transformers.TFAutoModelForTokenClassification.from_pretrained(pretrained_bert_model_path)
    else:
      self.bert_config = AutoConfig.from_pretrained(bert_model_name, num_labels=num_labels)
      self.bert_model = transformers.TFAutoModelForTokenClassification.from_pretrained(
        bert_model_name, config=self.bert_config)
    self._use_pos = "pos" in features
    self._use_morph = "morph" in features
    self._use_dep_labels = "dep_labels" in features
    # self.label_to_id = self._get_label_to_id()
    self.model = self._parsing_model(model_name=name)
    self.metrics = collections.Counter()
    self.output_dir = "./transformer/hf/pretrained/inheritance_test"
    self.train_batch_size=train_batch_size
    set_seed(42)



  @property
  def _optimizer(self):
    return tf.keras.optimizers.Adam(0.001, beta_1=0.9, beta_2=0.9)

  def _training_metrics(self):
    return {
      "heads": metrics.SparseCategoricalAccuracy(),
      "labels": metrics.SparseCategoricalAccuracy()
    }

  @property
  def _head_loss_function(self):
    """Returns loss per token for head prediction.
    As we use the SparseCategoricalCrossentropy function, we expect the target labels
    to be provided as integers indexing the correct labels rather than one hot vectors.
    For details, refer to:
    https://www.tensorflow.org/api_docs/python/tf/keras/losses/SparseCategoricalCrossentropy
    """
    return losses.SparseCategoricalCrossentropy(from_logits=True,
                                                reduction=tf.keras.losses.Reduction.NONE)

  @property
  def _label_loss_function(self):
    """Returns loss per token for label prediction.

    As we use the SparseCategoricalCrossentropy function, we expect the target labels to be
    to be provided as integers indexing the correct labels rather than one hot vectors. The predictions
    should be keeping the probs as float values for each label per token.

    For details, refer to:
    https://www.tensorflow.org/api_docs/python/tf/keras/losses/SparseCategoricalCrossentropy"""

    return losses.SparseCategoricalCrossentropy(from_logits=True,
                                                reduction=tf.keras.losses.Reduction.NONE)

  @property
  def inputs(self):
    input_dict = {}
    token_inputs = tf.keras.Input(shape=(None, ), name="tokens", dtype=tf.string)
    pos_inputs = tf.keras.Input(shape=(None,), name="pos")
    morph_inputs = tf.keras.Input(shape=(None, 56), name="morph")
    label_inputs = tf.keras.Input(shape=(None, ), name="labels")
    if self._use_pos:
      input_dict["pos"] = pos_inputs
    if self._use_morph:
      input_dict["morph"] = morph_inputs
    input_dict["labels"] = label_inputs
    input_dict["tokens"] = token_inputs
    return input_dict

  # def _get_label_to_id(self):
  #  features = self.raw_dataset["train"].features
  #  label_list = features["dep_labels"].feature.names
    # No need to convert the labels since they are already ints.
  #  label_to_id = {i: i for i in range(len(label_list))}
  #  return label_to_id

  def _n_words_in_batch(self, words, pad_mask=None):
    words_reshaped = tf.reshape(words, shape=pad_mask.shape)
    return len(tf.boolean_mask(words_reshaped, pad_mask))

  def _parsing_model(self, model_name):
    print(f"""Using features pos: {self._use_pos}, morph: {self._use_morph},
              dep_labels: {self._use_dep_labels}""")
    model = BertLabelFirstParsingModel(
      name=model_name,
      use_pos=self._use_pos,
      use_morph=self._use_morph,
      bert_model=self.bert_model,

    )
    # model(inputs=self.inputs)
    return model

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
      "dep_labels": [None],
      "input_ids": [None],
      "token_type_ids": [None],
      "attention_mask": [None],
      "words": [None]
    }
    padded_shapes = {
      "sent_id" : [None],
      "tokens": [None],
      "dep_labels": [None],
      "input_ids": [None],
      "token_type_ids": [None],
      "attention_mask": [None],
      "words": [None]
    }
    output_types = {
      "sent_id" : tf.string,
      "tokens": tf.string,
      "dep_labels": tf.int64,
      "input_ids": tf.int64,
      "token_type_ids": tf.int64,
      "attention_mask": tf.int64,
      "words": tf.int64
    }

    _padding_values = {
      "sent_id" : tf.constant("-pad-", dtype=tf.string),
      "tokens": tf.constant("-pad-", dtype=tf.string),
      "dep_labels": tf.constant(0, dtype=tf.int64),
      "input_ids": tf.constant(0, dtype=tf.int64),
      "token_type_ids": tf.constant(0, dtype=tf.int64),
      "attention_mask": tf.constant(0, dtype=tf.int64),
      "words": tf.constant(-100, dtype=tf.int64),
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
      example["dep_labels"].extend([0] * diff_len)
      yield_dict["tokens"] = example["tokens"]
      yield_dict["dep_labels"] = example["dep_labels"]
      yield_dict["input_ids"] = example["input_ids"]
      yield_dict["token_type_ids"] = example["token_type_ids"]
      yield_dict["attention_mask"] = example["attention_mask"]
      yield_dict["words"] = example["words"]

      yield yield_dict

  def train(self, epochs):
    processed_dataset = self._process_dataset()
    for example in processed_dataset["train"]:
      print("example ", example)
      input()
      break
    # tf_train_dataset, tf_validation_dataset = self._to_tf_dataset(processed_dataset["train"],
    #                                                               processed_dataset["validation"])
    # tf_train_dataset = processed_dataset["train"].with_format("tf")
    tf_train_dataset = self.make_dataset_from_generator(processed_dataset["train"], batch_size=2)
    for example in tf_train_dataset:
      print(example)
      input()
    loss = CustomNonPaddingTokenLoss()
    # for example in tf_validation_dataset:
    #   print(example)
    #   input()
    #   break
    num_train_steps = int(len(tf_train_dataset)) * epochs
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
     for k, v in self.metrics.items():
       self.metrics[k] = 0.0
     epoch_loss = 0.0
     for step, batch in enumerate(tf_train_dataset):
       with tf.GradientTape() as tape:
          inputs, labels = batch
          print("labels ", labels)
          input()
          inputs["labels"] = labels
          print("inputs ", inputs)
          input()
          predictions = self.model(inputs)
          logits = predictions.logits
          # print("logits ", logits)
          # input()
          loss_value = loss(labels, logits)
          epoch_loss += loss_value
          if step > 0:
            print("step ", step)
            loss_avg = epoch_loss / step
            print(f"Running Loss ", loss_avg)

       grads = tape.gradient(loss_value, self.model.trainable_weights)
       optimizer.apply_gradients(zip(grads, self.model.trainable_weights))
    logging.info(f"End of epoch {epoch}. Saving models.")

    # Saving
    self.save()
    # self.model.bert_model.save_pretrained(self.output_dir)
    # path = os.path.join(self.output_dir, self.model.name)
    # os.mkdir(path)
    # self.model.save_weights(os.path.join(path, self.model.name), save_format="tf")
    # logging.info(f"Saved model to  {path}")

    # self.test(tf_validation_dataset)

  def _flatten(self, _tensor):
    """The tensor should be a 2D tensor of shape batch_size, n_dim.
      Returns a tensor of shape (batch_size*n_dim, 1)"""
    reshaped = tf.reshape(_tensor, shape=(_tensor.shape[0]*_tensor.shape[1]))
    return reshaped

  def test(self, test_set):
    for example in test_set:
      inputs, labels = example
      predictions = self.model(inputs, training=False)
      logits = predictions.logits
      # print("logits ", logits)
      preds = tf.argmax(logits, -1)
      # print("preds ", preds)
      # input()
      true_preds = np.array([
        p.numpy() for (p, l) in zip(self._flatten(preds), self._flatten(labels)) if l != -100])
      # print("true preds ", true_preds)
      true_labels = np.array([
        l.numpy() for (p, l) in zip(self._flatten(preds), self._flatten(labels)) if l != -100])
      # print("true labels ", true_labels)
      # input()
      assert len(true_labels) == len(true_preds), "Fatal: Token mismatch!"
      n_tokens = len(true_labels)
      self.metrics["n_tokens"] += n_tokens
      n_accurate_labels = np.sum(true_preds == true_labels)
      # print("n accurate labels ", n_accurate_labels)
      self.metrics["n_accurate_labels"] += n_accurate_labels
      self.metrics["accuracy"] = self.metrics["n_accurate_labels"] / self.metrics["n_tokens"]
      print("accuracy ", self.metrics["accuracy"])
      # input()
    print(self.metrics)

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
    self.model.bert_model.save_pretrained(os.path.join(save_dir, "bert"))
    self.model.save_weights(os.path.join(save_dir, self.model.name), save_format="tf")
    logging.info(f"Saved model to  {save_dir}")


class BertLabelFirstParsingModel(tf.keras.Model):
  """Bert based transformer model predicting labels before edges."""
  def __init__(self, *,
               name="Bert_Based_Label_First_Parsing_Model",
               bert_model,
               use_pos:bool = True,
               use_morph:bool=True,
               ):
    super(BertLabelFirstParsingModel, self).__init__(name=name)

    self.bert_model = bert_model
    self.use_pos = use_pos
    self.use_morph = use_morph

    self.pretrained_pos_embeddings = layer_utils.EmbeddingLayer(input_dim=37,
                                                                output_dim=32,
                                                                name="pos_embeddings",
                                                                trainable=False)

    self.pretrained_label_embeddings = layer_utils.EmbeddingLayer(input_dim=43,
                                                                  output_dim=50,
                                                                  name="label_embeddings",
                                                                  trainable=False)

    self.concatenate = layers.Concatenate(name="concat")

    self.head_perceptron = layer_utils.Perceptron(n_units=256,
                                                  activation="relu",
                                                  name="head_mlp")
    self.dep_perceptron = layer_utils.Perceptron(n_units=256,
                                                 activation="relu",
                                                 name="dep_mlp")
    self.edge_scorer = layer_utils.EdgeScorer(n_units=256, name="edge_scorer")


  def call(self, inputs):

    """Forward pass."""
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]
    token_type_ids=inputs["token_type_ids"]
    labels = inputs["labels"]
    predictions = self.bert_model(input_ids, attention_mask, token_type_ids)
    logits = predictions.logits
    # print("logits ", logits)
    preds = tf.argmax(logits, -1)
    # preds_flat = tf.expand_dims(tf.reshape(preds, shape=(preds.shape[0]*preds.shape[1])), 0)
    print("preds ", preds)
    # labels_flat = tf.expand_dims(tf.reshape(labels, shape=(labels.shape[0]*labels.shape[1])), 0)
    print("labels ", labels)
    # input()
    true_preds = [[
      p.numpy() for (p, l) in zip(pred, label) if l != -100] for (pred, label) in zip(preds, labels)]
    print("true preds ",  true_preds)
    maxlen = max(len(l) for l in true_preds)
    true_preds = tf.convert_to_tensor([l+(maxlen-len(l))*[0] for l in true_preds], dtype=tf.int64)
    print("true preds ", true_preds)
    input()
    return predictions
    """
    concat_list = []
    print(inputs["tokens"])
    input()
    tokenized = self.tokenizer(inputs["tokens"], is_split_into_words=True, return_tensors='tf')
    print("tokenized ", tokenized)
    input()
    input_ids, attn_mask = tokenized["input_ids"], tokenized["attention_mask"]
    print("input ids ", input_ids)
    print("attention mask ", attn_mask)
    outputs = self.bert(input_ids, attn_mask, training=training)
    print("outputs ", outputs)
    input()
    unaligned_labels = tf.argmax(outputs.logits, -1)
    print("unaligned labels ", unaligned_labels)
    input()
    # label_preds = self._align_labels()
    # label_embeddings = self.label_embeddings(label_preds)
    # concat_list.append(label_embeddings)
    if self.use_pos:
      pos_inputs = inputs["pos"]
      pos_features = self.pos_embeddings(pos_inputs)
      concat_list.append(pos_features)
    if self.use_morph:
      morph_inputs = inputs["morph"]
      concat_list.append(morph_inputs)
    if len(concat_list) > 1:
      encoding_for_parse = self.concatenate(concat_list)
      # print("sentence repr ", sentence_repr)

    h_arc_head = self.head_perceptron(encoding_for_parse)
    h_arc_dep = self.dep_perceptron(encoding_for_parse)
    edge_scores = self.edge_scorer(h_arc_head, h_arc_dep)
    return {"edges": edge_scores}
    """

class CustomNonPaddingTokenLoss(tf.keras.losses.Loss):
  def __init__(self, name="custom_lfp_loss"):
    super().__init__(name=name)
    self.ignore_index = -100

  def call(self, y_true, y_pred):
    # print("labels before ", y_true)
    mask = tf.cast((y_true != self.ignore_index), dtype=tf.int64)
    # print("mask ", mask)
    y_true = y_true * mask
    # print("labels after ", y_true)
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(
      from_logits=True, reduction=tf.keras.losses.Reduction.NONE
    )
    # input()
    loss = loss_fn(y_true, y_pred)
    # print("loss before ", loss)
    mask = tf.cast(mask, tf.float32)
    loss = loss * mask
    # print("loss after ", loss)
    loss_value = tf.reduce_sum(loss) / tf.reduce_sum(mask)
    # print("loss value ", loss_value)
    # input()
    return loss_value