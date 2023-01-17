"""This is a seq2seq transformer label first parsing model based on pretrained BERT."""
import datasets
import tensorflow as tf
import transformers

from datasets import load_dataset
from tensorflow.keras import layers, metrics, losses, optimizers
from transformers import AutoConfig, AutoTokenizer, TFAutoModelForTokenClassification, DataCollatorForTokenClassification
from transformers import create_optimizer, set_seed

from parser import base_parser
from parser.utils import layer_utils
from util import reader

import logging
logging.basicConfig(format='%(levelname)s : %(message)s', level=logging.INFO)

from typing import List


tokenizer=AutoTokenizer.from_pretrained("bert-base-multilingual-cased", use_fast=True)
collate_fn = DataCollatorForTokenClassification(tokenizer=tokenizer, return_tensors='tf')

raw_dataset = load_dataset("./transformer/hf/dataset/hf_data_loader.py")
features = raw_dataset["train"].features
label_list = features["dep_labels"].feature.names
label_to_id = {i: i for i in range(len(label_list))}
print("label to id ", label_to_id)


def tokenize_and_align_labels(examples):
  tokenized_inputs = tokenizer(examples["tokens"],
                               max_length=False, padding=False,
                               truncation=True, is_split_into_words=True)
  # print("tokenized inputs ", tokenized_inputs)
  # input()
  labels = []
  for i, label in enumerate(examples["dep_labels"]):
    # print("i ", i, "label ", label)
    # print("words ", examples["tokens"][i])
    # input()
    word_ids = tokenized_inputs.word_ids(batch_index=i)
    # print("word ids ", word_ids)
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
        label_ids.append(label_to_id[label[word_idx]])
      previous_word_idx = word_idx

    labels.append(label_ids)
    # print("label ids ", label_ids)
  tokenized_inputs["labels"] = labels
  # print("tokenized inputs ", tokenized_inputs)
  # input()
  return tokenized_inputs


class BertLabelFirstParser:
  """A label first parser predicts labels before predicting heads."""
  def __init__(self, *,
               name:str = "Bert_Based_Label_First_Parsing_Model",
               num_labels: int,
               language="tr",
               features=["pos", "morph", "dep_labels"],
               bert_model_name="bert-base-multilingual-cased",
               log_dir=None,
               test_every: int = 5):
    self.bert_config = AutoConfig.from_pretrained(bert_model_name, num_labels=num_labels)
    self.bert_model = transformers.TFAutoModelForTokenClassification.from_pretrained(
      bert_model_name, config=self.bert_config)
    # self.raw_dataset = load_dataset("./transformer/hf/dataset/hf_data_loader.py")
    self._use_pos = "pos" in features
    self._use_morph = "morph" in features
    self._use_dep_labels = "dep_labels" in features
    # self.label_to_id = self._get_label_to_id()
    self.model = self._parsing_model(model_name=name)
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
      remove_columns=raw_dataset["train"].column_names,
      desc="Running tokenizer on dataset")

    return processed_dataset

  def _to_tf_dataset(self, train_dataset, validation_dataset):
    dataset_options = tf.data.Options()
    dataset_options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.OFF
    tf_train_dataset = self.bert_model.prepare_tf_dataset(
      train_dataset,
      collate_fn=collate_fn,
      batch_size=8,
      shuffle=True,
    ).with_options(dataset_options)

    tf_validation_dataset = self.bert_model.prepare_tf_dataset(
      validation_dataset,
      collate_fn=collate_fn,
      batch_size=8,
      shuffle=False,
    ).with_options(dataset_options)

    return tf_train_dataset, tf_validation_dataset

  def train(self, epochs):
    processed_dataset = self._process_dataset()
    tf_train_dataset, tf_validation_dataset = self._to_tf_dataset(processed_dataset["train"],
                                                                  processed_dataset["validation"])
    for example in tf_train_dataset:
      print(example)
      y = input()
      if y == "c":
        break
    num_train_steps = int(len(tf_train_dataset)) * epochs
    optimizer, lr_schedule = create_optimizer(
      init_lr=5e-05,
      num_train_steps=num_train_steps,
      num_warmup_steps=0,
      adam_beta1=0.9,
      adam_beta2=0.999,
      adam_epsilon=1e-08,
      weight_decay_rate=0.0,
      adam_global_clipnorm=1.0
    )
    self.model.compile(optimizer=optimizer)
    print(self.model)
    input()
    self.model.fit(tf_train_dataset, validation_data=tf_validation_dataset,
                   epochs=epochs)


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
                                                                output_dim=32, name="pos_embeddings",
                                                                trainable=True)

    self.pretrained_label_embeddings = layer_utils.EmbeddingLayer(input_dim=43,
                                                                  output_dim=50,
                                                                  name="label_embeddings",
                                                                  trainable=True)

    self.concatenate = layers.Concatenate(name="concat")

    self.head_perceptron = layer_utils.Perceptron(n_units=256,
                                                  activation="relu",
                                                  name="head_mlp")
    self.dep_perceptron = layer_utils.Perceptron(n_units=256,
                                                 activation="relu",
                                                 name="dep_mlp")
    self.edge_scorer = layer_utils.EdgeScorer(n_units=256, name="edge_scorer")

  def call(self, inputs, training=True): # inputs = Dict[str, tf.keras.Input]
    """Forward pass.
    Args:
      inputs: Dict[str, tf.keras.Input]. This consist of
        words: Tensor of shape (batch_size, seq_len)
        pos: Tensor of shape (batch_size, seq_len)
        morph: Tensor of shape (batch_size, seq_len, 66)
        dep_labels: Tensor of shape (batch_size, seq_len)
      The boolean values set up during the initiation of the model determines
      which one of these features to use or not.
    Returns:
      A dict which conteins:
        edge_scores: [batch_size, seq_len, seq_len] head preds for all tokens (i.e. 10, 34, 34)
        label_scores: [batch_size, seq_len, n_labels] label preds for tokens (i.e. 10, 34, 36)
    """
    pass
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