"""The preprocessing module creates tf.data.Dataset objects for the parser."""

import random
import numpy as np
np.set_printoptions(threshold=np.inf)

from pathlib import Path
import tensorflow as tf
import time

from data.treebank import sentence_pb2
from data.treebank import treebank_pb2
from input.embeddor import Embeddings
from tagset.reader import LabelReader
from tagset.morphology import morph_tags
from typing import List, Dict, Generator, Union
from util.nn import nn_utils
from util import reader
from util import converter

import logging
logging.basicConfig(format='%(levelname)s : %(message)s', level=logging.INFO)

Dataset = tf.data.Dataset
Sentence = sentence_pb2.Sentence
SequenceExample = tf.train.SequenceExample
Treebank = treebank_pb2.Treebank

class SequenceFeature:
  """A sequence feature is a map from a feature name to a list of int values."""
  def __init__(self,
               name: str,
               values: List[int]=[],
               dtype=tf.int64,
               one_hot : bool = False,
               n_values: int = None,
               is_label: bool = False,
               label_indices: List[int] = []):
    # the name of this feature.
    self.name = name
    # the value indices of this feature. This is generated on the fly for each data point in the
    # data during training/test based on a mapping.
    self.values = values
    # feature's data type.
    self.dtype = dtype
    # the number of values this feature holds. Can be useful e.g. during one-hot conversion.
    self.n_values = n_values
    # whether to set up the one hot version of this feature too.
    self.one_hot = self._set_one_hot() if one_hot else None
    # whether this is a label feature.
    self.is_label = is_label
    # the label indices (if the feature is a label feature).
    self.label_indices = label_indices
    # the one hot version

  def index_to_onehot(self, value):
    """Returns the one hot version of a given index or a tensor of indices."""
    if isinstance(value, list):
      value = tf.constant(value)
    if isinstance(value, int):
      one_hot_values = self.one_hot[value]
    elif isinstance(value, tf.Tensor):
      vectors = [tf.expand_dims(self.one_hot[i], 0) for i in value]
      one_hot_values = tf.concat(vectors, 0)
      one_hot_values = tf.cast(one_hot_values, self.dtype)
    else:
      raise ValueError("Input value should be an int or an tf.Tensor")
    return one_hot_values

  def _set_one_hot(self):
    if not self.n_values:
      raise RuntimeError("n_values must be defined for one hot conversion.")
    logging.info("Computing one hot features""")
    return tf.one_hot(range(self.n_values), depth=self.n_values)
  
  def __str__(self):
    return f"""name: {self.name},
            values: {self.values},
            one_hot : {self.one_hot},
            n_values: {self.n_values},
            label_indices: {self.label_indices}"""


class Preprocessor:
  """Prepares data as batched tf.Examples."""

  def __init__(self, *, 
               word_embeddings: Embeddings = None,
               features: List[str],
               labels: List[str],
               one_hot_features: List[str] =  [],
               head_padding_value=0):
    """Preprocessor prepares input datasets for TF models to consume."""
    assert all(o_h_feat in features for o_h_feat in one_hot_features), "Features don't match"

    if word_embeddings:
      self.word_embeddings = word_embeddings
    # The features to use for training the parser.
    self.features = features
    # The label feature; target feature to be predicted by the parser.
    self.labels = labels
    # Which of the features from self.features are one hot
    self.one_hot_features = one_hot_features
    # Initial sequence features
    self.sequence_features_dict = self._sequence_features_dict()
    # Padding value for the heads.
    self.head_padding_value = head_padding_value

  def prepare_sentence_protos(self, path: str):
    """Returns a list of sentence_pb2 formatted protocol buffer objects.

    This function can be used to either read a pbtxt file from disk or
    convert a conllu formatted data to protobuf on the fly.
    """
    if path.endswith("pbtxt"):
      logging.info(f"Reading pbtxt file from {path}")
      trb = reader.ReadTreebankTextProto(path)
      sentences = trb.sentence
    elif path.endswith("conllu"):
      logging.info(f"Converting conll formatted data in {path} to proto")
      conv = converter.Converter(path)
      sentences = conv.ConvertConllToProto(conll_sentences=conv.sentence_list)
    else:
      raise ValueError(
        "Invalid data format, input data should be either pbtxt or conllu!")
    return sentences

  def _sequence_features_dict(self) -> Dict[str, SequenceFeature]:
    """Sets up features and labels to be used by the parsers."""
    logging.info("Setting up sequence features.")
    sequence_features = {}

    # Add a tokens feature (to hold word strings) and a sentence_id feature.
    self.features.extend(["tokens", "sent_id"])
    for feat in self.features:
      if feat == "heads":
        sequence_features[feat] = SequenceFeature(name=feat, is_label=feat in self.labels)
      elif feat == "dep_labels":
        label_dict = LabelReader.get_labels(feat).labels
        label_indices = list(label_dict.values())
        label_feature = SequenceFeature(name=feat,
                                        dtype=tf.float32 if feat in self.one_hot_features else tf.int64,
                                        n_values=len(label_indices),
                                        is_label=feat in self.labels,
                                        label_indices=label_indices if feat in self.labels else [],
                                        one_hot=feat in self.one_hot_features)
        sequence_features[feat] = label_feature
      elif feat in ["tokens" , "sent_id"]:
        sequence_features[feat] = SequenceFeature(name=feat, dtype=tf.string)
      elif feat == "morph":
        sequence_features[feat] = SequenceFeature(name=feat, dtype=tf.float32)
      elif feat in ["words", "pos"]:
        sequence_features[feat] = SequenceFeature(name=feat, dtype=tf.int64)
      else:
        raise ValueError(f"{feat} is not a valid feature.")

    return sequence_features


  def numericalize(self, *,
                   values: List[str],
                   mapping: Union[Dict, Embeddings]) -> List[int]:
    """Returns numeric values (integers) for features based on a mapping.
    Args:
        values: the feature values, usually strings.
        mapping: A mapping that maps string values to integer to be used in NN.
    Returns:
      List of numeric values.
    """
    indices = []
    if isinstance(mapping, Embeddings):
      for value in values:
        try:
          indices.append(mapping.stoi(token=value))
        except KeyError:
          indices.append(mapping.stoi(token="-oov-"))
      return indices
    # TODO: sort out these wrongly annotated values in the data.
    elif isinstance(mapping, dict):
      for value in values:
        try:
          indices.append(mapping[value])
        except KeyError:   
          if value == "Postp":
            value = "PostP"
            indices.append(mapping[value])
          elif value == "Advmod":
            try: 
              indices.append(mapping[value])
            except:
              indices.append(mapping["-pad-"])
          elif value == "A3pl":
            value = "Zero"
            indices.append(mapping[value])
          else:
            logging.warning(f"Key error for value {value}")
            indices.append(mapping["-pad-"])
      return indices
    else:
      raise ValueError("Mapping should be a dict or an Embedding instance.")
  
  def get_index_mappings_for_features(self, feature_names: List[str]) -> Dict:
    "Returns a map dictionary for a feature based on predefined configs"
    mappings = {}
    if "heads" in feature_names:
      logging.debug("No mappings for head feature.")
    if "words" in feature_names:
      if self.word_embeddings:
        mappings["words"] = self.word_embeddings
      else:
        embeddings = nn_utils.load_embeddings()
        mappings["words"] = Embeddings(name="word2vec", matrix=embeddings)
    if "morph" in feature_names:
      mappings["morph"] = LabelReader.get_labels("morph").labels
    if "pos" in feature_names:
      mappings["pos"] = LabelReader.get_labels("pos").labels
    if "category" in feature_names:
      mappings["category"] = LabelReader.get_labels("category").labels
    if "dep_labels" in feature_names:
      mappings["dep_labels"] = LabelReader.get_labels("dep_labels").labels
    if "srl" in feature_names:
      raise RuntimeError("Semantic role features are not supported yet.")
    # print("mappings ", mappings)
    # input("press to cont.")
    return mappings
        
  def make_tf_example(self, *, features: List[SequenceFeature]
                     ) -> SequenceExample:
    """Creates a tf.train.SequenceExample from a single datapoint.
    
    A single datapoint is represented as a List of SequenceFeature objects.
    """
    example = tf.train.SequenceExample()
    for feature in features:
      feature_list = example.feature_lists.feature_list[feature.name]
      if feature.name in ["tokens", "sent_id"]:
        for value in feature.values:
          feature_list.feature.add().bytes_list.value.append(value)
      elif feature.name == "morph":
        for value in feature.values:
          feature_list.feature.add().float_list.value.extend(value)
          # feature_list.feature.add().int64_list.value.extend(value)
      else:
        for value in feature.values:
          feature_list.feature.add().int64_list.value.append(value)
    return example
    
  def write_tf_records(self, *, examples: List[SequenceExample], path: str
                      ) -> None:
    """Serializes tf.train.SequenceExamples to tfrecord files."""
    filename = path + ".tfrecords" if not path.endswith(".tfrecords") else path
    writer = tf.io.TFRecordWriter(filename)
    for example in examples:
      writer.write(example.SerializeToString())
    writer.close()
  
  def make_tf_examples(self, *,
                      sentences: List[Sentence]=None
                      ) -> List[SequenceExample]:
    """Returns a list of tf_examples from a treebank or a list of Sentences."""
    tf_examples = []
    feature_mappings = self.get_index_mappings_for_features(self.features)

    counter = 0
    for sentence in sentences:
      counter += 1
      tokens = [token.word.encode("utf-8") for token in sentence.token]
      sent_id = [sentence.sent_id.encode("utf-8")] * len(sentence.token)
      self.sequence_features_dict["tokens"].values = tokens
      self.sequence_features_dict["sent_id"].values = sent_id
      if "words" in self.features:
        word_indices = self.numericalize(
                          values=[token.word for token in sentence.token],
                          mapping=feature_mappings["words"])
        self.sequence_features_dict["words"].values = word_indices
      if "category" in self.features:
        cat_indices = self.numericalize(
                          values=[token.category for token in sentence.token],
                          mapping=feature_mappings["category"])
        self.sequence_features_dict["category"].values = cat_indices
      if "pos" in self.features:
        pos_indices = self.numericalize(
                          values=[token.pos for token in sentence.token],
                          mapping=feature_mappings["pos"])
        self.sequence_features_dict["pos"].values = pos_indices
      if "morph" in self.features:
        morph_features = []
        for token in sentence.token:
          morph_vector = np.zeros(len(feature_mappings["morph"]), dtype=float)
          tags = morph_tags.from_token(token=token)
          morph_indices = self.numericalize(
                          values=tags,
                          mapping=feature_mappings["morph"])
          # print(list(zip(tags, morph_indices)))
          # We put 1 to the indices for the active morphology features.
          np.put(morph_vector, morph_indices, [1])
          morph_features.append(morph_vector)
        self.sequence_features_dict["morph"].values = morph_features
      if "dep_labels" in self.features:
        sentence.token[0].label = "TOP" # workaround key errors
        dep_indices = self.numericalize(
                          values=[token.label for token in sentence.token],
                          mapping=feature_mappings["dep_labels"])
        self.sequence_features_dict["dep_labels"].values = dep_indices
      if "heads" in self.features:
        self.sequence_features_dict["heads"].values = [
          token.selected_head.address for token in sentence.token]

      example = self.make_tf_example(features=self.sequence_features_dict.values())
      # print("example ", example)
      # input("press to cont.")
      tf_examples.append(example)
    logging.info(f"Converted {counter} sentences to tf examples.")
    return tf_examples

  def read_dataset_from_tfrecords(self, *,
                                  batch_size: int = 10,
                                  records: str) -> Dataset:
    """Reads a tensorflow dataset from a saved tfrecords path."""
    
    _sequence_features = {}
    _dataset_shapes = {}
    _padded_shapes = {}
    _padding_values={}
    
    # Create a dictionary description of the tensors to parse the features from the
    # tf.records.
    for feature in self.sequence_features_dict.values():
      if feature.name == "morph":
        _sequence_features[feature.name]=tf.io.FixedLenSequenceFeature(
            shape=[56], dtype=feature.dtype)
        _dataset_shapes[feature.name]=tf.TensorShape([None, 56])
        _padded_shapes[feature.name]=tf.TensorShape([None, 56])
        _padding_values[feature.name] = tf.constant(0, dtype=feature.dtype)
      elif feature.name in ["tokens", "sent_id"]:
        _sequence_features[feature.name]=tf.io.FixedLenSequenceFeature(
          shape=[], dtype=feature.dtype)
        _dataset_shapes[feature.name]=tf.TensorShape([None])
        _padded_shapes[feature.name]=tf.TensorShape([None])
        _padding_values[feature.name] = tf.constant("-pad-", dtype=feature.dtype)
      elif feature.name == "heads":
        _sequence_features[feature.name] = tf.io.FixedLenSequenceFeature(
          shape=[], dtype=feature.dtype)
        _dataset_shapes[feature.name]=tf.TensorShape([None])
        _padded_shapes[feature.name]=tf.TensorShape([None])
        # The padding value for heads is -1.
        _padding_values[feature.name] = tf.constant(self.head_padding_value, dtype=feature.dtype)
      elif feature.name == ["dep_labels"]  and feature.one_hot is not None:
        _sequence_features[feature.name] = tf.io.FixedLenSequenceFeature(
          shape=[43], dtype=feature.dtype)
        _dataset_shapes[feature.name] = tf.TensorShape([None, 43])
        _padded_shapes[feature.name] = tf.TensorShape([None, 43])
        _padding_values[feature.name] = tf.constant(0, dtype=feature.dtype)
      else:
        _sequence_features[feature.name]=tf.io.FixedLenSequenceFeature(
            shape=[], dtype=feature.dtype)
        _dataset_shapes[feature.name]=tf.TensorShape([None])
        _padded_shapes[feature.name]=tf.TensorShape([None])
        _padding_values[feature.name] = tf.constant(0, dtype=feature.dtype)

    def _parse_tf_records(record):
      """Returns a dictionary of tensors."""
      _, parsed_example = tf.io.parse_single_sequence_example(
        serialized=record,
        sequence_features=_sequence_features
      )
      return {
        feature.name:parsed_example[feature.name] for feature in self.sequence_features_dict.values()
      }
    
    dataset = tf.data.TFRecordDataset([records])
    dataset = dataset.map(_parse_tf_records)    
    dataset = dataset.padded_batch(batch_size,
                                   padded_shapes=_padded_shapes,
                                   padding_values=_padding_values)
    return dataset
    
  def make_dataset_from_generator(self, *,
                                  sentences: List[Sentence],
                                  batch_size: int=50,
                                  generator: Generator=None) -> Dataset:
    """Makes a tensorflow dataset that is shuffled, batched and padded.
    Args:
      sentences: a list of sentence_pb2.Sentence objects.
      batch_size: the size of the dataset batches.
      generator: a generator function.
                 If not specified, the class generator is used.
    """
    if not generator:
      generator = lambda: self._example_generator(sentences)
    
    _output_types={}
    _output_shapes={}
    _padded_shapes={}
    _padding_values={}
    
    for feature in self.sequence_features_dict.values():
      _output_types[feature.name]=feature.dtype

      # Set up the padding values for features.
      if feature.name in ["tokens", "sent_id"]:
        _padding_values[feature.name] = tf.constant("-pad-", dtype=feature.dtype)
      # The padding value for heads is -1.
      elif feature.name == "heads":
        _padding_values[feature.name] = tf.constant(self.head_padding_value, dtype=feature.dtype)
      else:
        _padding_values[feature.name] = tf.constant(0, dtype=feature.dtype)

      # Set up the padded output shapes for features.
      if feature.name == "morph":
        _output_shapes[feature.name]=tf.TensorShape([None, 56]) # (, 56)
        _padded_shapes[feature.name]=tf.TensorShape([None, 56])
      elif feature.name =="dep_labels" and feature.one_hot is not None:
        _output_shapes[feature.name] = tf.TensorShape([None, 43])
        _padded_shapes[feature.name] = tf.TensorShape([None, 43])
      else:
        _output_shapes[feature.name]=[None]
        _padded_shapes[feature.name]=tf.TensorShape([None])

    # Generate the dataset.
    dataset = tf.data.Dataset.from_generator(
      generator,
      output_types=_output_types,
      output_shapes=_output_shapes
    )
    dataset = dataset.padded_batch(batch_size,
                                   padded_shapes=_padded_shapes,
                                   padding_values=_padding_values)

    logging.info("Shuffling dataset.")
    dataset = dataset.shuffle(buffer_size=5073)

    return dataset

  def _example_generator(self, sentences: List[Sentence]):
    print(f"Total sentences {len(sentences)}")
    # input("press to cont.")
    feature_mappings = self.get_index_mappings_for_features(self.features)
    yield_dict = {}
    
    for sentence in sentences:
      yield_dict["tokens"] = [
        token.word.encode("utf-8") for token in sentence.token]
      yield_dict["sent_id"] = [
        sentence.sent_id.encode("utf-8")] * len(sentence.token)
      for feature_name in self.features:
        if feature_name == "words":
          yield_dict[feature_name] = self.numericalize(
            values=[token.word for token in sentence.token],
            mapping=feature_mappings["words"])
        if feature_name == "pos":
          yield_dict[feature_name] = self.numericalize(
            values=[token.pos for token in sentence.token],
            mapping=feature_mappings["pos"])
        if feature_name == "category":
          yield_dict[feature_name] = self.numericalize(
            values=[token.category for token in sentence.token],
            mapping=feature_mappings["category"])
        if feature_name == "morph":
          morph_features = []
          for token in sentence.token:
            morph_vector = np.zeros(len(feature_mappings["morph"]), dtype=float)
            tags = morph_tags.from_token(token=token)
            morph_indices = self.numericalize(
                            values=tags,
                            mapping=feature_mappings["morph"]
            )
            np.put(morph_vector, morph_indices, [1])
            morph_features.append(morph_vector)
          yield_dict[feature_name] = morph_features
        if feature_name == "dep_labels":
          label_feature = self.sequence_features_dict[feature_name]
          sentence.token[0].label = "TOP" # workaround key errors
          vectors = self.numericalize(
            values=[token.label for token in sentence.token],
            mapping=feature_mappings["dep_labels"])
          if label_feature.one_hot is not None:
            yield_dict[feature_name] = label_feature.index_to_onehot(vectors)
          else:
            yield_dict[feature_name] = vectors
        if feature_name == "heads":
          yield_dict[feature_name] = [
            token.selected_head.address for token in sentence.token]
      yield yield_dict


def make_tfrecords(path: str, sample: int = 0):
  """Converts a set of tf.Examples to tf_records"""
  embeddings = nn_utils.load_embeddings()
  word_embeddings = Embeddings(name="word2vec", matrix=embeddings)
  input_path = Path(path)
  output_path = str(input_path.parent) + "/" + str(input_path.stem) + ".tfrecords"
  prep = Preprocessor(word_embeddings=word_embeddings,
                      features=["words", "tokens", "sent_id", "pos", "morph", "dep_labels", "heads"],
                      labels=["dep_labels", "heads"])
  sentences = prep.prepare_sentence_protos(path=path)
  if sample > 0:
    sentences = random.sample(sentences, sample)
    output_path = str(input_path.parent) + "/" + str(input_path.stem) + "sample_" + str(sample) + ".tfrecords"
  tf_examples = prep.make_tf_examples(sentences=sentences)
  prep.write_tf_records(examples=tf_examples, path=output_path)
  logging.info(f"tf_records written to {output_path}")


if __name__ == "__main__":

  data = "data/UDv29/test/tr/tr_boun-ud-test.pbtxt"
  make_tfrecords(data)

  ''' Make a dataset with example_generator
  # Load word embeddings
  embeddings = nn_utils.load_embeddings()
  word_embeddings = Embeddings(name="word2vec", matrix=embeddings)
  
  # Initialize the preprocessor.
  prep = Preprocessor(word_embeddings=word_embeddings,
                      features=["words", "pos", "morph", "dep_labels", "heads"],
                      labels=["dep_labels", "heads"],
                      # one_hot_features=["dep_labels"]
                      )
  datapath = "./data/UDv29/train/tr/tr_boun-ud-train-random1.pbtxt"
  sentences = prep.prepare_sentence_protos(path=datapath)
  dataset = prep.make_dataset_from_generator(sentences=sentences, batch_size=1)
  
  for batch in dataset:
    print(batch)
    input("press to cont.")
  '''

  '''Make a dataset as tf_records 
  # Make a dataset and save it
  tf_examples = prep.make_tf_examples(sentences=sentences)
  prep.write_tf_records(examples=tf_examples,
                        path="./input/tr_boun-train-random1.tfrecords")
  

  # Read dataset from saved tfrecords
  dataset = prep.read_dataset_from_tfrecords(
    records="./input/treebank_boun-train-random1.tfrecords")
  print("tf examples")
  for batch in dataset:
    print(batch)
  '''



