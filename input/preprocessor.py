"""The preprocessing module creates tf.data.Dataset objects for the parser."""

import numpy as np
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
               one_hot: bool = False,
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
    # whether this is a one hot feature.
    self.one_hot = one_hot
    # whether this is a label feature.
    self.is_label = is_label
    # the label indices (if the feature is a label feature).
    self.label_indices = label_indices
    
  def _convert_to_one_hot(self, values, n_values):
    """Converts an integer array to a one hot vector.
    
    An integer array of shape (1, m) becomes a one hot vector of shape
    (len(indices), n_labels).
    """
    if isinstance(values, list):
      logging.info("Converting list to array")
      values = np.array(values)
    if not isinstance(values, np.ndarray):
      raise ValueError(f"""Can only convert numpy arrays to one hot vectors,
                      received: {type(values)}""")
    return np.eye(n_values)[values.reshape(-1)]
  
  @property
  def one_hot(self):
    return self._one_hot
  
  @one_hot.setter
  def one_hot(self, one_hot):
    if one_hot:
      if not self.n_values:
        raise RuntimeError("n_values must be defined for one hot conversion.")
      logging.info("Converting values to one hot objects""")
      self._one_hot = self._convert_to_one_hot(self.values, self.n_values)
    else:
      self._one_hot = one_hot
    return self._one_hot
  
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
               features = List[str],
               labels = List[str]):
    if word_embeddings:
      self.word_embeddings = word_embeddings
    # The features to use for training the parser.
    self.features = features
    # The label feature; target feature to be predicted by the parser.
    self.labels = labels
    # Initial sequence features
    self.sequence_features_dict = self._sequence_features_dict()

  def _sequence_features_dict(self) -> Dict[str, SequenceFeature]:
    """Sets up features and labels to be used by the parsers."""
    logging.info("Setting up sequence features.")
    sequence_features = {}
    # Add a tokens feature (to hold word strings) and a sentence_id feature.
    sequence_features["tokens"] = SequenceFeature(name="tokens", dtype=tf.string)
    sequence_features["sent_id"] = SequenceFeature(name="sent_id", dtype=tf.string)

    for feat in self.features:
      if not feat in self.labels:
        sequence_features[feat] = SequenceFeature(name=feat)
      else:
        if feat == "heads":
          # We don't need to get the label_indices for "heads" as we don't do
          # label prediction on them in a typical sense.
          sequence_features[feat] = SequenceFeature(name=feat, is_label=True)
        else:
          label_dict = LabelReader.get_labels(feat).labels
          label_indices = list(label_dict.values())
          label_feature = SequenceFeature(name=feat,
                                          n_values=len(label_indices),
                                          is_label=True,
                                          label_indices=label_indices)
          sequence_features[feat] = label_feature

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
          feature_list.feature.add().int64_list.value.extend(value)
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
                      from_treebank: Treebank=None,
                      from_sentences: List[Sentence]=None
                      ) -> List[SequenceExample]:
    """Returns a list of tf_examples from a treebank or a list of Sentences."""
    tf_examples = []
    feature_mappings = self.get_index_mappings_for_features(self.features)

    counter = 0
    if from_treebank is not None:
      logging.info(f"Creating dataset from treebank {from_treebank}")
      sentences = from_treebank.sentence
    elif from_sentences is not None:
      logging.info(f"Creating dataset from list of sentences")
      sentences = from_sentences
    else:
      raise ValueError("Neither a treebank nor a list of sentences provided.")
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
          morph_vector = np.zeros(len(feature_mappings["morph"]), dtype=int)
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
            shape=[66], dtype=feature.dtype)
        _dataset_shapes[feature.name]=tf.TensorShape([None, 66])
        _padded_shapes[feature.name]=tf.TensorShape([None, 66])
        _padding_values[feature.name] = tf.constant(0, dtype=tf.int64)
      elif feature.name in ["tokens", "sent_id"]:
        _sequence_features[feature.name]=tf.io.FixedLenSequenceFeature(
          shape=[], dtype=feature.dtype)
        _dataset_shapes[feature.name]=tf.TensorShape([None])
        _padded_shapes[feature.name]=tf.TensorShape([None])
        _padding_values[feature.name] = tf.constant("-pad-", dtype=tf.string)
      else:
        _sequence_features[feature.name]=tf.io.FixedLenSequenceFeature(
            shape=[], dtype=feature.dtype)
        _dataset_shapes[feature.name]=tf.TensorShape([None])
        _padded_shapes[feature.name]=tf.TensorShape([None])
        _padding_values[feature.name] = tf.constant(0, dtype=tf.int64)

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
    
  def make_dataset_from_generator(self, *, path: str, batch_size: int=50,
                                  generator: Generator=None) -> Dataset:
    """Makes a tensorflow dataset that is shuffled, batched and padded.
    Args:
      path: path to the dataset treebank.
      batch_size: the size of the dataset batches.
      generator: a generator function.
                 If not specified, the class generator is used.
    """
    if not generator:
      generator = lambda: self._example_generator(path)
    
    _output_types={}
    _output_shapes={}
    _padded_shapes={}
    _padding_values={}
    
    for feature in self.sequence_features_dict.values():
      _output_types[feature.name]=feature.dtype
      
      if feature.name in ["tokens", "sent_id"]:
        _padding_values[feature.name] = tf.constant("-pad-", dtype=tf.string)
      else:
        _padding_values[feature.name] = tf.constant(0, dtype=tf.int64)
      
      if feature.name == "morph":
        _output_shapes[feature.name]=tf.TensorShape([None, 66]) # (, 66)
        _padded_shapes[feature.name]=tf.TensorShape([None, 66])
      else:
        _output_shapes[feature.name]=[None]
        _padded_shapes[feature.name]=tf.TensorShape([None])
    
    dataset = tf.data.Dataset.from_generator(
      generator,
      output_types=_output_types,
      output_shapes=_output_shapes
    )
    dataset = dataset.padded_batch(batch_size,
                                   padded_shapes=_padded_shapes,
                                   padding_values=_padding_values)
    dataset = dataset.shuffle(buffer_size=5073)
    # TODO dataset = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE,drop_remainder=True)

    return dataset

  def _example_generator(self, path: str):
    trb = reader.ReadTreebankTextProto(path)
    sentences = trb.sentence
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
            morph_vector = np.zeros(len(feature_mappings["morph"]), dtype=int)
            tags = morph_tags.from_token(token=token)
            morph_indices = self.numericalize(
                            values=tags,
                            mapping=feature_mappings["morph"]
            )
            np.put(morph_vector, morph_indices, [1])
            morph_features.append(morph_vector)
          yield_dict[feature_name] = morph_features
        if feature_name == "dep_labels":
          sentence.token[0].label = "TOP" # workaround key errors
          yield_dict[feature_name] = self.numericalize(
            values=[token.label for token in sentence.token],
            mapping=feature_mappings["dep_labels"])
        if feature_name == "heads":
          yield_dict[feature_name] = [
            token.selected_head.address for token in sentence.token]
      yield yield_dict

    
if __name__ == "__main__":
  
  # Load word embeddings
  embeddings = nn_utils.load_embeddings()
  word_embeddings = Embeddings(name="word2vec", matrix=embeddings)
  
  # Initialize the preprocessor.
  prep = Preprocessor(word_embeddings=word_embeddings,
                      features=["words", "pos", "morph", "dep_labels",  "heads"],
                      labels=["dep_labels",  "heads"])
  datapath = "data/UDv23/Turkish/training/treebank_train_0_3.pbtxt"

  dataset = prep.make_dataset_from_generator(path=datapath, batch_size=10)
  
  for batch in dataset:
    print(batch["tokens"])
    print(batch["words"])
    print(batch["pos"])
    print(batch["sent_id"])
    print(batch["morph"])
    print(batch["dep_labels"])
    print(batch["heads"])
  
  '''
  # Make a dataset and save it 
  trb = reader.ReadTreebankTextProto(datapath)
  tf_examples = prep.make_tf_examples(from_sentences=trb.sentence)
  prep.write_tf_records(examples=tf_examples,
                        path="./input/treebank_train_0_3.tfrecords")
  

  # Read dataset from saved tfrecords
  dataset = prep.read_dataset_from_tfrecords(
    records="./input/treebank_train_0_3.tfrecords")
  for batch in dataset:
    print(batch)
  '''