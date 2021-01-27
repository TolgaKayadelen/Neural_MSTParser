import tensorflow as tf
import numpy as np
import random
from enum import Enum
from tagset.reader import LabelReader
from typing import List, Dict, Generator, Union
from util.nn import nn_utils
from util import reader
import time

import logging
logging.basicConfig(format='%(levelname)s : %(message)s', level=logging.INFO)

Array = np.ndarray
Dataset = tf.data.Dataset
SequenceExample = tf.train.SequenceExample

class SanityCheck(Enum):
  UNKNOWN = 1
  PASS = 2
  FAIL = 3

class SequenceFeature:
  """A sequence feature is a map from a feature name to a list of int values."""
  def __init__(self, name: str, values: List[int]=[], dtype=tf.int64,
               one_hot: bool = False, n_values: int = None,
               is_label: bool = False):
    self.name = name
    self.values = values
    self.dtype = dtype
    self.n_values = n_values
    self.one_hot = one_hot
    self.is_label = is_label
  
  def _convert_to_one_hot(self, values, n_values):
    """Converts an integer array one hot vector.
    
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
            n_values: {self.n_values}"""


class Embeddings:
  """Class to initialize embeddings from a pretrained embedding matrix."""
  def __init__(self, *, name:str, matrix):
    self.name = name
    self.vocab = matrix
    self.vocab_size = len(self.vocab)
    self.embedding_dim = matrix
    self.token_to_index = self._token_to_index()
    self.index_to_token = self._index_to_token()
    self.token_to_vector = matrix
    self.index_to_vector = self._index_to_vector(matrix=matrix)
  
  def __str__(self):
    return f"""
      {self.name} embeddings, dimension: {self.dimension},
      vocab_size: {self.vocab_size}
      """

  def _token_to_index(self) -> Dict[int, int]:
    return {elem:ind for ind, elem in enumerate(self.vocab)}
  
  def _index_to_token(self) -> Dict[int, int]:
    return {ind:elem for ind, elem in enumerate(self.vocab)}       

  def _index_to_vector(self, *, matrix) -> Array:
    index_to_vector = np.zeros(shape=(self.vocab_size, self.embedding_dim))
    for token, idx in self.token_to_index.items():
      # print(token, idx)
      if token == "-pad-":
        index_to_vector[idx, :] = np.zeros(shape=(self.embedding_dim,))
      elif token == "-oov-":
        # randomly initialize
        index_to_vector[idx, :] = np.random.randn(self.embedding_dim,)
      else:
        index_to_vector[idx, :] = matrix[token]
        # print(matrix[token])
    return index_to_vector
  
  # Accessor methods for individual items are defined here.
  def stoi(self, *, token: str) -> int:
    return self.token_to_index[token]
  
  def itos(self, *, idx: int) -> str:
    return self.index_to_token[idx]
  
  def ttov(self, *, token: str):
    return self.token_to_vector[token]
  
  def itov(self, *, idx: int):
    return self.index_to_vector[idx]
  
  # Embedding properties are set based on the properties of the 
  # word2vec matrix.
  @property
  def vocab(self):
    return self._vocab
  
  @vocab.setter
  def vocab(self, matrix) -> List[str]:
    "Setting up the vocabulary."
    print("Setting up the vocabulary")
    vocab = []
    vocab.extend(["-pad-", "-oov-"])
    vocab.extend(sorted(set(matrix.wv.vocab.keys())))
    self._vocab = vocab
  
  @property
  def embedding_dim(self):
    return self._embedding_dim
  
  @embedding_dim.setter
  def embedding_dim(self, matrix):
    print("Setting embedding dimension")
    random_token = random.choice(self.vocab)
    self._embedding_dim = matrix[random_token].shape[0]
  
  @property
  def sanity_check(self) -> SanityCheck:
    print("Running sanity checks on the created embeddings")
    random_token = random.choice(self.vocab)
    random_tok_idx = self.stoi(token=random_token)
    if not len(self.token_to_vector.wv.vocab.keys()) == self.vocab_size - 2:
      return SanityCheck.FAIL
    if not self.vocab[0] == "-pad-" and self.vocab[1] == "-oov-":
      return SanityCheck.FAIL
    if not self.stoi(token=random_token) == self.token_to_index[random_token]:
      return SanityCheck.FAIL
    if not self.itos(idx=random_tok_idx) == self.index_to_token[random_tok_idx]:
      return SanityCheck.FAIL
    if not (self.token_to_vector[random_token] == self.index_to_vector[
      random_tok_idx]).all():
      return SanityCheck.FAIL
    if not (self.itov(idx=random_tok_idx) == self.index_to_vector[
      random_tok_idx]).all():
      return SanityCheck.FAIL
    if not (self.token_to_vector[self.itos(idx=random_tok_idx)
      ] == self.token_to_vector[random_token]).all():
      return SanityCheck.FAIL
    if not (self.index_to_vector[self.stoi(token=random_token)
      ] == self.token_to_vector[random_token]).all():
      return SanityCheck.FAIL
    return SanityCheck.PASS


class Preprocessor:
  """Prepares data as batched tfrecords."""
  def __init__(self, *, word_embeddings: Embeddings = None):
    if word_embeddings:
      self.word_embeddings = word_embeddings
  
  def numericalize(self, *, values: List[str],
                   mapping: Union[Dict, Embeddings]) -> List[int]:
    """Returns numeric values (integers) for features based on a mapping."""
    if isinstance(mapping, Embeddings):
      indices = []
      for value in values:
        try:
          indices.append(mapping.stoi(token=value))
        except KeyError:
          indices.append(mapping.stoi(token="-oov-"))
      return indices
    elif isinstance(mapping, dict):
      return [mapping[value] for value in values]
    else:
      raise ValueError("mapping should be a dict or an Embedding instance.")
  
  def get_index_mappings_for_features(self, feature_names: List[str]) -> Dict:
    "Returns a string:index map for a feature based on predefined configs"
    mappings = {}
    if "words" in feature_names:
      if self.word_embeddings:
        mappings["words"] = self.word_embeddings
      else:
        embeddings = nn_utils.load_embeddings()
        mappings["words"] = Embeddings(name="word2vec", matrix=embeddings)
    if "morph" in feature_names:
      raise RuntimeError("Morphology features are not supported yet.")
    if "pos" in feature_names:
      mappings["pos"] = LabelReader.get_labels("pos").labels
    if "category" in feature_names:
      mappings["category"] = LabelReader.get_labels("category").labels
    if "srl" in feature_names:
      raise RuntimeError("Semantic role features are not supported yet.")
    return mappings
        
  def make_tf_example(self, *, features: List[SequenceFeature]
                     ) -> SequenceExample:
    """Creates a tf.train.SequenceExample from a single datapoint.
    
    A single datapoint is represented as a List of SequenceFeature objects.
    """
    example = tf.train.SequenceExample()
    
    for feature in features:
      feature_list = example.feature_lists.feature_list[feature.name]
      for token in feature.values:
        if not feature.is_label:
          feature_list.feature.add().int64_list.value.append(token)
        # If the feature is a label feature, here we do a look up and 
        # find the one hot encoding of the label value in question.
        else:
          label = np.array(feature.one_hot[np.where(
                           feature.one_hot[:, token] == 1)][0])
          label = list([int(l) for l in label])
          feature_list.feature.add().int64_list.value.extend(label)
    print(example)
    input("press to cont.")
    return example
    
  def write_tf_records(self, *, examples: List[SequenceExample], path: str
                      ) -> None:
    """Serializes tf.train.SequenceExamples to tfrecord files."""
    filename = path + ".tfrecords" if not path.endswith(".tfrecords") else path
    writer = tf.io.TFRecordWriter(filename)
    for example in examples:
      writer.write(example.SerializeToString())
    writer.close()

  
  def make_dataset_from_treebank(self, *, features:List[str], label=str,
                                 treebank,
                                 save_path: str = "./input/test11.tfrecords"):
    """Saves tfrecords dataset from a treebank based on features."""
    tf_examples = []
    feature_mappings = self.get_index_mappings_for_features(features)
    label_mappings = self.get_index_mappings_for_features([label])
    sequence_features = {}
    for feature in features:
      sequence_features[feature] = SequenceFeature(name=feature)
    label_dict = LabelReader.get_labels(label).labels
    label_feature = SequenceFeature(name=label,
                                    values=list(label_dict.values()),
                                    n_values=len(list(label_dict)),
                                    is_label=True, one_hot=True)
    print(label_feature)
    sequence_features[label] = label_feature
    for sentence in treebank.sentence:
      if "words" in features:
        word_indices = self.numericalize(
                          values=[token.word for token in sentence.token],
                          mapping=feature_mappings["words"]
                        )
        print([token.word for token in sentence.token])
        sequence_features["words"].values = word_indices
      if "category" in features:
        cat_indices = prep.numericalize(
                          values=[token.category for token in sentence.token],
                          mapping=feature_mappings["category"])
        sequence_features["category"].values = cat_indices
      if "pos" in features:
        pos_indices = prep.numericalize(
                          values=[token.pos for token in sentence.token],
                          mapping=feature_mappings["pos"])
        sequence_features["pos"].values = pos_indices
      
      label_indices = prep.numericalize(
                          values=[token.pos for token in sentence.token],
                          mapping=label_mappings["pos"])
      print(label_indices)
      print(print([token.pos for token in sentence.token]))
      sequence_features[label].values = label_indices
      example = self.make_tf_example(features=list(sequence_features.values()))
      tf_examples.append(example)
    self.write_tf_records(examples=tf_examples, path=save_path)
      

  def read_dataset_from_tfrecords(self, *,
                                  batch_size: int = 4,
                                  features: List[SequenceFeature],
                                  records: str) -> Dataset:
    """Reads a tensorflow dataset from a saved tfrecords path."""
    
    _sequence_features = {}
    _dataset_shapes = {}
    
    # Create a dictionary description of the tensors to parse the features.
    for feature in features:
      if not feature.is_label:
        _sequence_features[feature.name]=tf.io.FixedLenSequenceFeature([],
                           dtype=feature.dtype)
        _dataset_shapes[feature.name]=tf.TensorShape([None])
      else:
        _sequence_features[feature.name]=tf.io.FixedLenSequenceFeature(
                           feature.n_values,
                           dtype=feature.dtype)
        _dataset_shapes[feature.name]=tf.TensorShape((None, feature.n_values))
    
    def _parse_tf_records(record):
      """Returns a dictionary of tensors."""
      _, parsed_example = tf.io.parse_single_sequence_example(
        serialized=record,
        sequence_features=_sequence_features
      )
      return {feature.name:parsed_example[feature.name] for feature in features}
    
    dataset = tf.data.TFRecordDataset([records])
    dataset = dataset.map(_parse_tf_records)    
    dataset = dataset.padded_batch(batch_size, padded_shapes=_dataset_shapes)
    dataset = dataset.shuffle(buffer_size=20)
    print(dataset)
    return dataset
    
  def make_dataset_from_generator(self, *, path: str, batch_size: int = 2,
                                  features=List[SequenceFeature],
                                  generator: Generator=None) -> Dataset:
    """Makes a tensorflow dataset that is shuffled, batched and padded.
    Args:
      path: path to the dataset treebank. 
      generator: a generator function.
                 If not specified, the class generator is used.
    """
    if not generator:
      generator = lambda: self._example_generator(path, features)
    
    _output_types={}
    _output_shapes={}
    _padded_shapes={}
    
    for feature in features:
      _output_types[feature.name]=feature.dtype
      
      if feature.is_label:
        _output_shapes[feature.name]=(None, feature.n_values)
        _padded_shapes[feature.name]=tf.TensorShape((None, feature.n_values))
      else:
        _output_shapes[feature.name]=[None]
        _padded_shapes[feature.name]=tf.TensorShape([None])

    
    dataset = tf.data.Dataset.from_generator(
      generator,
      output_types=_output_types,
      output_shapes=_output_shapes
    )

    dataset = dataset.padded_batch(batch_size, padded_shapes=_padded_shapes)
    dataset = dataset.shuffle(buffer_size=20)
    print(dataset)
    return dataset
  

  def _example_generator(self, path: str, features:List[SequenceFeature]):
    trb = reader.ReadTreebankTextProto(path)
    sentences = trb.sentence
    feature_names = [feature.name for feature in features]
    feature_mappings = self.get_index_mappings_for_features(feature_names)
    label_feature = next((f for f in features if f.is_label), None)
    yield_dict = {}
    
    for sentence in sentences:
      for feature_name in feature_names:
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
      # print(yield_dict["pos"])
      for k in yield_dict:
        if k == label_feature.name:
          labels = np.array([label_feature.one_hot[np.where(
            label_feature.one_hot[:, item] == 1)][0] for item in yield_dict[k]])
          yield_dict[k] = labels
          break
      # print(yield_dict["pos"])
      yield yield_dict

    
if __name__ == "__main__":
  
  # Load word embeddings
  embeddings = nn_utils.load_embeddings()
  word_embeddings = Embeddings(name="word2vec", matrix=embeddings)
  
  # Initialize the preprocessor.
  prep = Preprocessor(word_embeddings=word_embeddings)
  
  # Make a dataset and save it
  datapath = "data/UDv23/Turkish/training/treebank_0_3.pbtxt"
  trb = reader.ReadTreebankTextProto(datapath)
  prep.make_dataset_from_treebank(features=["words", "category"], label="pos",
                                  treebank=trb)
  
  # Read dataset from saved tfrecords
  features = [word_feature, pos_feature, cat_feature]
  dataset = prep.read_dataset_from_tfrecords(features=features,
                                             records="./input/test1.tfrecords")
  # for batch in dataset:
  #  print(batch)
