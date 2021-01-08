import tensorflow as tf
import numpy as np
import random
from enum import Enum
from typing import List, Dict, Generator, Union
from util.nn import nn_utils
from util import reader, writer
from google.protobuf import text_format

Array = np.ndarray
Dataset = tf.data.Dataset
SequenceExample = tf.train.SequenceExample

class SanityCheck(Enum):
  UNKNOWN = 1
  PASS = 2
  FAIL = 3

class SequenceFeature:
  """A sequence feature is a map from a feature name to a list of int values."""
  def __init__(self, name: str, values: List[int]):
    self.name = name
    self.values = values
  
  def __str__(self):
    return f"{self.name}: {self.values}"

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
      {self.name} embeddings, dimension: {self.dimension}, vocab_size: {self.vocab_size}
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
    if not (self.token_to_vector[random_token] == self.index_to_vector[random_tok_idx]).all():
      return SanityCheck.FAIL
    if not (self.itov(idx=random_tok_idx) == self.index_to_vector[random_tok_idx]).all():
      return SanityCheck.FAIL
    if not (self.token_to_vector[self.itos(idx=random_tok_idx)] == self.token_to_vector[
      random_token]).all():
      return SanityCheck.FAIL
    if not (self.index_to_vector[self.stoi(token=random_token)] == self.token_to_vector[
      random_token]).all():
      return SanityCheck.FAIL
    return SanityCheck.PASS

class Preprocessor:
  """Prepares data as batched tfrecords."""
  def __init__(self):
    pass
  
  def numericalize(self, *, values: List[str],
                   mapping: Union[Dict, Embeddings]) -> List[int]:
    """Returns numeric values (integers) for features based on a string:index mapping."""
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
      raise ValueError("mapping should be either a dict or an Embedding instance.")
  
  def make_tf_example(self, *, features: List[SequenceFeature]) -> SequenceExample:
    """Creates a tf.train.SequenceExample from a single datapoint.
    
    A single datapoint is represented as a List of SequenceFeature objects.
    """
    example = tf.train.SequenceExample()
    
    for feature in features:
      feature_list = example.feature_lists.feature_list[feature.name]
      for token in feature.values:
        feature_list.feature.add().int64_list.value.append(token)
    
    return example
  
  def write_tf_examples(self, *, examples: List[SequenceExample], path: str):
    """Serializes a set of tf.train.SequenceExamples to path."""
    for example in examples:
      print(type(example))
    writer.write_protolist_as_text(examples, path)
  
  def parse_tf_examples(self, *, path: str) -> List[SequenceExample]:
    """Reads a set of tf.train.SequenceExample from path."""
    import codecs
    with codecs.open(path, encoding="utf-8") as in_file:
      read = in_file.read().strip()
      print(text_format.Parse(read, tf.train.SequenceExample()))
  
  def write_tf_records(self, *, examples: List[SequenceExample], path: str) -> None:
    """Serializes tf.train.SequenceExamples to tfrecord files."""
    pass
  
  def parse_tf_records(self, *, path:str) -> List[SequenceExample]:
    """Reads a set of tf.train.SequenceExamples from a tf.records file path."""
    
    
  def make_dataset_from_generator(self, *, generator: Generator) -> Dataset:
    """Makes a tensorflow dataset that is shuffled, batched and padded."""
    pass
    
  def make_dataset_from_records(self, *, records: str) -> Dataset:
    """Makes a tensorflow dataset from tfrecords path."""
    pass
  

if __name__ == "__main__":
  # embeddings = nn_utils.load_embeddings()
  # myemb = Embeddings(name="word2vec", matrix=embeddings)
  # print(myemb.name)
  # print(myemb.vocab)
  # print(myemb.vocab_size)
  # print(myemb.embedding_dim)
  # print(myemb.itos(idx=493047))
  # print(myemb.sanity_check)
  sentence1 = ["tolga", "okula", "yyllss"]
  sentence2 = ["tolga", "okula", "gitmedi"]
  preprocessor = Preprocessor()
  # print(preprocessor.numericalize(values=sentence1, mapping=myemb))
  word_indices = preprocessor.numericalize(values=sentence2, mapping={"tolga": 1, "okula": 2, "gitmedi": 3})
  pos_indices = [6,7,8]
  examples = []
  for i in range(5):
    word_indices = [index+1 for index in word_indices]
    words = SequenceFeature(name="words", values=word_indices)
    pos_indices = [index+1 for index in pos_indices]
    postags = SequenceFeature(name="postags", values=pos_indices)
    example = preprocessor.make_tf_example(features=[words, postags])
    examples.append(example)
  # preprocessor.write_tf_examples(examples=examples, path="./input/test.pbtxt")
  preprocessor.parse_tf_examples(path="./input/test.pbtxt")