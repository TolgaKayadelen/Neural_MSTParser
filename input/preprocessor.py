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

# TODO: generate this dynamically.
_POS_TAGS = {'ANum': 1, 'Abr': 2, 'Adj': 3, 'Adverb': 4, 'Conj': 5, 'Demons': 6, 'Det': 7,
            'Dup': 8, 'Interj': 9, 'NAdj': 10, 'NNum': 11, 'Neg': 12, 'Ness': 13, 'Noun': 14, 
            'PCAbl': 15, 'PCAcc': 16, 'PCDat': 17, 'PCGen': 18, 'PCIns': 19, 'PCNom': 20, 'Pers': 21, 
            'PostP': 22, 'Prop': 23, 'Punc': 24, 'Quant': 25, 'Ques': 26, 'Reflex': 27, 'Rel': 28,
            'Since': 29, 'TOP': 30, 'Verb': 31, 'With': 32, 'Without': 33, 'Zero': 34, '-pad-': 0
            }

class SanityCheck(Enum):
  UNKNOWN = 1
  PASS = 2
  FAIL = 3

class SequenceFeature:
  """A sequence feature is a map from a feature name to a list of int values."""
  def __init__(self, name: str, values: List[int]=[], dtype=tf.int64):
    self.name = name
    self.values = values
    self.dtype = dtype
  
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
    
  def write_tf_records(self, *, examples: List[SequenceExample], path: str) -> None:
    """Serializes tf.train.SequenceExamples to tfrecord files."""
    filename = path + ".tfrecords" if not path.endswith(".tfrecords") else path
    writer = tf.io.TFRecordWriter(filename)
    for example in examples:
      writer.write(example.SerializeToString())
    writer.close()
    
  def make_dataset_from_tfrecords(self, *,
                                  batch_size=1,
                                  features: List[SequenceFeature],
                                  records: str) -> Dataset:
    """Makes a tensorflow dataset from tfrecords path."""
    
    _sequence_features = {}
    _dataset_shapes = {}
    
    # Create a description of the tensors to parse the features.
    for feature in features:
      _sequence_features[feature.name]=tf.io.FixedLenSequenceFeature([], dtype=feature.dtype)
      _dataset_shapes[feature.name]=tf.TensorShape([None])
    
    def _parse_tf_records(example):
      _, parsed_example = tf.io.parse_single_sequence_example(
        serialized=example,
        sequence_features=_sequence_features
      )
      return {feature.name:parsed_example[feature.name] for feature in features}
    
    dataset = tf.data.TFRecordDataset([records])
    dataset = dataset.map(_parse_tf_records)    
    dataset = dataset.padded_batch(batch_size, padded_shapes=_dataset_shapes)
    return dataset

  def make_dataset_from_generator(self, *, path: str,
                                  features=List[SequenceFeature],
                                  generator: Generator=None) -> Dataset:
    """Makes a tensorflow dataset that is shuffled, batched and padded.
    Args:
      path: path to the dataset treebank. 
      generator: a generator function. if not specified, the class generator is used.
    """
    if not generator:
      generator = lambda: self._example_generator(path, features)
    
    _output_types={}
    _output_shapes={}
    _padded_shapes={}
    for feature in features:
      _output_types[feature.name]=feature.dtype
      _output_shapes[feature.name]=[None]
      _padded_shapes[feature.name]=tf.TensorShape([None])
      
    dataset = tf.data.Dataset.from_generator(
      generator,
      output_types=_output_types,
      output_shapes=_output_shapes
    )
    # dataset = dataset.shuffle(buffer_size=3)
    dataset = dataset.padded_batch(4, padded_shapes=_padded_shapes)
    return dataset
  
  def _example_generator(self, path: str, features:List[SequenceFeature]):
    _words, _morph, _pos, _category = [False] * 4
    trb = reader.ReadTreebankTextProto(path)
    sentences = trb.sentence
    feature_names = [feature.name for feature in features]
    if "words" in feature_names:
      _words = True
      embeddings = nn_utils.load_embeddings()
      word_mapping = Embeddings(name="word2vec", matrix=embeddings)
    if "morph" in feature_names:
      _morph = True
    if "postags" in feature_names:
      _pos = True
      pos_mapping = _POS_TAGS
    if "category" in feature_names:
      _cat = True
    yield_dict = {}
    for sentence in sentences:
      if _words:
        words = self.numericalize(values=[token.word for token in sentence.token],
                                  mapping=word_mapping)
        yield_dict["words"] = words
      if _pos:
        postags = self.numericalize(values=[token.pos for token in sentence.token],
                                    mapping=pos_mapping)
        yield_dict["postags"] = postags
      
      yield yield_dict

if __name__ == "__main__":
  preprocessor = Preprocessor()
  # embeddings = nn_utils.load_embeddings()
  # myemb = Embeddings(name="word2vec", matrix=embeddings)
  # print(myemb.name)
  # print(myemb.vocab)
  # print(myemb.vocab_size)
  # print(myemb.embedding_dim)
  # print(myemb.itos(idx=493047))
  # print(myemb.sanity_check)
  """
  sentence1 = ["tolga", "okula", "yyllss"]
  sentence2 = ["tolga", "okula", "gitmedi"]
  
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
  # preprocessor.write_tf_records(examples=examples, path="./input/test.tfrecords")
  # features = [SequenceFeature(name="words"), SequenceFeature(name="postags")]
  # dataset = preprocessor.make_dataset_from_tfrecords(features=features, records="./input/test.tfrecords")
  """
  dataset = preprocessor.make_dataset_from_generator(
    path="data/UDv23/Turkish/training/treebank_0_3.pbtxt",
    features=[SequenceFeature(name="words"), SequenceFeature(name="postags")])
  for record in dataset:
    print(record)