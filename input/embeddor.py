"""A utility module to set up data and methods from a pretrained embeddings.

The Embeddings class defines a set of data structures and utility functions
to interact with word embeddings set up from a pretrained embedding matrix. 
Can work with both word2vec and GloVe embeddings.
"""
import numpy as np
import random
from enum import Enum
from typing import List, Dict

import logging
logging.basicConfig(format='%(levelname)s : %(message)s', level=logging.INFO)

Array = np.ndarray

class SanityCheck(Enum):
  UNKNOWN = 1
  PASS = 2
  FAIL = 3

class Embeddings:
  """Class to initialize embeddings from a pretrained embedding matrix."""
  def __init__(self, *, name:str, matrix, type="word2vec"):
    self.name = name
    self.type = type
    self.vocab = matrix
    self.vocab_size = len(self.vocab)
    self.embedding_dim = matrix
    self.token_to_index = self._token_to_index()
    self.index_to_token = self._index_to_token()
    self.token_to_vector = matrix
    self.index_to_vector = self._index_to_vector(matrix=matrix)
  
  def __str__(self):
    return f"""
      {self.name} embeddings, embedding dimension {self.embedding_dim},
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
        # index_to_vector[idx, :] = np.random.randn(self.embedding_dim,)
        # initialize as ones
        index_to_vector[idx, :] = np.ones(shape=(self.embedding_dim,))
      else:
        if self.type == "word2vec":
          index_to_vector[idx, :] = matrix[token]
        else:
          index_to_vector[idx, :] = np.array(matrix[token], dtype=np.float32)
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
    if self.type == "word2vec":
      vocab.extend(sorted(set(matrix.wv.vocab.keys())))
    else:
      vocab.extend(matrix.keys())
    self._vocab = vocab
  
  @property
  def embedding_dim(self):
    return self._embedding_dim
  
  @embedding_dim.setter
  def embedding_dim(self, matrix):
    print("Setting embedding dimension")
    random_token = random.choice(self.vocab[2:])
    if self.type == "word2vec":
      self._embedding_dim = matrix[random_token].shape[0]
    else:
      self._embedding_dim = len(matrix[random_token])
  
  @property
  def sanity_check(self) -> SanityCheck:
    print("Running sanity checks on the created embeddings")
    random_token = random.choice(self.vocab[2:])
    random_tok_idx = self.stoi(token=random_token)
    print(type(self.index_to_vector[random_tok_idx]))
    if not type(self.index_to_vector[random_tok_idx]) == np.ndarray:
      return SanityCheck.FAIL
    if self.type == "word2vec":
      if not len(self.token_to_vector.wv.vocab.keys()) == self.vocab_size - 2:
        return SanityCheck.FAIL
    else:
      if not len(self.token_to_vector.keys()) == self.vocab_size - 2:
        print("failed 1")
        return SanityCheck.FAIL
    if not self.vocab[0] == "-pad-" and self.vocab[1] == "-oov-":
      print("failed 2")
      return SanityCheck.FAIL
    if not self.stoi(token=random_token) == self.token_to_index[random_token]:
      print("failed 3")
      return SanityCheck.FAIL
    if not self.itos(idx=random_tok_idx) == self.index_to_token[random_tok_idx]:
      print("failed 4")
      return SanityCheck.FAIL
    if self.type == "word2vec":
      if not (self.token_to_vector[random_token] == self.index_to_vector[
        random_tok_idx]).all():
        print("failed 5")
        return SanityCheck.FAIL
      else:
        if not (np.array(self.token_to_vector[random_token], dtype=np.float32) == self.index_to_vector[
          random_tok_idx]).all():
          print("failed 5")
          return SanityCheck.FAIL
    if not (self.itov(idx=random_tok_idx) == self.index_to_vector[
      random_tok_idx]).all():
      print("failed 6")
      return SanityCheck.FAIL
    if self.type == "word2vec":
      if not (self.token_to_vector[self.itos(idx=random_tok_idx)] == self.token_to_vector[random_token]).all():
        print("failed 7")
        return SanityCheck.FAIL
    else:
      elem1 = np.array(self.token_to_vector[self.itos(idx=random_tok_idx)], dtype=np.float32)
      elem2 = np.array(self.token_to_vector[random_token], dtype=np.float32)
      if not (elem1 == elem2).all():
        print("failed 7")
        return SanityCheck.FAIL
    if self.type == "word2vec":
      if not (self.index_to_vector[self.stoi(token=random_token)] == self.token_to_vector[random_token]).all():
        print("failed 8")
        return SanityCheck.FAIL
    else:
      np_t_to_v_random_token = np.array(self.token_to_vector[random_token], dtype=np.float32)
      if not np.allclose(self.index_to_vector[self.stoi(token=random_token)], np_t_to_v_random_token):
        print("failed 8")
        return SanityCheck.FAIL
    return SanityCheck.PASS