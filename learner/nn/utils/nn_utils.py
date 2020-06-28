
import os
import gensim

import numpy as np

_EMBEDDING_DIR = "embeddings"


def convert_to_one_hot(indices, n_labels):
  """Converts an integer array of shape (1, m) to a one hot vector of shape (len(indices), n_labels)
  
  Args:
    indices: An integer array representing the labels or classes.
    n_labels: depth of the one hot vector.
  Returns:
    one hot vector of shape (len(indices), n_labels)
  """
  return np.eye(n_labels)[indices.reshape(-1)]
  

def load_embeddings(name="tr-word2vec-model_v3.bin"):
  """Loads a pretrained word embedding model into memory."""
  word2vec_bin = os.path.join(_EMBEDDING_DIR, name)
  word2vec = gensim.models.Word2Vec.load(word2vec_bin)
  
  return word2vec


def maxlen(data):
  """Returns the length of the longest list in a list of lists."""
  return len(max(data, key=len))