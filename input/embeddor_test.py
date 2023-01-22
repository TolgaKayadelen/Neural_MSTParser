# -*- coding: utf-8 -*-

"""Tests for the preprocessor."""
import pickle

import unittest
from input import embeddor
from util.nn import nn_utils

import logging
logging.basicConfig(format='%(levelname)s : %(message)s', level=logging.INFO)

class EmbeddorTest(unittest.TestCase):
  """Tests for the embeddor."""
  
  def setUp(self):
    embeddings = nn_utils.load_embeddings()
    self.word_embeddings = embeddor.Embeddings(name="word2vec", matrix=embeddings)
    
  def test_sanity_check(self):
    print("Running test_sanity_check..")
    self.assertTrue(self.word_embeddings.sanity_check == embeddor.SanityCheck.PASS)
    print("Passed!")
    
  def test_embedding_indices(self):
    print("Running test_embedding indices..")
    self.assertTrue(self.word_embeddings.stoi(token="Kerem") == 119951)
    print(self.word_embeddings.itos(idx=1))
    print("Passed!")

  def test_loading_indexes(self):
    print("Running test loading indices..")
    with open('./input/token_to_index_dictionary.pkl', 'rb') as f:
      loaded_dict = pickle.load(f)
      self.assertTrue(loaded_dict["Kerem"] == 119951)
      print(loaded_dict["1936"]) #11674
      print(loaded_dict["Adeta"]) #30510
    print("Passed!")


if __name__ == "__main__":
  unittest.main()