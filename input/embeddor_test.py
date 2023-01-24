# -*- coding: utf-8 -*-

"""Tests for the preprocessor."""
import pickle
import unittest
from input import embeddor
from util.nn import nn_utils
from parser.utils import load_models

import logging
logging.basicConfig(format='%(levelname)s : %(message)s', level=logging.INFO)

class EmbeddorTest(unittest.TestCase):
  """Tests for the embeddor."""
  
  def setUp(self):
    embeddings = nn_utils.load_embeddings()
    self.word_embeddings = embeddor.Embeddings(name="embeddings", matrix=embeddings)
    # embeddings = load_models.load_pickled_embeddings("de")
    # self.word_embeddings = embeddor.Embeddings(name="embeddings", matrix=embeddings, type="fasttext")


  def test_sanity_check(self):
    print("Running test_sanity_check..")
    #print(self.word_embeddings.embedding_dim)
    #input()
    #print(self.word_embeddings.token_to_index)
    #input()
    #print(self.word_embeddings.index_to_token)
    #input()
    #print(self.word_embeddings.index_to_vector)
    #input()
    #print(self.word_embeddings.token_to_vector)
    #input()
    self.assertTrue(self.word_embeddings.sanity_check == embeddor.SanityCheck.PASS)
    print(self.word_embeddings.vocab_size)
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