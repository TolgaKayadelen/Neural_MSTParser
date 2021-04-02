# -*- coding: utf-8 -*-

"""Tests for the preprocessor."""

import unittest
from input import embeddor 
from google.protobuf import text_format
from util.nn import nn_utils

import logging
logging.basicConfig(format='%(levelname)s : %(message)s', level=logging.INFO)

class PreprocessorTest(unittest.TestCase):
  """Tests for the embeddor."""

  def test_sanity_check(self):
    print("Running test_sanity_check..")
    embeddings = nn_utils.load_embeddings()
    word_embeddings = embeddor.Embeddings(name="word2vec", matrix=embeddings)
    self.assertTrue(word_embeddings.sanity_check == embeddor.SanityCheck.PASS)
    print("Passed!")

if __name__ == "__main__":
  unittest.main()