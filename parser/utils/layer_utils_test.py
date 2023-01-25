# -*- coding: utf-8 -*-

"""Tests for the layer_utils module."""

import logging

import tensorflow as tf

from parser.utils import layer_utils
from input.embeddor import Embeddings
from util.nn import nn_utils

logging.basicConfig(format='%(levelname)s : %(message)s', level=logging.INFO)

class LayerUtilsTest(tf.test.TestCase):
  """Tests for the preprocessor."""
  def setUp(self):
    self.embeddings = nn_utils.load_embeddings()

  def test_word_embeddings(self):
    word_embeddings = Embeddings(name="word2vec", matrix=self.embeddings)
    # print(word_embeddings.sanity_check)
    embedding_layer = layer_utils.EmbeddingLayer(pretrained=word_embeddings,
                                                 name="word_embeddings",
                                                 trainable=False)
    sentence = []
    for item in ["Selam", "tolga", "dedi"]:
      sentence.append(word_embeddings.stoi(token=item))
    logging.info(tf.constant(sentence))
    word_features=embedding_layer(tf.constant(sentence))
    # logging.info(word_features)
    self.assertListEqual(sentence, [173437, 493047, 274133])
    self.assertAlmostEqual(word_features[0][0].numpy(),  0.06717049)
    self.assertAlmostEqual(word_features[1][0].numpy(),  0.29322195)
    self.assertAlmostEqual(word_features[2][0].numpy(),  -2.75858951e+00)
    print("Passed")

  
  def test_pos_embeddings(self):
    features = layer_utils.EmbeddingLayer(input_dim=35, output_dim=32,
                                          name="pos_embeddings",
                                          trainable=True)(tf.constant([1,2,3]))
    self.assertEqual(features.shape, (3, 32))

if __name__ == "__main__":
  tf.test.main()