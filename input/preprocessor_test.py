# -*- coding: utf-8 -*-

"""Tests for the preprocessor."""

import os
import tensorflow as tf
from input import preprocessor
from input.embeddor import Embeddings
from google.protobuf import text_format
from util import reader
from util.nn import nn_utils
from tagset.morphology import morph_tags
from tagset.reader import LabelReader

import logging
logging.basicConfig(format='%(levelname)s : %(message)s', level=logging.INFO)


_TESTDATA_DIR = "data/testdata"
_PREPROCESSOR_DIR = os.path.join(_TESTDATA_DIR, "preprocessor")

embeddings = nn_utils.load_embeddings()
word_embeddings = Embeddings(name= "word2vec", matrix=embeddings)

class PreprocessorTest(tf.test.TestCase):
  """Tests for the preprocessor."""
  def setUp(self):
    self.datapath = os.path.join(_PREPROCESSOR_DIR, "treebank_train_0_3.pbtxt")
    self.prep = preprocessor.Preprocessor(
        word_embeddings=word_embeddings,
        features=["words", "pos", "morph", "heads", "dep_labels"],
        labels=["heads", "dep_labels"])
      
  def test_sequence_features(self):
    print("Running test_sequence_features..")
    self.assertEqual(
      [feature.name for _, feature in self.prep.sequence_features_dict.items()],
      ["words", "pos", "morph", "heads", "dep_labels", "tokens", "sent_id"]
    )
    print("Passed!")
  
  def test_make_dataset_from_generator(self):
    print("Running test make_dataset_from_generator..")
    sentences = self.prep.prepare_sentence_protos(path=self.datapath)
    dataset = self.prep.make_dataset_from_generator(sentences=sentences)
    expected_words = tf.constant([
       [1, 34756, 224906, 578174, 506596, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
       [1, 570228, 548366, 507341, 361412, 220841, 474903, 319297, 334201,
        258062, 380947, 396760, 559275, 508964, 1],
       [1, 152339, 380947, 484432, 340375, 536702, 1, 0, 0, 0, 0, 0, 0, 0, 0],
       [1, 119951,1, 562326, 490947, 305574, 359585, 453123, 444258,1, 0, 0, 0,
        0, 0]], dtype=tf.int64)
    expected_dep_labels = tf.constant([
        [34, 33, 24, 25, 14, 32,  0,  0,  0,  0,  0,  0,  0,  0,  0],
        [34, 28,  1,  5, 24, 23, 24, 23, 24, 23,  8, 23, 33, 11, 32],
        [34, 28,  8, 25,  3, 33, 32,  0,  0,  0,  0,  0,  0,  0,  0],
        [34, 25, 32, 27, 28, 12,  8,  3, 33, 32,  0,  0,  0,  0,  0]], 
       dtype=tf.int64)
    expected_pos = tf.constant(
      [[30,  5, 14, 14, 31, 24,  0,  0,  0,  0,  0,  0,  0,  0,  0],
       [30, 14, 31,  3, 14, 14, 14, 14, 14, 14, 28, 14, 14, 31, 24],
       [30, 14, 28, 31,  4,  3, 24,  0,  0,  0,  0,  0,  0,  0,  0],
       [30, 23, 24, 14, 14, 31, 20,  4, 31, 24,  0,  0,  0,  0,  0]],
       dtype=tf.int64
    )
    expected_heads = tf.constant(
      [[0, 0, 3, 4, 1, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1],
       [0, 2, 4, 4, 5, 12, 7, 12, 9, 11, 9, 12, 0, 12, 12],
       [0, 3, 1, 5, 5, 0, 5, -1, -1, -1, -1, -1, -1, -1, -1],
       [0, 8, 8, 4, 8,  4,  4,  8,  0, 8, -1, -1, -1, -1, -1]]
    )
    for batch in dataset:
      self.assertAllEqual(expected_words, batch["words"])
      self.assertAllEqual(expected_dep_labels, batch["dep_labels"])
      self.assertAllEqual(expected_pos, batch["pos"])
      self.assertAllEqual(expected_heads, batch["heads"])
    print("Passed!")
  
  def test_morph_features(self):
    print("Running test morph_features..")
    treebank = reader.ReadTreebankTextProto(self.datapath)
    sentences = treebank.sentence
    dataset = self.prep.make_dataset_from_generator(sentences=sentences)
    morph_mapping = LabelReader.get_labels("morph").labels
    # TODO: do this with itertools.
    for batch in dataset:
      for sentence_repr, sentence in zip(batch["morph"], sentences):
        for i in range(sentence_repr.shape[0]):
          # if there are any morphology features. we understand that from
          # having some 1's in the array.
          if tf.math.count_nonzero(sentence_repr[i]):
            morph_indexes = tf.where(sentence_repr[i] == 1)
            morph_values = [val.numpy()[0] for val in morph_indexes]
            tags = morph_tags.from_token(token=sentence.token[i])
            self.assertListEqual(
              sorted(morph_values),
              sorted([morph_mapping[tag] for tag in tags]))
    print("Passed!")

if __name__ == "__main__":
	tf.test.main()