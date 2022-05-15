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
        labels=["heads", "dep_labels"],
        head_padding_value=0)
      
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
    expected_dep_labels = tf.constant(
      [[42, 39, 31, 32, 17, 38,  0,  0,  0,  0,  0,  0,  0,  0,  0],
       [42, 35,  1,  5, 31, 30, 31, 30, 31, 30,  9, 30, 39, 14, 38],
       [42, 35,  9, 32,  3, 39, 38,  0,  0,  0,  0,  0,  0,  0,  0],
       [42, 32, 38, 34, 35, 15,  9,  3, 39, 38,  0,  0,  0,  0,  0]],
       dtype=tf.int64)
    expected_pos = tf.constant(
      [[31,  6, 15, 15, 32, 25,  0,  0,  0,  0,  0,  0,  0,  0,  0],
       [31, 15, 32,  2, 15, 15, 15, 15, 15, 15, 29, 15, 15, 32, 25],
       [31, 15, 29, 32,  3,  2, 25,  0,  0,  0,  0,  0,  0,  0,  0],
       [31, 24, 25, 15, 15, 32, 21,  3, 32, 25,  0,  0,  0,  0,  0]],
       dtype=tf.int64
    )
    expected_heads = tf.constant(
      [[0, 0, 3, 4, 1, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0],
       [0, 2, 4, 4, 5, 12, 7, 12, 9, 11, 9, 12, 0, 12, 12],
       [0, 3, 1, 5, 5, 0, 5, 0, 0, 0, 0, 0, 0, 0, 0],
       [0, 8, 8, 4, 8,  4,  4,  8,  0, 8, 0, 0, 0, 0, 0]]
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

  def test_equality_of_generation_methods(self):
    proto = "data/testdata/preprocessor/tr_boun-ud-train-random1.pbtxt"
    tf_records = "data/testdata/preprocessor/tr_boun-ud-train-random1.tfrecords"
    sentences = self.prep.prepare_sentence_protos(path=proto)
    dataset_from_proto = self.prep.make_dataset_from_generator(sentences=sentences, batch_size=1)

    dataset_from_rio = self.prep.read_dataset_from_tfrecords(records=tf_records)
    for batch in dataset_from_proto:
      words = batch["words"]
      pos = batch["pos"]
      labels = batch["dep_labels"]
      morph = batch["morph"]
      heads = batch["heads"]

    for batch in dataset_from_rio:
      tf.debugging.assert_equal(words, batch["words"])
      tf.debugging.assert_equal(pos, batch["pos"])
      tf.debugging.assert_equal(labels, batch["dep_labels"])
      tf.debugging.assert_equal(morph, batch["morph"])
      tf.debugging.assert_equal(heads, batch["heads"])
    print("Passed!")

if __name__ == "__main__":
	tf.test.main()