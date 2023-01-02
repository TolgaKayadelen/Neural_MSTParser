"""Testing parse method."""

import os
import unittest
import numpy as np
import tensorflow as tf

from input import embeddor
from input import preprocessor

from parser.nn import label_first_parser as lfp
from parser.nn import biaffine_parser as bfp
from parser_main.nn_deprecated import parse

from util import reader
from util.nn import nn_utils


_TEST_DATA_DIR = "data/testdata/parser_main/nn/parse"

class ParseTest(unittest.TestCase):
  
  def setUp(self):
    """Set up the parser and treebank."""
    embeddings = nn_utils.load_embeddings()
    word_embeddings = embeddor.Embeddings(name= "word2vec", matrix=embeddings)
    self.prep = preprocessor.Preprocessor(
      word_embeddings=word_embeddings,
      features=["words", "pos", "morph", "dep_labels", "heads"],
      labels=["heads", "dep_labels"])
    self.label_feature = next((f for f in self.prep.sequence_features if f.name == "dep_labels"),
                          None)
    self.dataset = self.prep.make_dataset_from_generator(
      path=os.path.join(_TEST_DATA_DIR, "treebank_1.pbtxt"),
      batch_size=1)
    
  
  def test_parse_label_first_parser(self):
    print("Running parse with label first parser test..")
    parser = lfp.LabelFirstMSTParser(
                            word_embeddings=self.prep.word_embeddings,
                            n_output_classes=self.label_feature.n_values,
                            predict=["edges", "labels"], 
                            model_name="label_first")
    parser.load(name="label_first", path=os.path.join(_TEST_DATA_DIR,
                "label_first"))
    
    
    
    for example in self.dataset:
      tokens = example["tokens"].numpy().tolist()
      edge_scores, label_scores = parser.parse(example)
      edge_preds = tf.argmax(edge_scores, axis=2)
      label_preds = tf.argmax(label_scores, axis=2)

      # input("press to cont..")
      expected_heads = tf.constant([0, 0, 3, 4, 1, 4], dtype=tf.int64)
      expected_labels = tf.constant([34, 33, 24, 27, 33, 32], dtype=tf.int64)
      tf.debugging.assert_equal(expected_heads, edge_preds)
      tf.debugging.assert_equal(expected_labels, label_preds)
      print("Passed!")
  
  
  def test_parse_with_biaffine_parser(self):
    print("Running parse with biaffine parser test..")
    parser = bfp.BiaffineMSTParser(
                            word_embeddings=self.prep.word_embeddings,
                            n_output_classes=self.label_feature.n_values,
                            predict=["edges", "labels"], 
                            model_name="biaffine")
    parser.load(name="biaffine", path=os.path.join(_TEST_DATA_DIR,
                "biaffine"))
    
    
    
    for example in self.dataset:
      tokens = example["tokens"].numpy().tolist()
      edge_scores, label_scores = parser.parse(example)
      label_scores = tf.transpose(label_scores, perm=[0,2,3,1])
      arc_maps = np.array(parse._arc_maps(example["heads"]))
      logits = tf.gather_nd(label_scores, indices=arc_maps)
      
      edge_preds = tf.argmax(edge_scores, axis=2).numpy().tolist()[0]
      label_preds = tf.argmax(logits, axis=1)
      
      expected_heads = tf.constant([0, 0, 3, 4, 1, 4], dtype=tf.int32)
      expected_labels = tf.constant([34, 33, 24, 25, 14, 32], dtype=tf.int64)
      tf.debugging.assert_equal(expected_heads, edge_preds)
      tf.debugging.assert_equal(expected_labels, label_preds)
      print("Passed!")


if __name__ == "__main__":
  unittest.main()
