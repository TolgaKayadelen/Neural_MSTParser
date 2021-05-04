"""Testing parse method."""

import os
import unittest
import tensorflow as tf

from input import embeddor
from input import preprocessor

from parser.nn import label_first_parser as lfp
from parser_main.nn import parse

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
    label_feature = next((f for f in self.prep.sequence_features if f.name == "dep_labels"),
                          None)
    self.parser = lfp.LabelFirstMSTParser(
                            word_embeddings=self.prep.word_embeddings,
                            n_output_classes=label_feature.n_values,
                            predict=["edges", "labels"], 
                            model_name="test2")
    self.parser.load(name="test2", path=os.path.join(_TEST_DATA_DIR, "test2"))
  
  def test_parse(self):
    print("Running parse test..")
    gold_treebank = reader.ReadTreebankTextProto(os.path.join(
      _TEST_DATA_DIR, "treebank_1.pbtxt"))
    dataset = self.prep.make_dataset_from_generator(
      path=os.path.join(_TEST_DATA_DIR, "treebank_1.pbtxt"),
      batch_size=1)
    
    for example in dataset:
      tokens = example["tokens"].numpy().tolist()
      edge_scores, label_scores = self.parser.parse(example)
      edge_preds = tf.argmax(edge_scores, axis=2)
      label_preds = tf.argmax(label_scores, axis=2)
      print(f"edge_preds: {edge_preds}")
      print(f"label preds {label_preds}")
      # input("press to cont..")
      expected_heads = tf.constant([0, 0, 3, 4, 1, 4], dtype=tf.int64)
      expected_labels = tf.constant([34, 33, 24, 27, 33, 32], dtype=tf.int64)
      tf.debugging.assert_equal(expected_heads, edge_preds)
      tf.debugging.assert_equal(expected_labels, label_preds)
      print("Passed!")


if __name__ == "__main__":
  unittest.main()
