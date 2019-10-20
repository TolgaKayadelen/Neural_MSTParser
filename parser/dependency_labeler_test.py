# -*- coding: utf-8 -*-

import unittest
import os

from parser import dependency_labeler as deplabel
from google.protobuf import text_format
from data.treebank import sentence_pb2
from util import common


_TESTDATA_DIR = "data/testdata"
_PARSER_DIR = os.path.join(_TESTDATA_DIR, "parser")
# (TODO): remove this later.
_PERCEPTRON_DIR = os.path.join(_TESTDATA_DIR, "perceptron")


def _read_file(path):
    with open(path, "r") as f:
        read = f.read()
    return read

def _read_parser_test_data(basename):
    path = os.path.join(_PARSER_DIR, "{}.pbtxt".format(basename))
    return text_format.Parse(_read_file(path), sentence_pb2.Sentence())

#(TODO): remove this later.  
def _read_perceptron_test_data(basename):
    path = os.path.join(_PERCEPTRON_DIR, "{}.pbtxt".format(basename))
    return text_format.Parse(_read_file(path), sentence_pb2.Sentence())

class DependencyLabelerTest(unittest.TestCase):
  """Test for the dependency labeler module."""
  
  def setUp(self):
    self.en_train = _read_perceptron_test_data("john_saw_mary")
    self.en_train.length = len(self.en_train.token)
  
  def testTrain(self):
    print("Running testTrain..")
    labeler = deplabel.DependencyLabeler()
    labeler.MakeFeatures([self.en_train])
    tokens = self.en_train.token
    labels = ["nsubj", "root", "obj"]
    for i, token in enumerate(tokens[1:]):
      token.label = labels[i]
    labeler.Train(3, [self.en_train])
    self.assertEqual(labeler.label_accuracy, 100.0)
    print("Passed!")


if __name__ == "__main__":
  unittest.main()