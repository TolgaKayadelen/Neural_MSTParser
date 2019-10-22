# -*- coding: utf-8 -*-

import unittest
import os

from parser import dependency_labeler as deplabel
from google.protobuf import text_format
from data.treebank import sentence_pb2
from util import common
from util import reader


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

#(TODO): maybe remove this later.  
def _read_perceptron_test_data(basename):
    path = os.path.join(_PERCEPTRON_DIR, "{}.pbtxt".format(basename))
    return text_format.Parse(_read_file(path), sentence_pb2.Sentence())

class DependencyLabelerTest(unittest.TestCase):
  """Test for the dependency labeler module."""
  
  def setUp(self):
    self.en_train = _read_perceptron_test_data("john_saw_mary")
    self.en_train.length = len(self.en_train.token)
    tokens = self.en_train.token
    labels = ["nsubj", "root", "obj"]
    for i, token in enumerate(tokens[1:]):
      token.label = labels[i]
  
  def testTrainSingleSentence(self):
    print("Running testTrain on a Single Sentence..")
    labeler = deplabel.DependencyLabeler()
    labeler.MakeFeatures([self.en_train])
    labeler.Train(3, [self.en_train])
    self.assertEqual(labeler.label_accuracy, 100.0)
    print("Passed!")

  def testTrainTreebank(self):
    print("Running testTrain on a treebank..")
    labeler = deplabel.DependencyLabeler()
    treebank = reader.ReadTreebankTextProto("data/UDv23/English/training/treebank_0_10.pbtxt")
    training_data = treebank.sentence
    labeler.MakeFeatures(training_data)
    #(TODO): proceed from here. 
    print("Total Number of Features {}".format(labeler.label_perceptron.feature_count))
  
  def testPredictLabels(self):
    print("Running testPredictLabels..")
    labeler = deplabel.DependencyLabeler()
    labeler.MakeFeatures([self.en_train])
    for i in range(3):
      _ = labeler.label_perceptron.Train([self.en_train])
    predicted_labels = labeler.PredictLabels(self.en_train)
    self.assertEqual(predicted_labels, [token.label for token in self.en_train.token])
    
    # Also insert some other labels, and test again.
    new_labels = [u"", u"cc", u"root", u"cc"]
    labeler.InsertLabels(self.en_train, new_labels)
    self.assertEqual(new_labels, [token.label for token in self.en_train.token])
    print("Passed!")
    
  def testInsertLabels(self):
    print("Running testInsertLabels..")
    labels = [u"", u"cc", u"root", u"cc"]
    labeler = deplabel.DependencyLabeler()
    labeled = labeler.InsertLabels(self.en_train, labels)
    self.assertEqual(labels, [token.label for token in labeled.token])
    print("Passed!")
    


if __name__ == "__main__":
  unittest.main()