# -*- coding: utf-8 -*-

import unittest
import os
import numpy as np
from parser.perceptron import dependency_labeler as deplabel
from google.protobuf import text_format
from data.treebank import sentence_pb2
from util import common
from util import reader


_TESTDATA_DIR = "data/testdata"
_LABELER_DIR = os.path.join(_TESTDATA_DIR, "labeler")

def _read_file(path):
    with open(path, "r") as f:
        read = f.read()
    return read

def _read_labeler_test_data(basename, type_="sentence"):
    path = os.path.join(_LABELER_DIR, "{}.pbtxt".format(basename))
    if type_ == "sentence":
      return text_format.Parse(_read_file(path), sentence_pb2.Sentence())
    else:
      return reader.ReadTreebankTextProto(path)

class DependencyLabelerTest(unittest.TestCase):
  """Test for the dependency labeler module."""
  
  def setUp(self):
    self.en_train = _read_labeler_test_data("john_saw_mary")
    self.en_train.length = len(self.en_train.token)
    tokens = self.en_train.token
    labels = ["nsubj", "root", "obj"]
    for i, token in enumerate(tokens[1:]):
      token.label = labels[i]
  
  def testTrainSingleSentence(self):
    print("Running testTrain on a Single Sentence..")
    labeler = deplabel.DependencyLabeler(feature_file="labelfeatures_base")
    labeler.MakeFeatures([self.en_train])
    labeler.Train(4, [self.en_train])
    self.assertEqual(labeler.label_accuracy_train, 100.0)
    print("Passed!")

  def testTrainTreebank(self):
    print("Running testTrain on a treebank..")
    labeler = deplabel.DependencyLabeler(feature_file="labelfeatures_base")
    treebank = _read_labeler_test_data("treebank_0_10", type_="treebank")
    training_data = list(treebank.sentence)
    labeler.MakeFeatures(training_data)
    labeler.Train(5, training_data)
    self.assertTrue(labeler.label_accuracy_train > 90.0)
    print("Passed!")
  
  def testTrainAndEvaluateWithTestData(self):
    print("Running test Train and Evaluate on different sets of data..")
    labeler = deplabel.DependencyLabeler(feature_file="labelfeatures_base")
    train_treebank = _read_labeler_test_data("treebank_0_50", type_="treebank")
    training_data = list(train_treebank.sentence)
    test_treebank = _read_labeler_test_data("treebank_0_10", type_="treebank")
    test_data = list(test_treebank.sentence)
    labeler.MakeFeatures(training_data)
    #print(labeler.label_perceptron.labels) 
    print("Total Number of Features {}".format(labeler.label_perceptron.feature_count))
    labeler.Train(5, training_data, test_data=test_data)
    self.assertTrue(labeler.label_accuracy_test > 65.0)
    print("Passed!")
  
  def testPredictLabels(self):
    print("Running testPredictLabels..")
    labeler = deplabel.DependencyLabeler(feature_file="labelfeatures_base")
    labeler.MakeFeatures([self.en_train])
    for i in range(5):
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
    labeler = deplabel.DependencyLabeler(feature_file="labelfeatures_base")
    labeled = labeler.InsertLabels(self.en_train, labels)
    self.assertEqual(labels, [token.label for token in labeled.token])
    print("Passed!")
    

if __name__ == "__main__":
  unittest.main()