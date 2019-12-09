# -*- coding: utf-8 -*-

import unittest
import os

from parser import dependency_parser as depparse
from google.protobuf import text_format
from data.treebank import sentence_pb2
from util import common


_TESTDATA_DIR = "data/testdata"
_PARSER_DIR = os.path.join(_TESTDATA_DIR, "parser")


def _read_file(path):
    with open(path, "r") as f:
        read = f.read()
    return read

def _read_parser_test_data(basename):
    path = os.path.join(_PARSER_DIR, "{}.pbtxt".format(basename))
    return text_format.Parse(_read_file(path), sentence_pb2.Sentence())


class DependencyParserTest(unittest.TestCase):
    """Tests for the dependency parser module."""
    
    def setUp(self):
        # Train an English parser with an English sentence.
        
        self.en_train = _read_parser_test_data("john_saw_mary_train")
        self.en_eval = _read_parser_test_data("john_saw_mary_eval")
        self.en_parser = depparse.DependencyParser(feature_file="arcfeatures_base")
        # NOTE: connecting sentence nodes is something we need to make features.
        self.en_training_data = map(common.ConnectSentenceNodes, [self.en_train])
        self.en_training_data = map(common.ExtendSentence, self.en_training_data)
        self.en_parser.MakeFeatures(self.en_training_data)
        self.en_parser.Train(3, self.en_training_data)
        
        
        #Train a Turkish parser with a Turkish sentence.
        self.tr_train = _read_parser_test_data("kerem_train")
        self.tr_eval = _read_parser_test_data("kerem_eval")
        self.tr_parser = depparse.DependencyParser(feature_file="arcfeatures_base")
        self.tr_training_data = map(common.ConnectSentenceNodes, [self.tr_train])
        self.tr_training_data = map(common.ExtendSentence, self.tr_training_data)
        self.tr_parser.MakeFeatures(self.tr_training_data)
        self.tr_parser.Train(3, self.tr_training_data)
        
        #common.PPrintWeights(self.tr_parser.arc_perceptron.weights)
    
    def testEvaluate(self):
        print("Running testEvaluate..")        
        en_acc = self.en_parser._Evaluate(self.en_training_data)
        self.assertTrue(en_acc == 100)
        
        tr_acc = self.tr_parser._Evaluate(self.tr_training_data)
        self.assertTrue(tr_acc == 100)
        print("Passed!")
    
    def testParse(self):
        print("Running test_Parse..")
        # parse on the training data
        en_eval_data = common.ConnectSentenceNodes(self.en_eval)
        en_eval_data = common.ExtendSentence(en_eval_data)
        en_parsed, en_predicted_heads = self.en_parser.Parse(en_eval_data)
        self.assertEqual(list(en_predicted_heads), [-1, 2, 0, 2])
        
        # test on some other test data
        en_test_data = common.ConnectSentenceNodes(_read_parser_test_data("sam_killed_pam_eval"))
        en_test_data = common.ExtendSentence(en_test_data)
        en_parsed, en_predicted_heads = self.en_parser.Parse(en_test_data)
        self.assertEqual(list(en_predicted_heads), [-1, 2, 0, 2])
        
        tr_eval_data = common.ConnectSentenceNodes(self.tr_eval)
        tr_eval_data = common.ExtendSentence(tr_eval_data)
        tr_parsed, tr_predicted_heads = self.tr_parser.Parse(tr_eval_data)
        self.assertEqual(list(tr_predicted_heads), [-1, 8, 8, 4, 8, 4, 4, 8, 0, 8])
        print("Passed!")

    

if __name__ == "__main__":
  unittest.main()