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
        self.en_train = _read_parser_test_data("john_saw_mary_train")
        self.en_eval = _read_parser_test_data("john_saw_mary_eval")
        self.parser = depparse.DependencyParser()
        self.training_data = map(common.ConnectSentenceNodes, [self.en_train])
        self.parser.MakeFeatures(self.training_data)
        self.parser.Train(3, self.training_data)
    
    def testEvaluate(self):        
        acc = self.parser._Evaluate(self.training_data)
        self.assertTrue(acc == 100)
    
    def testParse(self):
        eval_data = common.ConnectSentenceNodes(self.en_eval)
        eval_data = common.ExtendSentence(eval_data)
        parsed, predicted_heads = self.parser.Parse(eval_data)
        self.assertEqual(predicted_heads, [-1, 2, 0, 2])


if __name__ == "__main__":
  unittest.main()