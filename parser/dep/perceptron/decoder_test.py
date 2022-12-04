# -*- coding: utf-8 -*-

import unittest
import os

from parser.perceptron import decoder
from google.protobuf import text_format
from data.treebank import sentence_pb2
from util import common
import numpy as np


_TESTDATA_DIR = "data/testdata"
_PARSER_DIR = os.path.join(_TESTDATA_DIR, "parser")

def _read_file(path):
    with open(path, "r") as f:
        read = f.read()
    return read

def _read_parser_test_data(basename):
    path = os.path.join(_PARSER_DIR, "{}.pbtxt".format(basename))
    return text_format.Parse(_read_file(path), sentence_pb2.Sentence())


class DecoderTest(unittest.TestCase):
    """Tests for the dependency parser module."""
    def testDecode(self):
        print("Running testDecode()..")
        dc = decoder.Decoder()
        test_sentence = _read_parser_test_data("john_saw_mary_eval")
        scores = np.array([
            [0.,0.,0.,0.],
            [-13.,0.,19.,-2.],
            [ 13.,-19.,0.,-6.],
            [ -3.,-6.,10.,0.]])
        decoded, predicted_heads = dc(test_sentence, scores)
        self.assertEqual(list(predicted_heads), [-1,2,0,2])
        self.assertEqual(decoded.score, 42.0)
        print("Passed!")

if __name__ == "__main__":
  unittest.main()
