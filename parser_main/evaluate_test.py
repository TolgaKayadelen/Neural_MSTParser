# -*- coding: utf-8 -*-

import os
import unittest
from data.treebank import sentence_pb2
from data.treebank import treebank_pb2
from parser_main import evaluate
from util import common

from google.protobuf import text_format

import logging
logging.basicConfig(format='%(levelname)s : %(message)s', level=logging.DEBUG)


_TESTDATA_DIR = "data/testdata"
_PARSERMAIN_DIR = os.path.join(_TESTDATA_DIR, "parser_main")


def _read_file(path):
    with open(path, "r") as f:
        read = f.read()
    return read

def _read_parser_test_data(basename):
    path = os.path.join(_PARSERMAIN_DIR, "{}.pbtxt".format(basename))
    return text_format.Parse(_read_file(path), treebank_pb2.Treebank())


class EvaluateTest(unittest.TestCase):
	"""Tests for the Evaluator"""
	
	def test_UasTotal(self):
		print("Running testUasTotal..")
		gold_data = _read_parser_test_data("eval_data_gold")
		test_data = _read_parser_test_data("eval_data_test")
		evaluator = evaluate.Evaluator(gold_data, test_data)
		uas_total = evaluator._UasTotal()
		self.assertEqual(uas_total, 85.5)
		#print(uas_total)
		print("Passed!")
	
	def testEvaluate(self):
		print("Running testEvaluate..")
		print("Passed!")

if __name__ == "__main__":
	unittest.main()
