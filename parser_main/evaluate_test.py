# -*- coding: utf-8 -*-

import os
import unittest
import pandas as pd
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
        evaluator._UasTotal()
        self.assertEqual(evaluator.uas_total, 85.5)
        #print("uas total {}".format(uas_total))
        print("Passed!")

    def test_LasTotal(self):
        print("Running testLasTotal..")
        gold_data = _read_parser_test_data("eval_data_gold")
        test_data = _read_parser_test_data("eval_data_test")
        evaluator = evaluate.Evaluator(gold_data, test_data)
        evaluator._LasTotal()
        self.assertEqual(evaluator.las_total, 73.0)
        #print("las total {}".format(las_total))
        print("Passed!")

    def test_TypedLasPrec(self):
        print("Running test_TypedLasPrec..")
        gold_data = _read_parser_test_data("eval_data_gold")
        test_data = _read_parser_test_data("eval_data_test")
        evaluator = evaluate.Evaluator(gold_data, test_data)
        evaluator._TypedLasPrec()
        expected_result = {
            u'subj': 1.0,
            u'det': 1.0,
            u'dobj': 0.67,
            u'pobj': 1.0,
            u'jj': 0.0,
            u'root': 1.0,
            u'prep': 0.0
        }
        self.assertDictEqual(evaluator.typed_las_prec, expected_result)
        print("Passed!")

    def test_TypedLasRecall(self):
        print("Running test_TypedLasRecall..")
        gold_data = _read_parser_test_data("eval_data_gold")
        test_data = _read_parser_test_data("eval_data_test")
        evaluator = evaluate.Evaluator(gold_data, test_data)
        evaluator._TypedLasRecall()
        expected_result = {
            u'root': 1.0,
            u'det': 0.33,
            u'dobj': 1.0,
            u'pobj': 1.0,
            u'subj': 1.0,
            u'prep': 0.0
        }
        self.assertDictEqual(evaluator.typed_las_recall, expected_result)
        print("Passed!")

    def test_TypedLasF1(self):
        print("Running test_TypedLasF1..")
        gold_data = _read_parser_test_data("eval_data_gold")
        test_data = _read_parser_test_data("eval_data_test")
        evaluator = evaluate.Evaluator(gold_data, test_data)
        evaluator._TypedLasRecall()
        evaluator._TypedLasPrec()
        evaluator._TypedLasF1()
        expected_prec = {
            u"det": 1.00,
            u"dobj":0.67,
            u"jj":0.00,
            u"pobj":1.00,
            u"prep":0.00,
            u"root":1.00,
            u"subj":1.00,
        }
        expected_recall = {
            u'root': 1.0,
            u'det': 0.33,
            u'dobj': 1.0,
            u'pobj': 1.0,
            u'subj': 1.0,
            u'prep': 0.0,
            u"jj": 0.00
        }
        expected_f1 = {
            u"det": 0.496,
            u"dobj":0.802,
            u"jj":0.000,
            u"pobj":1.000,
            u"prep":0.000,
            u"root":1.000,
            u"subj":1.000
        }
        expected_result = pd.DataFrame([expected_prec, expected_recall, expected_f1],
            index=["precision", "recall", "f1"]).T
        self.assertItemsEqual(expected_result, evaluator.typed_las_f1)
        print("Passed!")



if __name__ == "__main__":
	unittest.main()
