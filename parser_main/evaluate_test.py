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

    def setUp(self):
        self.gold_data = _read_parser_test_data("eval_data_gold")
        self.test_data = _read_parser_test_data("eval_data_test")
        self.evaluator = evaluate.Evaluator(self.gold_data, self.test_data)

    def test_GetLabelCounts(self):
        print("Running testGetLabelCounts..")
        self.evaluator._GetLabelCounts()
        expected_counts = {
            u"det": 3,
            u"dobj":2,
            u"pobj":1,
            u"prep":1,
            u"root":2,
            u"subj":2
        }
        #print(self.evaluator.label_counts)
        self.assertTrue(pd.Series(expected_counts).equals(self.evaluator.label_counts))
        print("Passed!")

    def test_UasTotal(self):
        print("Running testUasTotal..")
        self.evaluator._UasTotal()
        self.assertEqual(self.evaluator.uas_total, 85.5)
        print("Passed!")

    def test_LasTotal(self):
        print("Running testLasTotal..")
        self.evaluator._LasTotal()
        self.assertEqual(self.evaluator.las_total, 73.0)
        print("Passed!")

    def test_TypedUas(self):
        print("Running test_TypedUas..")
        self.evaluator._TypedUas()
        expected_result = {
            u"det": 0.67,
            u"dobj":1.00,
            u"jj":0.00,
            u"pobj":1.00,
            u"prep":0.00,
            u"root":1.00,
            u"subj":1.00
        }
        self.assertTrue(pd.Series(expected_result).equals(self.evaluator.typed_uas))
        print("Passed..")

    def test_TypedLasPrec(self):
        print("Running test_TypedLasPrec..")
        self.evaluator._TypedLasPrec()
        expected_result = {
            u'subj': 1.0,
            u'det': 1.0,
            u'dobj': 0.67,
            u'pobj': 1.0,
            u'jj': 0.0,
            u'root': 1.0,
            u'prep': 0.0
        }
        self.assertDictEqual(self.evaluator.typed_las_prec, expected_result)
        print("Passed!")

    def test_TypedLasRecall(self):
        print("Running test_TypedLasRecall..")
        self.evaluator._TypedLasRecall()
        expected_result = {
            u'root': 1.0,
            u'det': 0.33,
            u'dobj': 1.0,
            u'pobj': 1.0,
            u'subj': 1.0,
            u'prep': 0.0
        }
        self.assertDictEqual(self.evaluator.typed_las_recall, expected_result)
        print("Passed!")

    def test_TypedLasF1(self):
        print("Running test_TypedLasF1..")
        self.evaluator._TypedLasRecall()
        self.evaluator._TypedLasPrec()
        self.evaluator._TypedLasF1()
        expected_counts = {
            u"det": 3.0,
            u"dobj":2.0,
            u"jj":0.0,
            u"pobj":1.0,
            u"prep":1.0,
            u"root":2.0,
            u"subj":2.0,
        }
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
        expected_result = pd.DataFrame([expected_counts,expected_prec, expected_recall, expected_f1],
            index=["count", "label_precision", "label_recall", "label_f1"]).T
        self.assertTrue(expected_result.equals(self.evaluator.typed_las_f1))
        print("Passed!")

    def testEvaluate(self):
        print("Running testEvaluate..")
        uas, las, eval_matrix = self.evaluator.Evaluate("all")
        self.assertEqual(uas, 85.5)
        self.assertEqual(las, 73.0)
        cols = ["count", "unlabeled_attachment", "label_prec", "label_recall", "label_f1"]
        index = ["det", "dobj", "jj", "pobj", "prep", "root", "subj"]
        expected_matrix = pd.DataFrame(
            columns=cols,
            index=index,
            data=[
                [3.0, 0.67, 1.00, 0.33,0.496],
                [2.0, 1.00, 0.67, 1.00, 0.802],
                [0.0, 0.00, 0.00, 0.00, 0.000],
                [1.0, 1.00, 1.00, 1.00, 1.000],
                [1.0, 0.00, 0.00, 0.00, 0.000],
                [2.0, 1.00, 1.00, 1.00, 1.000],
                [2.0, 1.00, 1.00, 1.00, 1.000]
            ])
        self.assertTrue(expected_matrix.equals(self.evaluator.evaluation_matrix))
        print("Passed!")



if __name__ == "__main__":
	unittest.main()
