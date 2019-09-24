# -*- coding: utf-8 -*-

import os
import unittest
import pandas as pd
from data.treebank import sentence_pb2
from data.treebank import treebank_pb2
from model import evaluation_pb2
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
            u"obj":2,
            u"pobj":1,
            u"prep":1,
            u"root":2,
            u"nsubj":2
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
            u"obj":1.00,
            u"amod":0.00,
            u"pobj":1.00,
            u"prep":0.00,
            u"root":1.00,
            u"nsubj":1.00
        }
        self.assertTrue(pd.Series(expected_result).equals(self.evaluator.typed_uas))
        print("Passed..")

    def test_TypedLasPrec(self):
        print("Running test_TypedLasPrec..")
        self.evaluator._TypedLasPrec()
        expected_result = {
            u'nsubj': 1.0,
            u'det': 1.0,
            u'obj': 0.67,
            u'pobj': 1.0,
            u'amod': 0.0,
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
            u'obj': 1.0,
            u'pobj': 1.0,
            u'nsubj': 1.0,
            u'prep': 0.0
        }
        self.assertDictEqual(self.evaluator.typed_las_recall, expected_result)
        print("Passed!")

    def test_UasTotalLasTotal(self):
      print("Running test_UasTotalLasTotal")
      results = self.evaluator.Evaluate(["uas_total", "las_total"])
      uas = results["uas_total"]
      las = results["las_total"]
      self.assertEqual(uas, 85.5)
      self.assertEqual(las, 73.0)
      print("Passed!")

    def test_TypedLasF1(self):
        print("Running test_TypedLasF1..")
        self.evaluator._TypedLasRecall()
        self.evaluator._TypedLasPrec()
        self.evaluator._TypedLasF1()
        expected_counts = {
            u"det": 3.0,
            u"obj":2.0,
            u"amod":0.0,
            u"pobj":1.0,
            u"prep":1.0,
            u"root":2.0,
            u"nsubj":2.0,
        }
        expected_prec = {
            u"det": 1.00,
            u"obj":0.67,
            u"amod":0.00,
            u"pobj":1.00,
            u"prep":0.00,
            u"root":1.00,
            u"nsubj":1.00,
        }
        expected_recall = {
            u'root': 1.0,
            u'det': 0.33,
            u'obj': 1.0,
            u'pobj': 1.0,
            u'nsubj': 1.0,
            u'prep': 0.0,
            u"amod": 0.00
        }
        expected_f1 = {
            u"det": 0.496,
            u"obj":0.802,
            u"amod":0.000,
            u"pobj":1.000,
            u"prep":0.000,
            u"root":1.000,
            u"nsubj":1.000
        }
        expected_result = pd.DataFrame([expected_counts,expected_prec, expected_recall, expected_f1],
            index=["count", "label_precision", "label_recall", "label_f1"]).T
        self.assertTrue(expected_result.equals(self.evaluator.typed_las_f1))
        print("Passed!")

    def testCreateConfusionMatrix(self):
        print("Running testCreateConfusionMatrix..")
        self.evaluator.CreateConfusionMatrix()
        index = ["det", "nsubj", "obj", "pobj", "prep", "root", "All"]
        cols = ["amod", "det", "nsubj", "obj", "pobj", "prep", "root", "All"]
        expected_matrix = pd.DataFrame(
            columns = cols,
            index = index,
            data = [
                [1,1,0,1,0,0,0,3], #det
                [0,0,2,0,0,0,0,2], #nsubj
                [0,0,0,2,0,0,0,2], #obj
                [0,0,0,0,1,0,0,1], #pobj
                [0,0,0,0,0,1,0,1], #prep
                [0,0,0,0,0,0,2,2], #root
                [1,1,2,3,1,1,2,11] #all
            ])
        #print(expected_matrix)
        self.assertTrue(expected_matrix.equals(self.evaluator.labels_conf_matrix))
        print("Passed!")

    def testEvaluate(self):
        print("Running testEvaluate..")
        results = self.evaluator.Evaluate("all")
        uas = results["uas_total"]
        las = results["las_total"]
        eval_matrix = results["eval_matrix"]
        self.assertEqual(uas, 85.5)
        self.assertEqual(las, 73.0)
        cols = ["count", "unlabeled_attachment", "label_prec", "label_recall", "label_f1"]
        index = ["amod", "det", "nsubj", "obj", "pobj", "prep", "root"]
        expected_matrix = pd.DataFrame(
            columns=cols,
            index=index,
            data=[
                [0.0, 0.00, 0.00, 0.00, 0.000], # amod
                [3.0, 0.67, 1.00, 0.33,0.496],  # det 
                [2.0, 1.00, 1.00, 1.00, 1.000], # nsubj
                [2.0, 1.00, 0.67, 1.00, 0.802], # obj
                [1.0, 1.00, 1.00, 1.00, 1.000], # pobj
                [1.0, 0.00, 0.00, 0.00, 0.000], # prep
                [2.0, 1.00, 1.00, 1.00, 1.000]  # root
            ])
        self.assertTrue(expected_matrix.equals(self.evaluator.evaluation_matrix))
        print("Passed!")
        
        # test that the evaluation proto is correct
        expected_proto = text_format.Parse("""
        uas_total: 85.5
        las_total: 73.0
        typed_uas {
          uas {
            label: OBJ
            score: 1.0
          }
          uas {
            label: NSUBJ
            score: 1.0
          }
          uas {
            label: DET
            score: 0.67
          }
          uas {
            label: AMOD
            score: 0.0
          }
          uas {
            label: POBJ
            score: 1.0
          }
          uas {
            label: ROOT
            score: 1.0
          }
          uas {
            label: PREP
            score: 0.0
          }
        }
        typed_las_prec {
          prec {
            label: OBJ
            score: 0.67
          }
          prec {
            label: NSUBJ
            score: 1.0
          }
          prec {
            label: DET
            score: 1.0
          }
          prec {
            label: AMOD
            score: 0.0
          }
          prec {
            label: POBJ
            score: 1.0
          }
          prec {
            label: ROOT
            score: 1.0
          }
          prec {
            label: PREP
            score: 0.0
          }
        }
        typed_las_recall {
          recall {
            label: OBJ
            score: 1.0
          }
          recall {
            label: NSUBJ
            score: 1.0
          }
          recall {
            label: DET
            score: 0.33
          }
          recall {
            label: POBJ
            score: 1.0
          }
          recall {
            label: ROOT
            score: 1.0
          }
          recall {
            label: PREP
            score: 0.0
          }
        }
        typed_las_f1 {
          f1 {
            label: AMOD
            count: 0
            prec: 0.0
            recall: 0.0
            f1: 0.0
          }
          f1 {
            label: DET
            count: 3
            prec: 1.0
            recall: 0.33
            f1: 0.33
          }
          f1 {
            label: NSUBJ
            count: 2
            prec: 1.0
            recall: 1.0
            f1: 1.0
          }
          f1 {
            label: OBJ
            count: 2
            prec: 0.67
            recall: 1.0
            f1: 1.0
          }
          f1 {
            label: POBJ
            count: 1
            prec: 1.0
            recall: 1.0
            f1: 1.0
          }
          f1 {
            label: PREP
            count: 1
            prec: 0.0
            recall: 0.0
            f1: 0.0
          }
          f1 {
            label: ROOT
            count: 2
            prec: 1.0
            recall: 1.0
            f1: 1.0
          }
        }""", evaluation_pb2.Evaluation())
        self.assertEqual(expected_proto, self.evaluator.evaluation)
      
      



if __name__ == "__main__":
	unittest.main()
