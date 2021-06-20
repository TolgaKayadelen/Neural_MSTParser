# -*- coding: utf-8 -*-

import os
import unittest
import pandas as pd
from data.treebank import sentence_pb2
from data.treebank import treebank_pb2
from proto import evaluation_pb2
from eval import evaluate
from util import common

from google.protobuf import text_format

import logging
logging.basicConfig(format='%(levelname)s : %(message)s', level=logging.INFO)


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
   
    def test_label_counts(self):
        print("Running test label_counts..")
        expected_counts = {
            "det": 3,
            "obj":2,
            "pobj":1,
            "root":2,
            "nsubj":2,
            "prep":1,
        }
        function_label_counts = self.evaluator.label_counts.sort_index()
        expected_label_counts = pd.Series(expected_counts, name="count").sort_index()
        self.assertTrue(function_label_counts.equals(expected_label_counts))
        print("Passed!")
    
    def test_uas_total(self):
        print("Running test uas_total..")
        self.assertEqual(round(self.evaluator.uas_total, 1), 85.7)
        print("Passed!")
    
    def test_las_total(self):
        print("Running test las_total..")
        self.assertEqual(round(self.evaluator.las_total, 1), 73.2)
        print("Passed!")
    
    def test_typed_uas(self):
        print("Running test typed_uas..")
        expected_result = {
            "amod": 0.00,
            "det": 0.67,
            "nsubj":1.00,
            "obj":1.00,
            "pobj":1.00,
            "prep":0.00,
            "root":1.00
        }
        self.assertTrue(expected_result, self.evaluator.typed_uas)
        print("Passed..")
    
    def test_labeled_attachment_precision(self):
        print("Running test labeled_attachment_precision..")
        expected_result = {
            'amod': 0.00,
            'nsubj': 1.0,
            'det': 1.0,
            'obj': 0.67,
            'pobj': 1.0,
            'root': 1.0,
            'prep': 0.0
        }
        self.assertDictEqual(self.evaluator.labeled_attachment_prec, expected_result)
        print("Passed!")
    
    def test_labeled_attachment_recall(self):
        print("Running test labeled_attachment_recall..")
        expected_result = {
            'root': 1.0,
            'det': 0.33,
            'obj': 1.0,
            'pobj': 1.0,
            'nsubj': 1.0,
            'prep': 0.0
        }
        self.assertDictEqual(self.evaluator.labeled_attachment_recall, expected_result)
        print("Passed!")
   
    def test_uas_total_las_total(self):
      print("Running test uas_total_las_total")
      results = self.evaluator.evaluate(["las_total", "uas_total"])
      uas = results["uas_total"]
      las = results["las_total"]
      self.assertEqual(round(uas, 1), 85.7)
      self.assertEqual(round(las, 1), 73.2)
      print("Passed!")
    
    def test_labeled_attachment_metrics(self):
        print("Running test_labeled_attachment_metrics..")
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
        expected_result = pd.DataFrame([expected_counts,
                                        expected_prec,
                                        expected_recall,
                                        expected_f1],
            index=["count", "labeled_attachment_precision",
                  "labeled_attachment_recall", "labeled_attachment_f1"]).T
        expected_metrics = expected_result.sort_index()
        self.assertTrue(expected_metrics.equals(self.evaluator.labeled_attachment_metrics))
        print("Passed!")
    
    def test_labels_confusion_matrix(self):
        print("Running test labels_confusion_matrix..")
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
        self.assertTrue(expected_matrix.equals(self.evaluator.labels_conf_matrix))
        print("Passed!")
    
    def test_evaluate(self):
        print("Running testEvaluate..")
        results = self.evaluator.evaluate("all")
        uas = results["uas_total"]
        las = results["las_total"]
        eval_matrix = results["eval_matrix"]
        self.assertEqual(round(uas, 1), 85.7)
        self.assertEqual(round(las, 1), 73.2)
        cols = ["count", "uas", "l_a_prec", "l_a_recall", "l_a_f1"]
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
        expected_eval_matrix = expected_matrix.sort_index()
        function_eval_matrix = self.evaluator.evaluation_matrix.sort_index()
        self.assertTrue(expected_eval_matrix.equals(function_eval_matrix))
        print("Passed!")
        
        # test that the evaluation proto is correct
        expected_proto = text_format.Parse("""
        uas_total: 85.71428571428572
        las_total: 73.21428571428572
        typed_uas {
          uas {
            label: amod
            score: 0.0
          }
          uas {
            label: det
            score: 0.67
          }
          uas {
            label: nsubj
            score: 1.0
          }
          uas {
            label: obj
            score: 1.0
          }
          uas {
            label: pobj
            score: 1.0
          }
          uas {
            label: prep
            score: 0.0
          }        
          uas {
            label: root
            score: 1.0
          }
        }
        labeled_attachment_prec {
          prec {
            label: amod
            score: 0.0
          }
          prec {
            label: det
            score: 1.0
          }
          prec {
            label: nsubj
            score: 1.0
          }
          prec {
            label: obj
            score: 0.67
          }
          prec {
            label: pobj
            score: 1.0
          }
          prec {
            label: prep
            score: 0.0
          }
          prec {
            label: root
            score: 1.0
          }
        }
        labeled_attachment_recall {
          recall {
            label: det
            score: 0.33
          }
          recall {
            label: nsubj
            score: 1.0
          }
          recall {
            label: obj
            score: 1.0
          }
          recall {
            label: pobj
            score: 1.0
          }
          recall {
            label: prep
            score: 0.0
          }
          recall {
            label: root
            score: 1.0
          }
        }
        labeled_attachment_metrics {
          labeled_attachment_metric {
            label: amod
            count: 0
            prec: 0.0
            recall: 0.0
            f1: 0.0
          }
          labeled_attachment_metric {
            label: det
            count: 3
            prec: 1.0
            recall: 0.33
            f1: 0.496
          }
          labeled_attachment_metric {
            label: nsubj
            count: 2
            prec: 1.0
            recall: 1.0
            f1: 1.0
          }
          labeled_attachment_metric {
            label: obj
            count: 2
            prec: 0.67
            recall: 1.0
            f1: 0.802
          }
          labeled_attachment_metric {
            label: pobj
            count: 1
            prec: 1.0
            recall: 1.0
            f1: 1.0
          }
          labeled_attachment_metric {
            label: prep
            count: 1
            prec: 0.0
            recall: 0.0
            f1: 0.0
          }
          labeled_attachment_metric {
            label: root
            count: 2
            prec: 1.0
            recall: 1.0
            f1: 1.0
          }
        }""", evaluation_pb2.Evaluation())
        self.assertEqual(expected_proto, self.evaluator.evaluation)
        
if __name__ == "__main__":
	unittest.main()
