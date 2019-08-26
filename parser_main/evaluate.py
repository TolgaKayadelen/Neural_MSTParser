# -*- coding: utf-8 -*-

"""Module to evaluate sentences parsed with the dependency parser.

This module implements the following evaluation metrics:

UAS: Unlabeld attachment score.
LAS: Labeled attachment score.

We provide typed based precision, recall and F1 scores for LAS and type based
accuracy for UAS.

"""

import pandas as pd
from data.treebank import sentence_pb2
from data.treebank import treebank_pb2
from util import common
from util import reader

from google.protobuf import text_format

import logging
logging.basicConfig(format='%(levelname)s : %(message)s', level=logging.DEBUG)

def evaluate_parser(args):
    """Function to evaluate the dependency parser output on gold data.
    Args:
    	command line arguments, values for requested metrics, gold data and
    	test data path.
    """
    gold_data = reader.ReadTreebankTextProto(args.gold_data)
    test_data = reader.ReadTreebankTextProto(args.test_data)
    eval_metrics = args.metrics
    logging.info("Requested eval metrics {}".format(eval_metrics))
    evaluator = Evaluator(gold_data, test_data)
    results = evaluator.Evaluate(eval_metrics)
    for result in results:
        print(result)

get_labels = lambda sent: [token.label for token in sent.token[1:]]

same_head = (lambda t1, t2:
	t1.selected_head.address == t2.selected_head.address)

class Evaluator:

    def __init__(self, gold, test):
        """
        Args:
            gold: treebank, the gold treebank.
            test: treebank, the treebank annotated by the system.

            Initializes this evaluator with a gold and test treebank.
        """
        assert isinstance(gold, treebank_pb2.Treebank)
        assert isinstance(test, treebank_pb2.Treebank)
        self.gold = list(gold.sentence)
        self.test = list(test.sentence)
        self.gold_and_test = zip(self.gold, self.test)
        self.uas_total = 0.0
        self.las_total = 0.0
        self.typed_las_prec = {}
        self.typed_las_recall = {}
        self.typed_uas = None # pd.Series
        self.label_counts = None # pd.Series
        self.typed_las_f1 = None # pd.DataFrame
        self.evaluation_matrix = None # pd.DataFrame showing all the results

    def _GetLabelCounts(self):
        """Return number of occurences for each label in the data."""
        labels = list(set().union(*map(get_labels, self.gold)))
        assert labels, "Tokens don't have any labels!!!"
        label_counts = {}
        for label in labels:
            label_count = 0
            for sent in self.gold:
                for token in sent.token:
                    if not token.label == label:
                        continue
                    label_count += 1
            label_counts[label] = label_count
        self.label_counts = pd.Series(label_counts).rename("count", inplace=True)
        #print(self.label_counts)

    def _UasTotal(self):
        "Computes the total Unlabeled Attachement Score of the parser."
        uas = 0.0
        for gold_sent, test_sent in self.gold_and_test:
            gold_heads = [token.selected_head.address for token in gold_sent.token[1:]]
            pred_heads = [token.selected_head.address for token in test_sent.token[1:]]
            assert len(gold_heads) == len(pred_heads), "Tokenization mismatch!!"
            #for gold_head, predicted_head in zip(gold_heads, pred_heads):
            #	print("gold: {}, predicted: {}, equal: {}".format(
            #   gold_head, predicted_head, gold_head == predicted_head))
            #print("---")
            uas += 100 * sum(
                gh == ph for gh, ph in zip(gold_heads, pred_heads)) / len(gold_heads)
        self.uas_total = uas / len(self.gold)

    def _LasTotal(self):
        "Computes the total Labeled Attachment Score of the parser."
        las = 0.0
        for gold_sent, test_sent in self.gold_and_test:
            gold_heads = [(token.selected_head.address, token.label)
                for token in gold_sent.token[1:]]
            pred_heads = [(token.selected_head.address, token.label)
                for token in test_sent.token[1:]]
            assert len(gold_heads) == len(pred_heads), "Tokenization mismatch!!"
            las += 100 * sum(
                gh == ph for gh, ph in zip(gold_heads, pred_heads)) / len(gold_heads)
        self.las_total = las / len(self.gold)

    def _TypedUas(self):
        """Computes Unlabeled Attachment Score for all dependency types."""
        labels = list(set().union(*map(get_labels, self.test)))
        assert labels, "Cannot compute typed Uas without labels!!"
        typed_uas = {}
        for label in labels:
            correct = 0.0
            label_uas = 0.0
            match = []
            sentence_idx = 0
            #print("label is {}".format(label))
            for gold_sent, test_sent in self.gold_and_test:
                sentence_idx = 1
                for gold_tok, test_tok in zip(gold_sent.token, test_sent.token):
                    if not gold_tok.label == label:
                        continue
                    correct += 1.0
                    if test_tok.selected_head.address == gold_tok.selected_head.address:
                        label_uas += 1.0
                        #match.append((sentence_idx, test_tok.index, test_tok.word))
            #logging.info("matches are {}".format(match))
            typed_uas[label] = label_uas / correct if correct else 0.0
        self.typed_uas = pd.Series(typed_uas).rename("unlabeled_attachment", inplace=True).round(2)
        #print(self.typed_uas)

    def	_TypedLasPrec(self):
        """Computes Precision for all dependency types.
        For each relation X, precision computes the percentage of relations X
        in the system that are correct (correct / system). That is, it checks
        whether the X's that are found in the system also exists in the gold.
        """
        labels = list(set().union(*map(get_labels, self.test)))
        for label in labels:
            #print("Computing precision for {}".format(label))
            correct = 0.0
            system = 0.0
            match = []
            sentence_idx = 0
            for gold_sent, test_sent in self.gold_and_test:
                sentence_idx += 1
                for gold_tok, test_tok in zip(gold_sent.token, test_sent.token):
                    if not test_tok.label == label:
                        continue
                    # system has found this label.
                    system += 1
                    if gold_tok.label == label and same_head(gold_tok, test_tok):
                        correct += 1
                        match.append((sentence_idx, gold_tok.index, gold_tok.word))
            #logging.info("System has: {} \"{}\"".format(system, label))
            #logging.info("{} of the system token(s) were also \"{}\" in gold".format(correct, label))
            #logging.info("Precision of the model for label \"{}\" is {} / {} = {}".format(
            #	label, correct, system, correct / system)
            #)
            #logging.info("matches are: {}".format(match))
            self.typed_las_prec[label] = round((correct / system), 2)
            #print("----------------------------------------------------------")

    def _TypedLasRecall(self):
        """Computes Recall for all dependency types.

        For each relation X, recall computes the percentage of relations that
        exists in the gold which are recovered by the system (correct / gold).
        """
        labels = list(set().union(*map(get_labels, self.gold)))
        for label in labels:
            #print("Computing recall for {}".format(label))
            correct = 0.0
            gold = 0.0
            match = []
            sentence_idx = 0
            for gold_sent, test_sent in self.gold_and_test:
                sentence_idx += 1
                for gold_tok, test_tok in zip(gold_sent.token, test_sent.token):
                    if not gold_tok.label == label:
                        continue
                    gold += 1
                    if test_tok.label == label and same_head(gold_tok, test_tok):
                        correct += 1
                        match.append((sentence_idx, test_tok.index, test_tok.word))
            #logging.info("Gold has: {} \"{}\"".format(gold, label))
            #logging.info("{} of the gold label(s) were recovered by system".format(correct))
            #logging.info("Recall  of the model for label \"{}\" is  {} / {} = {}".format(
            #    label, correct, gold, correct / gold)
            #)
            #logging.info("matches are: {}".format(match))
            self.typed_las_recall[label] = round((correct / gold), 2)
            #print("----------------------------------------------------------")

    def _TypedLasF1(self):
        """Computes F1 score for all dependency types"""
        f1 = lambda recall, precision: 2 * (recall * precision) / (recall + precision)
        if not self.typed_las_prec:
            logging.info("Computing precision..")
            self._TypedLasPrec()
        if not self.typed_las_recall:
            logging.info("Computing recall..")
            self._TypedLasRecall()
        assert self.typed_las_prec, "Cannot compute f1 without precision score!"
        assert self.typed_las_recall, "Cannote compute f1 without recall score!"
        cols = ["count", "label_precision", "label_recall", "label_f1"]
        if self.label_counts is None:
            logging.info("Getting label counts..")
            self._GetLabelCounts()
        self.typed_las_f1 = pd.DataFrame(
			 [self.typed_las_prec, self.typed_las_recall],
			 index=["label_precision", "label_recall"]).fillna(0).T
        self.typed_las_f1["label_f1"] = f1(self.typed_las_f1["label_recall"],
                                           self.typed_las_f1["label_precision"]).fillna(0).round(3)
        self.typed_las_f1["count"] = self.label_counts

        self.typed_las_f1 = self.typed_las_f1[cols]
        self.typed_las_f1.fillna(0, inplace=True)
        #print(self.typed_las_f1)

    def _EvaluateAll(self):
        "Runs all the evaluation metrics."
        self._GetLabelCounts()
        self._UasTotal()
        self._LasTotal()
        self._TypedUas()
        self._TypedLasPrec()
        self._TypedLasRecall()
        self._TypedLasF1()
        self.evaluation_matrix = pd.DataFrame()
        self.evaluation_matrix["count"] = self.typed_las_f1["count"]
        self.evaluation_matrix["unlabeled_attachment"] = self.typed_uas
        self.evaluation_matrix["label_prec"] = self.typed_las_f1["label_precision"]
        self.evaluation_matrix["label_recall"] = self.typed_las_f1["label_recall"]
        self.evaluation_matrix["label_f1"] = self.typed_las_f1["label_f1"]

    def Evaluate(self, *args):
        requested_metrics = args[0]
        metrics = ["uas_total", "las_total", "typed_uas", "typed_las_prec",
				   "typed_las_recall", "typed_las_f1", "all"]
        assert any(metric in requested_metrics for metric in metrics), "No valid metric!"
        if "all" in requested_metrics:
            self._EvaluateAll()
            return ["uas_total: ", self.uas_total,
                    "las_total: ", self.las_total,
                    "eval_matrix: ", self.evaluation_matrix]
        else:
            if "uas_total" in requested_metrics:
                self._UasTotal()
                return ["uas_total: ", self.uas_total]
            if "las_total" in requested_metrics:
                self._LasTotal()
                return ["las_total: ", self.las_total]
            if "typed_uas" in requested_metrics:
                self._TypedUas()
                return ["typed_uas: ", self.typed_uas]
            if  "typed_las_prec" in requested_metrics:
                self._TypedLasPrec()
                return ["las_prec: ", self.typed_las_prec]
            if	"typed_las_recall" in requested_metrics:
                self._TypedLasRecall()
                return ["las_recall: ", self.typed_las_recall]
            if "typed_las_f1" in requested_metrics:
                self._TypedLasF1()
                # this returns precision and recall as well
                return ["las_f1: ", self.typed_las_f1]
