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

from google.protobuf import text_format

import logging
logging.basicConfig(format='%(levelname)s : %(message)s', level=logging.DEBUG)

def evaluate_parser(args):
	"""Function to evaluate the dependency parser output on gold data.
	Args:
		command line arguments, values for requested metrics, gold data and
			test data path.
	"""
	#gold_data = args.gold_data
	#test_data = args.test_data
	eval_metrics = args.metrics
	print("Requested eval metrics {}".format(eval_metrics))
	#evaluator = Evaluator(gold_data, test_data)
	#evaluator.Evaluate(eval_metrics)

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
		self.typed_uas = {}
		self.typed_las_prec = {}
		self.typed_las_recall = {}
		self.typed_las_f1 = None

	def _UasTotal(self):
		"Computes the total Unlabeled Attachement Score of the parser."
		uas = 0.0
		for gold_sent, test_sent in self.gold_and_test:
			gold_heads = [token.selected_head.address for token in
			 gold_sent.token[1:]]
			pred_heads = [token.selected_head.address for token in
			 test_sent.token[1:]]
			assert len(gold_heads) == len(pred_heads), "Tokenization mismatch!!"
			#for gold_head, predicted_head in zip(gold_heads, pred_heads):
			#	print("gold: {}, predicted: {}, equal: {}".format(
			#		gold_head, predicted_head, gold_head == predicted_head))
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
			#for gold_head, predicted_head in zip(gold_heads, pred_heads):
			#	print("gold: {}, predicted: {}, equal: {}".format(
			#		gold_head, predicted_head, gold_head == predicted_head))
			las += 100 * sum(
				gh == ph for gh, ph in zip(gold_heads, pred_heads)) / len(gold_heads)
		self.las_total = las / len(self.gold)

	def _TypedUas(self):
		"Computes Unlabeled Attachment Score for all dependency types."
		pass

	def	_TypedLasPrec(self):
		"""Computes Precision for all dependency types.

		For each relation X, precision computes the percentage of relations X
		in the system that are correct (correct / system)
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
		"Computes Recall for all dependency types."
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
			#logging.info("Recall of the model for label \"{}\" is {} / {} = {}".format(
			#	label, correct, gold, correct / gold)
			#)
			#logging.info("matches are: {}".format(match))
			self.typed_las_recall[label] = round((correct / gold), 2)
			#print("----------------------------------------------------------")

	def _TypedLasF1(self):
		"Computes F1 score for all dependency types."
		#F1 Score = 2*(Recall * Precision) / (Recall + Precision)
		f1 = lambda recall, precision: 2 * (recall * precision) / (recall + precision)
		assert self.typed_las_prec, "Cannot compute f1 without precision score!"
		assert self.typed_las_recall, "Cannote compute f1 without recall score!"
		self.typed_las_f1 = pd.DataFrame(
		 	[self.typed_las_prec, self.typed_las_recall],
		 	index=["precision", "recall"]).fillna(0).T

		self.typed_las_f1["f1"] = f1(self.typed_las_f1["recall"],
									 self.typed_las_f1["precision"]).fillna(0).round(3)

		print(self.typed_las_f1)

	def _EvaluateAll(self):
		"Runs all the evaluation metrics."
		self._UasTotal()
		self._LasTotal()
		self._TypedUas()
		self._TypedLasPrec()
		self._TypedLasRecall()
		self._TypedLasF1()

	def Evaluate(self, *args):
		metrics = ["uas_total", "las_total", "typed_uas", "typed_las_prec",
					"typed_las_recall", "typed_las_f1", "all"]
		assert any(metric in args for metric in metrics), "No valid metric!"

		if "all" in args:
			self._EvaluateAll()
		else:
			if "uas_total" in args:
				self._UasTotal()
			if "las_total" in args:
				self._LasTotal()
			if "typed_uas" in args:
				self._TypedUas()
			if  "typed_las_prec" in args:
				self._TypedLasPrec()
			if	"typed_las_recall" in args:
				self._TypedLasRecall()
			if "typed_las_f1" in args:
				self._TypedLasF1()
