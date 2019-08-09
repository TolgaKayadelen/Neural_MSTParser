# -*- coding: utf-8 -*-

"""Module to evaluate sentences parsed with the dependency parser. 

This module implements the following evaluation metrics: 

UAS: Unlabeld attachment score.
LAS: Labeled attachment score.

We provide typed based precision, recall and F1 scores for LAS and type based
accuracy for UAS. 

"""
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
		self.uas_total = 0.0
		self.las_total = 0.0
		self.typed_uas = {}
		self.typed_las_prec = {}
		self.typed_las_recall = {}
		self.typed_las_f1 = {} 	
	
	def _UasTotal(self):
		"Computes the total Unlabeled Attachement Score of the parser."
		uas = 0.0
		for gold_sent, test_sent in zip(self.gold, self.test):
			gold_heads = [token.selected_head.address for token in gold_sent.token[1:]]
			pred_heads = [token.selected_head.address for token in test_sent.token[1:]]
			assert len(gold_heads) == len(pred_heads), "Tokenization mismatch!!"
			#for gold_head, predicted_head in zip(gold_heads, pred_heads):
			#	print("gold: {}, predicted: {}, equal: {}".format(
			#		gold_head, predicted_head, gold_head == predicted_head))
			#print("---")
			uas += 100 * sum(
					gh == ph for gh, ph in zip(gold_heads, pred_heads)) / len(gold_heads)
		self.uas_total = uas / len(self.gold)
		return self.uas_total
			
	
	def _LasTotal(self):
		"Computes the total Labeled Attachment Score of the parser."
		pass
	
	def _TypedUas(self):
		"Computes Unlabeled Attachment Score for all dependency types."
		pass
	
	def	_TypedLasPrec(self):
		"Computes Precision for all dependency types."
		pass
	
	def _TypedLasRecall(self):
		"Computes Recall for all dependency types."
		pass
	
	def _TypedLasF1(self):
		"Computes F1 score for all dependency types."
		pass
	
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
		
	
	
		
	


