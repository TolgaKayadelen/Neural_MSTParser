# -*- coding: utf-8 -*-

"""Module to evaluate sentences parsed with the dependency parser.

This module implements the following evaluation metrics:

UAS: Unlabeld attachment score.
LAS: Labeled attachment score.

We provide typed based precision, recall and F1 scores for LAS and type based
accuracy for UAS.

Usage:
bazel build parser_main:evaluate && 
bazel-bin/parser_main/evaluate \
--gold_data=./data/testdata/parser_main/eval_data_gold.pbtxt \
--test_data= ./data/testdata/parser_main/eval_data_test.pbtxt \
--metrics all \
--print_results=True
"""

import pandas as pd
from data.treebank import sentence_pb2
from data.treebank import treebank_pb2
from proto import evaluation_pb2
from util import common
from util import reader
from tagset.reader import LabelReader
from typing import Dict
from google.protobuf import text_format

import logging
logging.basicConfig(format='%(levelname)s : %(message)s', level=logging.INFO)

label_to_enum = LabelReader.get_labels("dep_labels").labels

def evaluate_parser(args, print_results=False):
  """Function to evaluate the dependency parser output on gold data.
  Args:
    command line arguments, values for requested metrics, gold data and
    test data path.
  Returns:
    eval_proto: evaluation_pb2.Evaluation, proto containing results for
    requested metrics.
  """
  gold_data = reader.ReadTreebankTextProto(args.gold_data)
  test_data = reader.ReadTreebankTextProto(args.test_data)
  eval_metrics = args.metrics
  logging.info("Requested eval metrics {}".format(eval_metrics))
  evaluator = Evaluator(gold_data, test_data)
  results = evaluator.Evaluate(eval_metrics)
  if print_results:
    for key in results:
      print(results[key])
  return results["eval_proto"]

get_labels = lambda sent: [token.label for token in sent.token[1:]]

same_head = (lambda t1, t2:
	t1.selected_head.address == t2.selected_head.address)

f1 = lambda recall, precision: 2 * (recall * precision) / (recall + precision)

class Evaluator:
  """The evaluator module."""
  def __init__(self, gold, test):
    """
    Initializes this evaluator with a gold and test treebank.
    Args:
      gold: treebank, the gold treebank.
      test: treebank, the treebank annotated by the system.
    """
    if isinstance(gold, treebank_pb2.Treebank):
      self.gold = dict((sent.sent_id, sent) for sent in gold.sentence)
    elif isinstance(gold, dict):
      self.gold = gold
    else:
      raise ValueError("Invalid value for self.gold!!")
    
    if isinstance(test, treebank_pb2.Treebank):
      self.test = dict((sent.sent_id, sent) for sent in test.sentence)
    elif isinstance(test, dict):
      self.test = test
    else:
      raise ValueError("Invalid value for self.test!!")
    self.metrics = ["uas_total", "las_total", "typed_uas", "typed_las_prec",
				            "typed_las_recall", "typed_las_f1", "all"]
  
  @property
  def gold_and_test(self):
    """Ensures that the sentences in gold and test side are ordered
       by the same key (sent_id)."""
    #TODO(assert that the sentence ids are unique)
    gold_and_test = []
    for sent_id in self.gold:
      gold_and_test.append((self.gold[sent_id], self.test[sent_id]))
    return gold_and_test
  
  @property
  def gold_sentences(self):
    return [sentences[0] for sentences in self.gold_and_test]
  
  @property
  def test_sentences(self):
    return [sentences[1] for sentences in self.gold_and_test]

  @property
  def gold_labels(self):
    return sorted(list(set().union(*map(get_labels, self.gold_sentences))))
  
  @property
  def test_labels(self):
    return sorted(list(set().union(*map(get_labels, self.test_sentences))))

  @property
  def evaluation(self):
    """Serializes all metrics to the evaluation proto."""
    evaluation = evaluation_pb2.Evaluation()
    evaluation.uas_total = self.uas_total
    evaluation.las_total = self.las_total
        
    for label in self.typed_uas.keys():
      evaluation.typed_uas.uas.add(
                              label=label_to_enum[label],
                              score=round(self.typed_uas[label],2)
                              )
    for label in self.typed_las_prec.keys():
      evaluation.typed_las_prec.prec.add(
                              label=label_to_enum[label],
                              score=round(self.typed_las_prec[label],2)
                              )
    for label in self.typed_las_recall.keys():
      evaluation.typed_las_recall.recall.add(
                              label=label_to_enum[label],
                              score=round(self.typed_las_recall[label], 2)
                              )
        
    for _, row in self.typed_las_f1.reset_index().iterrows():
      evaluation.typed_las_f1.f1.add(
        label=label_to_enum[row["index"]], count=int(row["count"]),
        prec=row["label_precision"],
        recall=row["label_recall"], f1=row["label_f1"]
        )
        
    return evaluation
    
  @property
  def label_counts(self):
    """Return a pd.Series of occurences for each label in the data."""
    assert self.gold_labels, "Tokens don't have any labels!!!"
    label_counts = {}
    for label in self.gold_labels:
      label_count = 0
      for sent in self.gold_sentences:
        for token in sent.token:
          if not token.label == label:
            continue
          label_count += 1
      label_counts[label] = label_count
    return pd.Series(label_counts).rename("count", inplace=True)

  @property
  def uas_total(self) -> float:
    """Computes the total Unlabeled Attachement Score of the parser."""
    uas = 0.0
    
    for gold_sent, test_sent in self.gold_and_test:
      gold_heads = [tok.selected_head.address for tok in gold_sent.token[1:]]
      pred_heads = [tok.selected_head.address for tok in test_sent.token[1:]]
      assert len(gold_heads) == len(pred_heads), "Tokenization mismatch!!"
      uas += 100 * sum(
          gh == ph for gh, ph in zip(gold_heads, pred_heads)) / len(gold_heads)
    return uas / len(self.gold)
        
  @property
  def las_total(self) -> float:
    """Computes Labeled Attachment Score."""
    las = 0.0
    assert len(self.test_labels) > 1, """Can't compute LAS:
                                         Test data doesn't have labels!"""
   
    for gold_sent, test_sent in self.gold_and_test:
      gold_heads = [
        (tok.selected_head.address, tok.label) for tok in gold_sent.token[1:]
      ]
      pred_heads = [
        (tok.selected_head.address, tok.label) for tok in test_sent.token[1:]
      ]
      assert len(gold_heads) == len(pred_heads), "Tokenization mismatch!!"
      las += 100 * sum(
          gh == ph for gh, ph in zip(gold_heads, pred_heads)) / len(gold_heads)
    
    return las / len(self.gold)
    
  @property
  def typed_uas(self) -> Dict:
    """Computes Unlabeled Attachment Score for all dependency labels.
    
    Returns:
      typed_uas: dict, [label:uas] for all dependency labels.
    """
    assert self.test_labels, "Cannot compute typed Uas without labels!!"
  
    typed_uas = {}
    for label in self.test_labels:
      correct = 0.0
      label_uas = 0.0
      sentence_idx = 0
      for gold_sent, test_sent in self.gold_and_test:
        sentence_idx += 1
        for gold_tok, test_tok in zip(gold_sent.token, test_sent.token):
          if not gold_tok.label == label:
            continue
          correct += 1.0
          if test_tok.selected_head.address == gold_tok.selected_head.address:
            label_uas += 1.0
      typed_uas[label] = label_uas / correct if correct else 0.0
    return typed_uas

  @property
  def	typed_las_prec(self):
    """Computes attachment precision for all dependency labels.
        
    For each relation X, precision computes the percentage of relations X
    in the system that are correct (correct / system). That is, it checks
    whether the X's that are found in the system also exists in the gold.
    """
    typed_las_prec = {}
    for label in self.test_labels:
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
      typed_las_prec[label] = round((correct / system), 2)
    return typed_las_prec
    
  @property
  def typed_las_recall(self) -> Dict:
    """Computes Recall for all dependency types.

    For each relation X, recall computes the percentage of relations that
    exists in the gold which are recovered by the system (correct / gold).
    """
    typed_las_recall = {}
    for label in self.gold_labels:
      # print("Computing recall for {}".format(label))
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
      # Compute recall for this label if it exists in the gold data.
      # Otherwise it doesn't make sense to compute recall for a label.
      if gold:
        typed_las_recall[label] = round((correct / gold), 2)
      else:
        pass
    return typed_las_recall
    
  @property 
  def typed_las_f1(self):
    """Computes F1 score for all dependency types"""
    cols = ["count", "label_precision", "label_recall", "label_f1"]
        
    typed_las_f1 = pd.DataFrame(
      [self.typed_las_prec, self.typed_las_recall],
			index=["label_precision", "label_recall"]).fillna(0).T

    typed_las_f1["label_f1"] = f1(
      typed_las_f1["label_recall"], typed_las_f1["label_precision"]
      ).fillna(0).round(3)

    typed_las_f1["count"] = self.label_counts
    typed_las_f1 = typed_las_f1[cols]
    typed_las_f1.fillna(0, inplace=True)
    return typed_las_f1
  
  @property
  def labels_conf_matrix(self):
    """Creates a labels confusion matrix."""
    def get_token_labels(sentence_list):
      labels = []
      for sent in sentence_list:
        for token in sent.token[1:]:
          labels.append(token.label)
      return labels
    gold_labels = pd.Series(get_token_labels(self.gold_sentences),
                                             name="gold_labels")
    test_labels = pd.Series(get_token_labels(self.test_sentences),
                                             name="test_labels")
    return pd.crosstab(gold_labels, test_labels, rownames=['Gold Labels'],
                       colnames=['Test Labels'], margins=True)
    print(self.labels_conf_matrix)
  
  @property
  def evaluation_matrix(self):
    evaluation_matrix = pd.DataFrame()
    evaluation_matrix["count"] = self.typed_las_f1["count"]
    evaluation_matrix["uas"] = pd.Series(self.typed_uas).rename("uas",
                                         inplace=True).round(2)
    evaluation_matrix["label_prec"] = self.typed_las_f1["label_precision"]
    evaluation_matrix["label_recall"] = self.typed_las_f1["label_recall"]
    evaluation_matrix["label_f1"] = self.typed_las_f1["label_f1"]
    return evaluation_matrix.fillna(0)

  def evaluate(self, *args):
    requested_metrics = args[0]
    assert any(metric in requested_metrics for metric in self.metrics
               ), "No valid metric!"
    results = {}
    if "all" in requested_metrics:
      results["uas_total"] = self.uas_total
      results["las_total"] = self.las_total
      results["eval_matrix"] = self.evaluation_matrix
      results["confusion_matrix"] = self.labels_conf_matrix
      results["eval_proto"] = self.evaluation
      logging.info("Eval Matrix:")
      print(self.evaluation_matrix)
      logging.info(f"Confusion Matrix:")
      print(self.labels_conf_matrix)
      logging.info(f"Eval Proto:")
      print(self.evaluation)
    else:
      if "uas_total" in requested_metrics:
        results["uas_total"] = self.uas_total
      if "las_total" in requested_metrics:
        results["las_total"] = self.las_total
      if "typed_uas" in requested_metrics:
        results["typed_uas"] = pd.Series(self.typed_uas).rename("uas",
                                         inplace=True).round(2)
      if  "typed_las_prec" in requested_metrics:
        results["typed_las_prec"] = self.typed_las_prec
      if	"typed_las_recall" in requested_metrics:
        results["typed_las_recall"] = self.typed_las_recall
      if "typed_las_f1" in requested_metrics:
        # this returns precision and recall as well
        results["typed_las_f1"] = self.typed_las_f1
        results["eval_proto"] = self.evaluation
    return results
