# -*- coding: utf-8 -*-

"""Module to evaluate sentences parsed with the dependency parser.

This module implements the following evaluation metrics:

UAS: Unlabeled attachment score.
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

import os
import argparse
import pandas as pd
import datetime
from data.treebank import sentence_pb2
from data.treebank import treebank_pb2
from proto import evaluation_pb2
from util import reader
from tagset.reader import LabelReader
from typing import Dict
from google.protobuf import text_format
from tagset.dep_labels import dep_label_enum_pb2 as dep_label_tags

import logging
logging.basicConfig(format='%(levelname)s : %(message)s', level=logging.INFO)

label_reader = LabelReader.get_labels("dep_labels")
# print("label to enum ", label_to_enum)

def _label_name_to_index(label_name):
  return label_reader.vtoi(label_name)

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
  write_results = args.write_results
  logging.info("Requested eval metrics {}".format(eval_metrics))
  evaluator = Evaluator(gold_data, test_data, write_results)
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
  def __init__(self, gold, test, write_results, model_name=None, write_dir=None):
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
    self.metrics = ["uas_total", "las_total", "typed_uas",
                    "label_accuracy",
                    "labeled_attachment_precision",
				            "labeled_attachment_recall",
                    "labeled_attachment_metrics", "all"]
    self.write_results = write_results
    self.model_name = model_name
    self.write_dir = write_dir
  
  @property
  def gold_and_test(self):
    """Ensures that the sentences in gold and test side are ordered
       by the same key (sent_id)."""
    #TODO(assert that the sentence ids are unique)
    gold_and_test = []
    for sent_id in self.gold:
      gold_and_test.append((self.gold[sent_id], self.test[sent_id]))
    # print("gold and test ", gold_and_test)
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
    evaluation.label_accuracy = self.label_accuracy
        
    for label in self.typed_uas.keys():
      evaluation.typed_uas.uas.add(
                              label=_label_name_to_index(label),
                              score=round(self.typed_uas[label],2)
                              )
    for label in self.labeled_attachment_prec.keys():
      evaluation.labeled_attachment_prec.prec.add(
                              label=_label_name_to_index(label),
                              score=round(self.labeled_attachment_prec[label],2)
                              )
    for label in self.labeled_attachment_recall.keys():
      evaluation.labeled_attachment_recall.recall.add(
                              label=_label_name_to_index(label),
                              score=round(self.labeled_attachment_recall[label], 2)
                              )
        
    for _, row in self.labeled_attachment_metrics.reset_index().iterrows():
      evaluation.labeled_attachment_metrics.labeled_attachment_metric.add(
                              label=_label_name_to_index(row["index"]),
                              count=int(row["count"]),
                              prec=row["labeled_attachment_precision"],
                              recall=row["labeled_attachment_recall"],
                              f1=row["labeled_attachment_f1"]
                              )
    
    return evaluation
    
  @property
  def label_counts(self):
    """Return a pd.Series of occurrences for each label in the data."""
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
  def label_prediction_metrics(self):
    """Returns label recall, precision and f1 metrics.
    
    Differently from labeled_attachment_metrics, these metrics do not take into
    account the head accuracy. They only consider whether the two tokens in the
    gold and test have the same label, whether or not they have the same head.
    """
    label_recall, label_precision, label_f1 = {}, {}, {}
    
    # Compute label recall
    for label in self.gold_labels:
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
          if test_tok.label == label:
            correct += 1
            match.append((sentence_idx, test_tok.index, test_tok.word))
      # Compute recall for this label if it exists in the gold data.
      # Otherwise it doesn't make sense to compute recall for a label.
      if gold:
        label_recall[label] = round((correct / gold), 2)
      else:
        pass
    
    # Compute label precision
    for label in self.test_labels:
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
          if gold_tok.label == label:
            correct += 1
            match.append((sentence_idx, gold_tok.index, gold_tok.word))
      label_precision[label] = round((correct / system), 2)
      
    cols = ["count",
            "label_precision",
            "label_recall",
            "label_f1"]
        
    label_prediction_metrics = pd.DataFrame(
      [label_precision, label_recall],
			index=["label_precision", "label_recall"]).fillna(0).T

    label_prediction_metrics["label_f1"] = f1(
      label_prediction_metrics["label_recall"],
      label_prediction_metrics["label_precision"]
      ).fillna(0).round(3)

    label_prediction_metrics["count"] = self.label_counts
    label_prediction_metrics = label_prediction_metrics[cols]
    label_prediction_metrics.fillna(0, inplace=True)

    return label_prediction_metrics
        

  @property
  def uas_total(self) -> float:
    """Computes the total Unlabeled Attachement Score of the parser."""
    uas = 0.0
    
    for gold_sent, test_sent in self.gold_and_test:
      # print(gold_sent.sent_id, test_sent.sent_id)
      # input()
      gold_heads = [tok.selected_head.address for tok in gold_sent.token[1:]]
      # print("gold heads ", gold_heads)
      pred_heads = [tok.selected_head.address for tok in test_sent.token[1:]]
      # print("test heads ", pred_heads)
      assert len(gold_heads) == len(pred_heads), "Tokenization mismatch!!"
      uas += sum(
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
      las +=  sum(
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
  def	labeled_attachment_prec(self):
    """Computes attachment precision for all dependency labels.
        
    For each relation X, precision computes the percentage of relations X
    in the system that are correct (correct / system). That is, it checks
    whether the X's that are found in the predictions also exist in the gold.
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
  def labeled_attachment_recall(self) -> Dict:
    """Computes attachment recall for all dependency labels.

    For each relation X, recall computes the percentage of relations that
    exists in the gold which are recovered by the system (correct / gold).
    
    This computation based on the both having the same labels and same heads.
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
  def labeled_attachment_metrics(self):
    """Computes attachment F1 score for all dependency labels"""
    cols = ["count",
            "labeled_attachment_precision",
            "labeled_attachment_recall",
            "labeled_attachment_f1"]
        
    labeled_attachment_metrics = pd.DataFrame(
      [self.labeled_attachment_prec, self.labeled_attachment_recall],
			index=["labeled_attachment_precision", "labeled_attachment_recall"]).fillna(0).T

    labeled_attachment_metrics["labeled_attachment_f1"] = f1(
      labeled_attachment_metrics["labeled_attachment_recall"],
      labeled_attachment_metrics["labeled_attachment_precision"]
      ).fillna(0).round(3)

    labeled_attachment_metrics["count"] = self.label_counts
    labeled_attachment_metrics = labeled_attachment_metrics[cols]
    labeled_attachment_metrics.fillna(0, inplace=True)
    return labeled_attachment_metrics

  def compute_label_accuracy(self, confusion_matrix=True):
    """Computes label accuracy and optionally creates a label confusion matrix.

    The label accuracy is based on the labels for the gold and the test tokens (without heads).

    Returns label confusion matrix and label accuracy.
    """
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
    assert len(gold_labels) == len(test_labels), "Length of gold and test labels must match!!"
    ls = 0
    for gold, test in zip(gold_labels, test_labels):
      if gold == test:
        ls += 1
    label_accuracy = ls / len(gold_labels)
    if not confusion_matrix:
      return label_accuracy
    confusion_matrix = pd.crosstab(gold_labels, test_labels, rownames=['Gold Labels'],
                                   colnames=['Test Labels'], margins=True)
    self.label_accuracy = label_accuracy
    return label_accuracy, confusion_matrix
    
  
  @property
  def evaluation_matrix(self):
    evaluation_matrix = pd.DataFrame()
    evaluation_matrix["count"] = self.labeled_attachment_metrics["count"]
    evaluation_matrix["uas"] = pd.Series(self.typed_uas).rename("uas",
                                         inplace=True).round(2)
    evaluation_matrix["l_a_prec"] = self.labeled_attachment_metrics[
                                                "labeled_attachment_precision"]
    evaluation_matrix["l_a_recall"] = self.labeled_attachment_metrics[
                                                "labeled_attachment_recall"]
    evaluation_matrix["l_a_f1"] = self.labeled_attachment_metrics[
                                                "labeled_attachment_f1"]
    return evaluation_matrix.fillna(0)

  def evaluate(self, *args):
    requested_metrics = args[0]
    assert any(metric in requested_metrics for metric in self.metrics
               ), "No valid metric!"
    results = {}
    if "all" in requested_metrics:
      results["uas_total"] = self.uas_total
      results["las_total"] = self.las_total
      label_accuracy, label_confusion_matrix = self.compute_label_accuracy(confusion_matrix=True)
      results["label_accuracy"] = label_accuracy
      results["label_confusion_matrix"] = label_confusion_matrix
      results["eval_matrix"] = self.evaluation_matrix
      results["eval_proto"] = self.evaluation
      results["label_prediction_metrics"] = self.label_prediction_metrics
      logging.info("Label Prediction Metrics")
      print(f"{self.label_prediction_metrics}\n\n\n")
      logging.info(f"Label Prediction Confusion Matrix:")
      print(f"{label_confusion_matrix}\n\n\n")
      logging.info("Labeled Attachment Evaluation Matrix:")
      print(f"{self.evaluation_matrix}\n\n\n")
      logging.info(f"Evaluation Proto:")
      print(self.evaluation)
    else:
      if "uas_total" in requested_metrics:
        results["uas_total"] = self.uas_total
      if "las_total" in requested_metrics:
        results["las_total"] = self.las_total
      if "typed_uas" in requested_metrics:
        results["typed_uas"] = pd.Series(self.typed_uas).rename("uas",
                                         inplace=True).round(2)
      if "label_accuracy" in requested_metrics:
        results["label_accuracy"] = self.compute_label_accuracy(confusion_matrix=False)
      if  "labeled_attachment_precision" in requested_metrics:
        results["labeled_attachment_precision"] = self.labeled_attachment_prec
      if "labeled_attachment_recall" in requested_metrics:
        results["labeled_attachment_recall"] = self.labeled_attachment_recall
      if "labeled_attachment_metrics" in requested_metrics:
        # this returns precision and recall as well
        results["labeled_attachment_metrics"] = self.labeled_attachment_metrics
        results["eval_proto"] = self.evaluation
    if self.write_results:
      if self.model_name:
        output_file_name = self.model_name +"_eval_results.txt"

      else:
        current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        output_file_name = f"{current_time}_eval_results"
      with open(os.path.join(self.write_dir, output_file_name), "w") as f:
        for key in results.keys():
          f.write(key)
          f.write("\n")
          f.write(str(results[key]))
          f.write("\n\n")
      logging.info(f"Eval results written to {output_file_name}")
    return results


def _read_file(path):
  with open(path, "r") as f:
    read = f.read()
  return read

def _read_parser_test_data(basename, eval_dir):
  path = os.path.join(eval_dir, "{}.pbtxt".format(basename))
  return text_format.Parse(_read_file(path), treebank_pb2.Treebank())


def main(args):
  gold_data = _read_parser_test_data(args.gold_data, args.eval_dir)
  test_data = _read_parser_test_data(args.test_data, args.eval_dir)
  model_name = os.path.basename(args.test_data)
  evaluator = Evaluator(gold_data, test_data, args.write_results, model_name,
                        write_dir=args.eval_dir)
  results = evaluator.evaluate(args.metrics)


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--test_data", type=str,
                      help="The test data to read (.protobuf)")
  parser.add_argument("--gold_data", type=str,
                      help="Gold data to use for evaluating a model.")
  parser.add_argument("--write_results", type=bool, default=True,
                      help="Whether to write a eval results to a file.")
  parser.add_argument("--metrics", nargs='*',
                      default=["uas_total", "las_total"],
                      help="Space separated list of metrics to evaluate.",
                      choices=["uas_total", "las_total", "typed_uas",
                               "typed_las_prec", "typed_las_recall", "typed_las_f1",
                               "all"])
  parser.add_argument("--eval_dir",
                      type=str,
                      default="./eval/eval_data",
                      help="Eval dir to read treebanks from.")

  args = parser.parse_args()
  main(args)
