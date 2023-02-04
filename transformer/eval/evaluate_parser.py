"""A Parser eval script.

This script can be used when you have a treebank that is labeled with a labeler, and the head tags
are not predicted (gold). You can then pass this treebank through the lfp-parser using the predicted
dependency labels and predict head.

You need to have the original (gold) treebank with gold heads and labels and the version
of the same treebank where only the labels are predicted but heads were not.
"""

import os
import collections
import logging

import tensorflow as tf
import numpy as np

from eval import evaluate
from parser.utils import load_models
from util import reader, writer
from tagset.reader import LabelReader as label_reader
from data.treebank import treebank_pb2

logging.basicConfig(format='%(levelname)s : %(message)s', level=logging.INFO)

_DATA_DIR = "./transformer/eval/eval_data/bert-finetuned-20230204-023713-en-pud-final" # TODO

class ParserEval:
  """Parses a treebank that has already predicted labels and evals uas/las."""
  def __init__(self, parser_name, labeled_treebank_name, gold_treebank_name, language="tr"):
    # self.word_embeddings = load_models.load_word_embeddings()
    self.word_embeddings = load_models.load_i18n_embeddings(language=language)
    self.preprocessor = load_models.load_preprocessor(word_embedding_indexes=self.word_embeddings.token_to_index,
                                                      language=language,
                                                      features=["words", "dep_labels"],
                                                      embedding_type="conll")
    self.parser = load_models.load_parser(
      parser_name=parser_name,
      word_embeddings=self.word_embeddings,
      prep=self.preprocessor,
      language=language)
    self.test_dataset = self.get_dataset(labeled_treebank_name)
    self.gold_treebank = reader.ReadTreebankTextProto(os.path.join(_DATA_DIR, gold_treebank_name))
    self.labeled_treebank = reader.ReadTreebankTextProto(os.path.join(_DATA_DIR, labeled_treebank_name))
    self.stats = collections.Counter()
    self.label_reader = label_reader.get_labels("dep_labels", language)
    self.parsed_and_labeled_treebank = treebank_pb2.Treebank()
    self.language = language

  def get_dataset(self, treebank_name):
    """Returns a tf.data.dataset from a treebank."""
    _, _, test_dataset = load_models.load_data(
      preprocessor=self.preprocessor,
      test_data_dir=_DATA_DIR,
      test_treebank=treebank_name,
      test_batch_size=1
    )
    return test_dataset

  @property
  def gold_treebank_dict(self):
    return {s.sent_id: s for s in self.gold_treebank.sentence}

  @property
  def labeled_treebank_dict(self):
    return {s.sent_id: s for s in self.labeled_treebank.sentence}

  def eval(parser_name, treebank_name):
    test_results = parser.test(dataset=test_dataset)
    print("Test results ", test_results)

  def correct_predictions(self, *,
                          head_predictions=None,
                          correct_heads=None,
                          label_predictions=None,
                          correct_labels=None):
    """Computes correctly predicted edges and labels and relevant stats for them."""
    correct_predictions_dict = {"chp": None, "n_chp": None, "clp": None, "n_clp": None,
                                "n_clp_topk": None}

    correct_head_preds = (head_predictions == correct_heads)
    n_correct_head_preds = tf.math.reduce_sum(tf.cast(correct_head_preds, tf.int32))
    correct_predictions_dict["chp"] = correct_head_preds
    correct_predictions_dict["n_chp"] = n_correct_head_preds.numpy()


    correct_label_preds = (label_predictions == correct_labels)
    n_correct_label_preds = tf.math.reduce_sum(tf.cast(correct_label_preds, tf.int32))

    correct_predictions_dict["clp"] = correct_label_preds
    correct_predictions_dict["n_clp"] = n_correct_label_preds.numpy()

    return correct_predictions_dict

  def update_stats(self, correct_predictions_dict, n_words_in_batch):
    """Updates parsing stats at the end of each training or test step.

    The stats we keep track of are the following:
      n_tokens: total number of tokens in the data.
      n_chp: number of correctly predicted heads.
      n_clp: number of correctly predicted labels.
      n_chlp: number of tokens for which both head and label is correctly predicted.

    These are later used for computing eval metrics like UAS, LS, and LAS.
    """
    # print("words in batch ", n_words_in_batch)
    self.stats["n_tokens"] += n_words_in_batch

    # Correct head predictions.
    self.stats["n_chp"] += correct_predictions_dict["n_chp"]
    h = correct_predictions_dict["chp"]

    # Correct label predictions.
    self.stats["n_clp"] += correct_predictions_dict["n_clp"]
    l = correct_predictions_dict["clp"]

    # Tokens where both head and label predictions are correct.
    if h is not None and l is not None:
      if not len(h) == len(l):
        raise RuntimeError("Fatal: Mismatch in the number of heads and labels.")
      self.stats["n_chlp"] += np.sum(
        [1 for tok in zip(h, l) if tok[0] == True and tok[1] == True]
      )

  def _compute_metrics(self):
    """Computes metrics for uas, ls, and las."""
    _metrics = {}
    _metrics["uas"] = (self.stats["n_chp"] / self.stats["n_tokens"])
    _metrics["ls"] = (self.stats["n_clp"] / self.stats["n_tokens"])
    _metrics["las"] = (self.stats["n_chlp"] / self.stats["n_tokens"])
    return _metrics

  def evaluate(self):
    """Tests the performance of this parser on some dataset."""
    print("Testing on the test set..")
    head_accuracy = tf.keras.metrics.Accuracy()
    label_accuracy = tf.keras.metrics.Accuracy()

    # We traverse the test dataset not batch by batch, but example by example.
    for example in self.test_dataset:
      sent_id = tf.keras.backend.get_value(example["sent_id"][0][0]).decode()
      # We need to corresponding gold sentence so that we can get the corresponding gold labels
      # per sentence.
      gold_sentence = self.gold_treebank_dict[sent_id]

      # We also get the corresponding labeled sentence so that we update with head predictions later.
      labeled_sentence = self.labeled_treebank_dict[sent_id]

      # Making sure that the two sentences match.
      words_from_gold = [token.word for token in gold_sentence.token]
      words_from_labeled = [token.word for token in labeled_sentence.token]
      print("words from gold ", words_from_gold)
      print("words from labeled ", words_from_labeled)
      assert (words_from_labeled == words_from_gold), "Fatal: Mismatch in identified sentences!"
      # print(example["words"])
      # input()
      # print("example ", example)
      # Get the head scores and head preds from the parser.
      scores = self.parser.parse(example)
      head_scores = scores["edges"]
      head_preds = self.parser._flatten(tf.argmax(head_scores, 2))

      # Get the correct heads from the example/
      correct_heads = self.parser._flatten(example["heads"])

      # Sanity check that the gold heads in the labeled sentence are the same.
      c_heads_from_labeled_sent = [token.selected_head.address for token in labeled_sentence.token]
      c_heads_from_labeled_sent = tf.expand_dims(
        tf.convert_to_tensor(c_heads_from_labeled_sent, dtype=tf.int64), -1)
      assert all(tf.math.equal(correct_heads, c_heads_from_labeled_sent)), "Mismatch in correct heads!"
      # Update accuracy stats for heads.
      head_accuracy.update_state(correct_heads, head_preds)
      # print("head preds ", head_preds)
      # print("correct heads ", correct_heads)
      # input()

      # These labels were previously predicted by the fine tuned bert model.
      label_preds = self.parser._flatten(example["dep_labels"])

      # Make sure that the labeled preds in the example and the labeled_sentence from treebank match.
      l_preds_from_labeled_sent = [
        self.label_reader.vtoi(value=token.label) for token in labeled_sentence.token]
      l_preds_from_labeled_sent = tf.expand_dims(
        tf.convert_to_tensor(l_preds_from_labeled_sent, dtype=tf.int64), -1)
      print("labed preds from example ", label_preds)
      print("label preds from labeled sentencen ", l_preds_from_labeled_sent)
      assert all(tf.math.equal(label_preds, l_preds_from_labeled_sent)), "Mismatch in predicted labels!"
      # input()

      # Get the correct labels from the gold sentence.
      correct_label_names = [token.label for token in gold_sentence.token]
      correct_label_values = [self.label_reader.vtoi(value=label) for label in correct_label_names]
      correct_label_values = tf.cast(tf.expand_dims(correct_label_values, 1), tf.int64)
      # print("label preds ", label_preds)
      # print("correct labels names ", correct_label_names)
      # print("correct label values ", correct_label_values)
      # input()

      # Update correct label accuracy
      label_accuracy.update_state(correct_label_values, label_preds)

      correct_predictions_dict = self.correct_predictions(
        head_predictions=head_preds,
        correct_heads=correct_heads,
        label_predictions=label_preds,
        correct_labels=correct_label_values)

      # Update stats
      self.update_stats(correct_predictions_dict, example["words"].shape[1])

      # Add the sentence to the treebank with the predicted heads.
      self.add_to_parsed_treebank(head_preds, labeled_sentence)

    logging.info(f"Test stats: {self.stats}")
    results = self._compute_metrics()
    writer.write_proto_as_text(self.parsed_and_labeled_treebank,
                               os.path.join(_DATA_DIR, "parsed_and_labeled_test_treebank.pbtxt"))
    return results

  def add_to_parsed_treebank(self, head_preds, labeled_sentence):
    sentence = self.parsed_and_labeled_treebank.sentence.add()
    sentence.sent_id = labeled_sentence.sent_id
    # print("head preds ", head_preds)
    # print("correct heads in labeled sentence ",
    #       [token.selected_head.address for token in labeled_sentence.token])
    # input()
    for head_pred, token in zip(head_preds, labeled_sentence.token):
      # print("head pred ", head_pred)
      # print("token ", token)
      new_token = sentence.token.add()
      new_token.CopyFrom(token)
      new_token.selected_head.address = tf.keras.backend.get_value(head_pred[0])
      # print("new token ", new_token)
      # input()

if __name__ == "__main__":
  eval = ParserEval(
    parser_name = "en_lfp_predicted_head_gold_labels_only_20230204-060701", # TODO
    # parser_name="label_first_predicted_head_gold_labels_only",
    # This is the treebank where the labels are parsed with the Bert finetuned model (iter6)
    labeled_treebank_name = "labeled_test_treebank.pbtxt",
    gold_treebank_name = "gold_test_treebank.pbtxt",
    language="en" # TODO
  )
  results = eval.evaluate()
  gold_trb = reader.ReadTreebankTextProto(os.path.join(_DATA_DIR, "gold_test_treebank.pbtxt"))
  parsed_and_labeled_trb = reader.ReadTreebankTextProto(os.path.join(_DATA_DIR,
                                                                     "parsed_and_labeled_test_treebank.pbtxt"))
  evaluator = evaluate.Evaluator(gold_trb, parsed_and_labeled_trb,
                                 write_results=True,
                                 language="en", # TODO
                                 write_dir=_DATA_DIR)
  evaluator.evaluate("all")