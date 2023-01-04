"""This script is used to evaluate the performance of rank based parsing system against the non-ranked based one."""

import os

import argparse
import datetime
import logging
import tensorflow as tf
import numpy as np

from data.treebank import sentence_pb2
from data.treebank import treebank_pb2
from eval import evaluate

from parser.utils import load_models
from proto import ranker_data_pb2
from ranker.preprocessing import ranker_preprocessor
from ranker.preprocessing import feature_extractor
from ranker.training import model as ranker_model
from tagset.dep_labels import dep_label_enum_pb2 as dep_label_tags
from typing import List
from util import reader, writer




Datapoint = ranker_data_pb2.RankerDatapoint
Sentence = sentence_pb2.Sentence

_CURRENT_TIME = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
_EVAL_PATH = "./eval/eval_data/ranker_based_parser"
_DATA_DIR = "./data/UDv29/dev/tr"

_feature_extractor = feature_extractor.FeatureExtractor()

logging.basicConfig(format='%(levelname)s : %(message)s', level=logging.INFO)

class RankerBasedDependencyParser:
  def __init__(self, word_embeddings, labeler_name, parser_name, ranker_name, treebank_name, k):
    self.preprocessor = load_models.load_preprocessor(word_embeddings=word_embeddings)
    self.labeler = load_models.load_labeler(labeler_name=labeler_name,
                                            prep=self.preprocessor)
    self.parser = load_models.load_parser(parser_name=parser_name,
                                          prep=self.preprocessor)
    self.ranker = ranker_model.Ranker(word_embeddings=word_embeddings,
                                      from_disk=True,
                                      name=ranker_name)
    self.ranker_prep = ranker_preprocessor.RankerPreprocessor(word_embeddings=word_embeddings)
    self.treebank = reader.ReadTreebankTextProto(os.path.join(_DATA_DIR, treebank_name))
    self.treebank_dataset =  self._get_dataset_from_treebank(treebank_name)
    self.eval_dir = os.path.join(_EVAL_PATH, _CURRENT_TIME)
    os.mkdir(self.eval_dir)
    self.k = k

  def _get_dataset_from_treebank(self, treebank_name):
    _, dev_treebank_dataset, test_treebank_dataset = load_models.load_data(
                                                        preprocessor=self.preprocessor,
                                                        dev_treebank=treebank_name,
                                                        test_treebank=None,
                                                        dev_batch_size=1,
                                                        test_batch_size=1)

    return dev_treebank_dataset

  def _get_topk_labels(self, example):
    scores = self.labeler.model({"words": example["words"],
                                  "pos": example["pos"], "morph": example["morph"]})
    label_scores = scores["labels"]
    top_scores, top_k_labels  = tf.math.top_k(label_scores, k=self.k)
    top_k_labels = self.labeler._flatten(top_k_labels, outer_dim=top_k_labels.shape[2])
    return tf.cast(top_k_labels, tf.int64)

  def _generate_datapoints(self, sentence:Sentence, example, top_k_labels):
    """
      :param sentence: sentence_pb2.Sentence.
      :param example: tf.data.Dataset version of the same sentence.
      :param top_k_labels: top_k_label hypothesis for each token in the sentence
      :return: List[Datapoint]
    """
    datapoints = []
    tokens = sentence.token
    # the embedding ids of words in the sentence
    word_ids = example["words"]
    # print("word ids ", word_ids)
    # print("words from example ", example["tokens"])
    # print("words from sentence ", [token.word for token in sentence.token])
    for i, word in enumerate(word_ids[0]):
      dp = ranker_data_pb2.RankerDatapoint()
      word_id = word_ids[:, i][0]
      token = tokens[i]
      word = token.word
      top_k_for_token = top_k_labels[i, :]
      # print("original data")
      # print("word ", word)
      # print("top k labels for word ", top_k_for_token)
      # input()
      dp.word_id = tf.keras.backend.get_value(word_id)
      dp.word = tf.keras.backend.get_value(word)
      dp.features.CopyFrom(_feature_extractor.get_features(token, sentence.token, n_prev=-2, n_next=2))
      for k in range(top_k_labels.shape[1]):
        hypothesis_k = tf.keras.backend.get_value(top_k_for_token[k])
        dp.hypotheses.add(
          label=dep_label_tags.Tag.Name(hypothesis_k),
          label_id=hypothesis_k,
          rank=k+1
        )
      # print("datapoint ", dp)
      # input()
      datapoints.append(dp)
    return datapoints

  def _rerank(self, datapoints):
    """
      :param ranker: a pretrained ranker.
      :param datapoints: a list of Datapoint objects.
      :return: reranked label predictions.
    """
    # We need to make datasets of batch size k so that each token and their k hypothesis are captured.
    reranked_labels = []
    dataset = self.ranker_prep.make_dataset_from_generator(datapoints=datapoints, batch_size=self.k)
    for example in dataset:
      # print("word ", example["word_string"][0])
      scores = self.ranker.label_ranker(example, training=False)
      # print("scores ", scores)
      top_scoring_hypothesis = np.argmax(scores)
      label = tf.keras.backend.get_value(example["hypo_label_id"][top_scoring_hypothesis])
      # print("top scoring hypothesis ", top_scoring_hypothesis)
      # print("top scoring label ", label)
      # label_name = tf.keras.backend.get_value(example["hypo_label"][top_scoring_hypothesis])
      # print("top scoring label name ", label_name)
      # input()
      reranked_labels.append(label)
    return tf.convert_to_tensor(reranked_labels)

  def parse_and_save(self):
    """
      :param labeler: the pretrained labeler.
      :param parser: the pretrained parser.
      :param ranker: the pretrained ranker.
      :param test_treebank_dataset: a tf.data.Dataset generated from the test treebank
      :param treebank: the treebank version of the generated dataset.
      :return:
    """

    gold_treebank_name = f"ranker_eval_{_CURRENT_TIME}_gold.pbtxt"
    parsed_trb_no_ranker_name = f"ranker_eval_{_CURRENT_TIME}_parsed_no_ranker.pbtxt"
    parsed_trb_with_ranker_name = f"ranker_eval_{_CURRENT_TIME}_parsed_with_ranker.pbtxt"

    gold_treebank = treebank_pb2.Treebank()
    parsed_trb_no_ranker = treebank_pb2.Treebank()
    parsed_trb_with_ranker = treebank_pb2.Treebank()


    sentences = {sentence.sent_id:sentence for sentence in self.treebank.sentence}

    for example in self.treebank_dataset:
      # sent_id = tf.keras.backend.get_value(example['sent_id'][0][0])
      # tokens = sentences[sent_id.decode("utf-8")]

      top_k_label_hypo = self._get_topk_labels(example)
      top1_labels = tf.expand_dims(top_k_label_hypo[:, 0], 0)
      # print("top k label hypo ", top_k_label_hypo)
      # print("top labels ", top1_labels)
      # print("top 1 labels from scores ", labeler._flatten(tf.argmax(label_scores, 2)))
      # input()

      sent_id = tf.keras.backend.get_value(example["sent_id"][0][0])
      # print("sent id ", sent_id)
      sentence_proto = sentences[sent_id.decode("utf-8")]
      datapoints = self._generate_datapoints(sentence_proto, example, top_k_label_hypo)
      reranked_labels = tf.expand_dims(self._rerank(datapoints), 0)
      # print("top labels ", top1_labels)
      # print("reranked labels ", reranked_labels)
      # print("gold  labels ", example["dep_labels"])
      # input()

      gold_sentence = gold_treebank.sentence.add()
      gold_sentence.CopyFrom(self._sentence_proto_from_example(example=example))
      # print("gold sentence ", gold_sentence)
      # input()

      parsed_w_labeler_labels = parsed_trb_no_ranker.sentence.add()
      parsed_w_labeler_labels.CopyFrom(self._parse(example, top1_labels))
      # print("sentence with labeler labels ", parsed_w_labeler_labels)
      # input()

      parsed_w_reranked_labels = parsed_trb_with_ranker.sentence.add()
      parsed_w_reranked_labels.CopyFrom(self._parse(example, reranked_labels))
      # print("sentence with reranked labels ", parsed_w_reranked_labels)
      # input()

    writer.write_proto_as_text(gold_treebank, os.path.join(self.eval_dir, gold_treebank_name))
    writer.write_proto_as_text(parsed_trb_no_ranker, os.path.join(self.eval_dir, parsed_trb_no_ranker_name))
    writer.write_proto_as_text(parsed_trb_with_ranker, os.path.join(self.eval_dir, parsed_trb_with_ranker_name))
    logging.info("All treebanks written to eval dir.")
    return gold_treebank_name, parsed_trb_no_ranker_name, parsed_trb_with_ranker_name


  def _parse(self, example, labels):
    """
    Parses an example with the parsing model.
    Uses the predicted labels provided with the labels argument, not the gold ones.

    :param example: A tf.data.Dataset example.
    :param labels: Predicted dependency labels.
    :return: A Sentence proto object.
    """

    gold_labels = example["dep_labels"]
    gold_heads = example["heads"]
    # print("gold_labels", gold_labels, "label predictions ", labels, "gold heads ", gold_heads)
    # input()
    scores = self.parser.model({"words": example["words"], "morph": example["morph"], "labels": labels},
                               training=False)
    edge_scores = scores["edges"]
    head_preds = tf.argmax(edge_scores, axis=2)
    # print("head preds ", head_preds)

    # First create the gold sentence proto from the example
    sentence_proto = self._sentence_proto_from_example(example=example, dep_labels=labels, heads=head_preds)
    return sentence_proto

  def _sentence_proto_from_example(self, *, example, dep_labels=None, heads=None):
    """
    Populates a Sentence object from a tf.data example.

    If dep_labels and/or heads are provided separately as arguments, those are used to populate the proto.

    :param dep_labels: a list of dep labels for each token.
    :return: a Sentence object
    """
    sentence = sentence_pb2.Sentence()
    if dep_labels is None:
      dep_labels = example["dep_labels"]
    if heads is None:
      heads = example["heads"]
    sent_id, tokens = (example["sent_id"], example["tokens"])
    sentence.sent_id = sent_id[0][0].numpy()
    index = 0
    for token, dep_label, head in zip(tokens[0], dep_labels[0], heads[0]):
      token = sentence.token.add(
        word=tf.keras.backend.get_value(token),
        label=dep_label_tags.Tag.Name(tf.keras.backend.get_value(dep_label)),
        index=index)
      token.selected_head.address=tf.keras.backend.get_value(head)
      index += 1
    return sentence

  def read_treebank(self, treebank_name):
    return reader.ReadTreebankTextProto(os.path.join(self.eval_dir, treebank_name))

def main(args):
  word_embeddings = load_models.load_word_embeddings()
  parser = RankerBasedDependencyParser(word_embeddings=word_embeddings,
                                       labeler_name=args.labeler_name,
                                       parser_name=args.parser_name,
                                       ranker_name=args.ranker_name,
                                       treebank_name=args.treebank_name,
                                       k=5)

  gold_name, parsed_no_ranker_name, parsed_w_ranker_name = parser.parse_and_save()
  gold = parser.read_treebank(gold_name)
  parsed_no_ranker = parser.read_treebank(parsed_no_ranker_name)
  parsed_w_ranker = parser.read_treebank(parsed_w_ranker_name)
  evaluator1 = evaluate.Evaluator(gold, parsed_no_ranker, write_results=True,
                                  model_name=f"no_ranker_{_CURRENT_TIME}_{args.treebank_name}",
                                  write_dir=parser.eval_dir)
  evaluator1.evaluate("all")
  evaluator2 = evaluate.Evaluator(gold, parsed_w_ranker, write_results=True,
                                  model_name=f"with_ranker_{_CURRENT_TIME}_{args.treebank_name}",
                                  write_dir=parser.eval_dir)
  evaluator2.evaluate("all")
  logging.info(f"Eval results written to {_CURRENT_TIME}_{args.treebank_name}")


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--parser_name",
                      type=str,
                      default="label_first_gold_morph_and_labels",
                      help="Pretrained parser to load.")
  parser.add_argument("--labeler_name",
                      type=str,
                      default="bilstm_labeler_topk",
                      help="Pretrained labeler to load.")
  parser.add_argument("--ranker_name",
                      type=str,
                      default="k20-edge-only-mse-error",
                      help="Pretrained ranker model name to load.")
  parser.add_argument("--treebank_name",
                      type=str,
                      default="tr_boun-ud-dev.pbtxt")
  args = parser.parse_args()
  main(args)