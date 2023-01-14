import os
import datasets
import argparse
import logging
import tensorflow as tf
import numpy as np

from transformers import AutoTokenizer, TFAutoModelForTokenClassification

from data.treebank import sentence_pb2
from data.treebank import treebank_pb2
from eval import evaluate
from util import reader, writer
from tagset import reader as tagset_reader

model_path = "./transformer/hf/pretrained"
_data_dir = "./data/UDv29/test/tr"
_test_data_pbtxt = "tr_boun-ud-test.pbtxt"

class BertInferencePipeline:
  def __init__(self, model_path, tokenizer_name="bert-base-multilingual-uncased"):
    self.model = TFAutoModelForTokenClassification.from_pretrained(model_path)
    self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)
    self.n_tokens  = 0.0
    self.n_accurate_labels = 0.0
    self.accuracy = 0.0
    self.label_reader = tagset_reader.LabelReader("dep_labels")
    self.parsed_treebank = treebank_pb2.Treebank()

  def _align_labels(self, *, unaligned_labels, word_ids, gold_labels):
    prev_word_index=None
    aligned_labels = []
    for i, word_idx in enumerate(word_ids):
      # print("prev word index ", prev_word_index)
      # input()
      # print("word idx ", word_idx)
      # input()
      if word_idx is None:
        prev_word_index=word_idx
        continue
      elif word_idx != prev_word_index:
        label = unaligned_labels[0][i].numpy()
        # print("label ", label)
        aligned_labels.append(label)
        prev_word_index = word_idx

    print("aligned labels ", aligned_labels)
    assert len(aligned_labels)  == len(gold_labels), "Mismatch in number of labels"
    # input()
    return aligned_labels


  def predict_labels(self):
    # test_dataset = self.read_dataset()
    # print("test dataset ", test_dataset)
    test_treebank, test_dataset = self.read_dataset()
    test_treebank_dict = {s.sent_id: s for s in test_treebank.sentence}
    # for sentence in test_treebank.sentence:
    for example in test_dataset:
      sent_id = example["sent_id"]
      sentence = test_treebank_dict[sent_id]
      tokens_from_pbtxt = [t.word for t in sentence.token[1:]]
      tokens_from_example = example["tokens"]
      assert (tokens_from_example == tokens_from_pbtxt), "Mistmatch in tokens on test data files!"
      print("tokens from pbtxt ", tokens_from_pbtxt)
      print("tokes from example ", example["tokens"])
      # input()
      # tokenized = self.tokenizer(example["tokens"], is_split_into_words=True, return_tensors='tf')
      tokenized = self.tokenizer(tokens_from_pbtxt, is_split_into_words=True, return_tensors='tf')
      # print("tokenized ", tokenized)
      # gold_labels = example["dep_labels"]
      gold_label_names = [t.label for t in sentence.token[1:]]
      gold_labels_pbtxt = [self.label_reader.vtoi(label) for label in gold_label_names]
      print("gold labels from pbtxt", gold_labels_pbtxt)
      print("gold labels from example ", example["dep_labels"])
      assert (gold_labels_pbtxt == example["dep_labels"]), "Mistmatch in gold_labels on test data files!"
      # input()
      word_ids = tokenized.word_ids()
      print("word ids ", word_ids)
      outputs = self.model(tokenized["input_ids"])
      # print("outputs ", outputs)
      unaligned_labels = tf.argmax(outputs.logits, -1)
      print("unaligned labels ", unaligned_labels)
      # input()
      label_preds = self._align_labels(unaligned_labels=unaligned_labels, word_ids=word_ids,
                                       gold_labels=gold_labels_pbtxt)
      self._update_accuracy(label_preds=label_preds, gold_labels=gold_labels_pbtxt)
      self._add_sentence_to_parsed_treebank(sentence, label_preds)
    gold_path = os.path.join("./transformer/eval_data", "gold_treebank.pbtxt")
    parsed_path = os.path.join("./transformer/eval_data", "parsed_treebank.pbtxt")
    self.write_treebank(test_treebank, path=gold_path)
    self.write_treebank(self.parsed_treebank, path=parsed_path)
    logging.info("All treebanks written to /transformer/eval_data")
    return gold_path, parsed_path

  def _update_accuracy(self, *, label_preds, gold_labels):
    label_preds = np.array(label_preds)
    gold_labels = np.array(gold_labels)
    self.n_tokens += len(gold_labels)
    self.n_accurate_labels += np.sum(label_preds == gold_labels)
    print("n tokens ", self.n_tokens)
    print("n accurate labels ", self.n_accurate_labels)
    # input()
    self.accuracy = self.n_accurate_labels / self.n_tokens
    print("accuracy ", self.accuracy)
    # input()

  def _add_sentence_to_parsed_treebank(self, sentence, labels):
    parsed_sentence = sentence_pb2.Sentence()
    parsed_sentence.sent_id = sentence.sent_id
    root_token = parsed_sentence.token.add()
    root_token.CopyFrom(sentence.token[0])
    # print("parsed sentence ", parsed_sentence)
    # input()
    assert len(labels) == len(sentence.token[1:]), "Mismatch in number of tokens and dep labels"
    for label, token in zip(labels, sentence.token[1:]):
      # print(label, token)
      # input()
      # First copy the token from the gold sentence
      parsed_sent_token = parsed_sentence.token.add()
      parsed_sent_token.CopyFrom(token)
      # Then change the label with the predicted label
      token.label = self.label_reader.itov(label)

    # print("parsed sentence ", parsed_sentence)
    # print("sentence ", sentence)
    # input()
    treebank_sentence = self.parsed_treebank.sentence.add()
    treebank_sentence.CopyFrom(parsed_sentence)

  @staticmethod
  def write_treebank(treebank, path):
    writer.write_proto_as_text(treebank, path)

  @staticmethod
  def read_dataset():
    dataset = datasets.load_dataset("./transformer/hf/dataset/hf_data_loader.py")
    test_dataset = dataset["test"]
    test_treebank = reader.ReadTreebankTextProto(os.path.join(_data_dir, _test_data_pbtxt))
    # test_treebank_dict = {s.sent_id: s for s in test_treebank.sentence}
    assert len(test_dataset) == len(test_treebank.sentence), "Mismatch in number of sentences!"
    return test_treebank, test_dataset



if __name__ == "__main__":
  pipeline = BertInferencePipeline(model_path="./transformer/hf/pretrained",
                                   tokenizer_name="bert-base-multilingual-uncased")
  gold_trb_path, parsed_trb_path = pipeline.predict_labels()
  gold = reader.ReadTreebankTextProto(gold_trb_path)
  parsed = reader.ReadTreebankTextProto(parsed_trb_path)
  evaluator = evaluate.Evaluator(gold, parsed,
                                 write_results=True,
                                 write_dir="./transformer/eval_data")
  evaluator.evaluate("all")
