# -*- coding: utf-8 -*-

"""Script to check which (if any) sentences occur in both training/dev training/test or test/dev sets."""

from __future__ import print_function

import os
import pandas as pd
from data.treebank import sentence_pb2
from data.treebank import treebank_pb2
from google.protobuf import text_format
from util import reader
import logging
logging.basicConfig(format='%(levelname)s : %(message)s', level=logging.DEBUG)

_TREEBANK_DIR = "data/UDv23/Turkish"
_TRAIN_DATA_DIR = os.path.join(_TREEBANK_DIR, "training")
_TEST_DATA_DIR = os.path.join(_TREEBANK_DIR, "test")

pd.set_option('display.max_rows', None)

def read_data():

  # get training data
  train_path = os.path.join(_TRAIN_DATA_DIR, "treebank_tr_imst_ud_train.pbtxt")
  train_treebank = reader.ReadTreebankTextProto(train_path)
  logging.info("Total sentences in train data {}".format(len(train_treebank.sentence)))

  # get dev data
  dev_path = os.path.join(_TEST_DATA_DIR, "treebank_tr_imst_ud_dev_fixed.pbtxt")
  dev_treebank = reader.ReadTreebankTextProto(dev_path)
  logging.info("Total sentences in dev data {}".format(len(dev_treebank.sentence)))

  # get test data
  test_path = os.path.join(_TEST_DATA_DIR, "treebank_tr_imst_ud_test_fixed.pbtxt")
  test_treebank = reader.ReadTreebankTextProto(test_path)
  logging.info("Total sentences in test data {}".format(len(test_treebank.sentence)))
  
  return train_treebank, dev_treebank, test_treebank
  #return dev_treebank, test_treebank
  

def cooccuring_sentences(set_one, set_two):
  
  set_one_list = [sentence.sent_id for sentence in set_one.sentence]
  set_one_sents = pd.Series(
    set_one_list,
    name="set one sentences"
  )
  set_two_list = [sentence.sent_id for sentence in set_two.sentence]
  set_two_sents = pd.Series(
    set_two_list,
    name="set two sentences"
  )  
  cooccuring_sents = pd.Series(
  [sent_id for sent_id in set_two_list if sent_id in set_one_list],
  name="cooccuring sentences"
  )
  
  print("set one: {}".format(set_one_sents))
  print("set two: {}".format(set_two_sents))
  print("cooccuring: {}".format(cooccuring_sents))
  print("Total cooccuring sentences: {}".format(len(cooccuring_sents)))
  

if __name__ == "__main__":
  train, dev, test = read_data()
  cooccuring_sentences(train, dev)