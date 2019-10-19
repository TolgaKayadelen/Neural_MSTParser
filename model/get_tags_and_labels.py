# -*- coding: utf-8 -*-

"""Simply script to get all coarse and fine part of speech tags in the data.

Pass the data to read from the --data argument. The data can be a protobuf or
pbtxt.

"""

import argparse
import os

import pandas as pd
from util import reader
from collections import defaultdict
from data.treebank import sentence_pb2
from data.treebank import treebank_pb2
from google.protobuf import text_format
from google.protobuf import json_format

import logging
logging.basicConfig(format='%(levelname)s : %(message)s', level=logging.DEBUG)

def GetPosTags(data):
    sentence_list = []
    coarse_tags = defaultdict(int)
    fine_tags = defaultdict(int)
    labels = defaultdict(int)
    basename = os.path.basename(data)
    if basename.endswith(".pbtxt"):
        treebank = reader.ReadTreebankTextProto(data)
    elif basename.endswith(".protobuf"):
        treebank = reader.ReadTreebankProto(data)
    else:
        raise ValueError('Unsupported data type!!')

    if isinstance(treebank, treebank_pb2.Treebank):
        sentence_list = treebank.sentence
    for sentence in sentence_list:
        tokens = sentence.token
        for token in tokens:
            fine_tags[token.pos] += 1
            coarse_tags[token.category] += 1
            labels[token.label] += 1

    print("labels: {}\n".format(pd.Series(labels)))
    print("coarse_tags: {}\n".format(pd.Series(coarse_tags)))
    print("fine_tags: {}\n".format(pd.Series(fine_tags)))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, help="data to read",
        default="./data/UDv23/Turkish/training/treebank_tr_imst_ud_train.protobuf")
    args = parser.parse_args()
    GetPosTags(args.data)
