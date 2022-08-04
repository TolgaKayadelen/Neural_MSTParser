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

def get_tags_and_labels(data):
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
            coarse_tags[token.category] += 1
            fine_tags[token.pos] += 1
            labels[token.label] += 1

    # labels = {v: k for k, v in labels.items()}
    print("labels: {}\n".format(pd.Series(labels).sort_values()))
    # print(coarse_tags)
    # coarse_tags = {v: k for k, v in coarse_tags.items()}
    print("coarse_tags: {}\n".format(pd.Series(coarse_tags).sort_values()))
    # fine_tags = {v: k for k, v in fine_tags.items()}
    print("fine_tags: {}\n".format(pd.Series(fine_tags).sort_values()))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, help="data to read",
        default="./data/UDv29/test/tr/tr_boun-ud-test.pbtxt")
    args = parser.parse_args()
    get_tags_and_labels(args.data)

