# -*- coding: utf-8 -*-
"""
Simply script to get all coarse and fine part of speech tags in the data.
Pass the data to read from the --data argument. The data can be a protobuf or
pbtxt.

"""
import argparse
import os
import sys
import pandas as pd
from util import reader
from collections import defaultdict
from data.treebank import treebank_pb2


import logging
logging.basicConfig(format='%(levelname)s : %(message)s', level=logging.DEBUG)

def tags_to_ids(data, tagset=None, dict=False, print_dict=False):
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
    if dict:
        coarse, fine, labels = coarse_tags.keys(), fine_tags.keys(), labels.keys()
        coarse_tags = {key:index for index,key in enumerate(coarse)}
        fine_tags = {key:index for index,key in enumerate(fine)}
        labels = {key:index for index,key in enumerate(labels)}
        if print_dict:
            print("COARSE TAGS")
            for k, v in coarse_tags.items():
                print(f"{k} = {v};")
            print("\n\n")
            print("FINE TAGS")
            keys = list(fine_tags.keys())
            keys.sort()
            index = 0
            for k in keys:
                print(f"{k} = {index};")
                index += 1
            print("\n\n")
            print("LABELS")
            keys = list(labels.keys())
            keys.sort()
            index=0
            for k in keys:
                print(f"{k} = {index};")
                index+=1
        if tagset == "pos":
            return fine_tags
        elif tagset == "coarse":
            return coarse
        elif tagset == "dep_labels":
            return labels
        elif tagset == "all":
            return fine_tags, coarse, labels
        else:
            sys.exit("No tagset requested!")
    else:
        # labels = {v: k for k, v in labels.items()}
        print("labels: {}\n".format(pd.Series(labels).sort_values()))
        print("labels: {}\n".format(pd.Series(labels).sort_index()))
        # print(coarse_tags)
        # coarse_tags = {v: k for k, v in coarse_tags.items()}
        print("coarse_tags: {}\n".format(pd.Series(coarse_tags).sort_values()))
        print("coarse_tags: {}\n".format(pd.Series(coarse_tags).sort_index()))
        # fine_tags = {v: k for k, v in fine_tags.items()}
        print("fine_tags: {}\n".format(pd.Series(fine_tags).sort_values()))
        print("fine_tags: {}\n".format(pd.Series(fine_tags).sort_index()))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data",
                        type=str,
                        help="data to read",
                        default="./data/UDv29/train/tr/tr_boun-ud-train.pbtxt")
    args = parser.parse_args()
    fine_tags, coarse, labels = tags_to_ids(args.data, tagset="all", dict=True)