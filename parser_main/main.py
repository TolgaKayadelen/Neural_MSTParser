# -*- coding: utf-8 -*-

"""Main module to train and parse sentences with dependency parser."""

import argparse
import os
import re
import pickle


from util import reader

import numpy as np
import matplotlib.pyplot as plt
from google.protobuf import text_format

import logging
logging.basicConfig(format='%(levelname)s : %(message)s', level=logging.DEBUG)


def get_data(args):
    _TREEBANK_DIR = "data/UDv23"
    _TRAIN_DATA_DIR = os.path.join(_TREEBANK_DIR, args.language, "training")
    logging.info("Loading dataset from args.data")
    path = os.path.join(_TRAIN_DATA_DIR, "{}.protobuf".format(args.data))
    treebank = reader.ReadTreebankProto(path)
    print(text_format.MessageToString(treebank, as_utf8=True))



def train():
    pass
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["train", "eval", "plot", "parse"], 
                        help="Choose action for the parser.")
    
    
    # Data args.
    parser.add_argument("--language", type=str, choices=["English", "Turkish"],
                        help="language")
    parser.add_argument("--data", type=str, help="The data to read.")
    
    # Model args.
    parser.add_argument("--decoder", type=str, choices=["mst", "eisner"],
                        default="mst", help="decoder to extract tree from scores.")
    parser.add_argument("--model", type=str, default="./model/model.pkl",
                        help="path to save the model to, or load the model from.")
    parser.add_argument("--load", type=bool,
                        help="Load a pretrained model, specify which one with --model.")
    
    # Training args
    parser.add_argument("--epochs", type=int, default=10,
                        help="Number of epochs to train on.")
    parser.add_argument("--out", type=str, help="dir to put the parsed output files.")
    
    args = parser.parse_args()
    
    if args.mode == "train":
        get_data(args)
    