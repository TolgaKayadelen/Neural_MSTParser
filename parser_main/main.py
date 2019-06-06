# -*- coding: utf-8 -*-

"""Main module to train and parse sentences with dependency parser."""

import argparse
import os
import pickle
import tempfile

from data.treebank import sentence_pb2
from parser.dependency_parser import DependencyParser
from util import common
from util import reader

import numpy as np
import matplotlib.pyplot as plt
from google.protobuf import text_format

import logging
logging.basicConfig(format='%(levelname)s : %(message)s', level=logging.DEBUG)


def _get_data(args):
    _TREEBANK_DIR = "data/UDv23"
    _TRAIN_DATA_DIR = os.path.join(_TREEBANK_DIR, args.language, "training")
    logging.info("Loading dataset")
    path = os.path.join(_TRAIN_DATA_DIR, "{}.protobuf".format(args.data))
    treebank = reader.ReadTreebankProto(path)
    training_data = [sentence for sentence in treebank.sentence]
    #print(text_format.MessageToString(training_data[0], as_utf8=True))
    return training_data

def _get_size(object):
    """Dump a pickle of object to accurately get the size."""
    tmp = tempfile.gettempdir()
    path = os.path.join(tmp, 'object.pkl')
    print(path)
    with open(path, 'wb') as f:
        pickle.dump(object, f)
    bytes = os.path.getsize(path)
    os.remove(path)
    print(path)
    gigabytes = bytes / 1e9
    return gigabytes

def train(args):
    training_data = map(common.ConnectSentenceNodes, _get_data(args))
    logging.info("Training Data Size {}".format(len(training_data)))
    #feature_opts = get_feature_opts(args.features)
    
    # Make the model
    model = DependencyParser(decoding="mst")
    if args.load:
        logging.info("Loading model from args.model...")
        #TODO: implement load and save methods for the dependency parser
        model.load(args.model)
    else:
        logging.info("Creating featureset..")
        model.MakeFeatures(training_data)
    #common.PPrintWeights(model.arc_perceptron.weights)
    totals = [len(model.arc_perceptron.weights[key]) for key in model.arc_perceptron.weights.keys()]
    print("Number of features {}".format(sum(totals)))
    #print("Memory used by model: {} GB.".format(_get_size(model)))
    
    model.Train(5, training_data, approx=7)

    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["train", "eval", "plot", "parse"], 
                        help="Choose action for the parser.")
    
    # Data args.
    parser.add_argument("--language", type=str, choices=["English", "Turkish"],
                        help="language")
    parser.add_argument("--data", type=str, help="The data to read (.protobuf)")
    
    # Model args.
    parser.add_argument("--decoder", type=str, choices=["mst", "eisner"],
                        default="mst", help="decoder to extract tree from scores.")
    parser.add_argument("--model", type=str, default="./model/model.pkl",
                        help="path to save the model to, or load the model from.")
    parser.add_argument("--load", type=bool,
                        help="Load a pretrained model, specify which one with --model.",
                        default=False)
    parser.add_argument('--features', nargs='+', default=[],
                        help='space separated list of additional features',
                        choices=['dist', 'surround', 'between'])
    
    # Training args
    parser.add_argument("--epochs", type=int, default=10,
                        help="Number of epochs to train on.")
    parser.add_argument("--out", type=str, help="dir to put the parsed output files.")
    
    args = parser.parse_args()
    
    if args.mode == "train":
        train(args)
    