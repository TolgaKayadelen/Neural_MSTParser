# -*- coding: utf-8 -*-

"""Module to parse sentences with dependency parser."""


import argparse
import os
from copy import deepcopy
from data.treebank import sentence_pb2
from data.treebank import treebank_pb2
from parser.dependency_parser import DependencyParser
from util import common
from util import reader

from google.protobuf import text_format

import logging
logging.basicConfig(format='%(levelname)s : %(message)s', level=logging.DEBUG)

_TREEBANK_DIR = "data/UDv23"


def parse(args):
    """Parses a set of sentences with the loaded parsing model. 
    
    Args:
        args: the command line arguments.   
    Returns:
        treebank: treebank_pb2.Treebank, a treebank of sentences parsed with the loaded model.
    """
    languages = {"Turkish": 1, "English": 2}
    output_treebank = treebank_pb2.Treebank()
    output_treebank.language = languages[args.language]
    
    # sanity check on the command line arguments
    assert args.train_data is None, "Cannot train in parse mode!"
    assert args.load is not False, "Specify which model to use!"
    if args.model == "model.json":
        logging.info("No model specified, using the default one!")
    
    # load the model.
    model = DependencyParser(decoding="mst")
    model.Load(args.model)
    logging.info("Loading model from {}".format(args.model))
    logging.info("Number of features: {}".format(model.arc_perceptron.feature_count))
    raw_input("Press any key to continue: ")
    
    # read the sentences to parse into proto
    _TEST_DATA_DIR = os.path.join(_TREEBANK_DIR, args.language, "test")
    test_path = os.path.join(_TEST_DATA_DIR, "{}.pbtxt".format(args.test_data))
    test_treebank = reader.ReadTreebankTextProto(test_path)
    logging.info("Total sentences in test data {}".format(len(test_treebank.sentence)))
    test_data = list(test_treebank.sentence)
    print(text_format.MessageToString(test_data[0], as_utf8=True))
    
    # parse sentences with the loaded model.
    test_data = map(common.ConnectSentenceNodes, test_data)
    test_data = map(common.ExtendSentence, test_data)
    for sentence in test_data:
        parsed, predicted_heads = model.Parse(sentence)
        output_treebank.sentence.extend([parsed])
    print("----------")
    print("Treebank")
    print(text_format.MessageToString(output_treebank, as_utf8=True))
    
    return output_treebank
        