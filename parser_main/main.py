# -*- coding: utf-8 -*-

"""Main module to train and parse sentences with dependency parser."""

import argparse
import os
import pickle
import tempfile
from copy import deepcopy
from data.treebank import sentence_pb2
from data.treebank import treebank_pb2
from parser.dependency_parser import DependencyParser
from util import common
from util import reader

import numpy as np
import matplotlib.pyplot as plt
from google.protobuf import text_format

import logging
logging.basicConfig(format='%(levelname)s : %(message)s', level=logging.DEBUG)

_TREEBANK_DIR = "data/UDv23"

def _get_data(args):
    """Function to retrieve training and dev data from args.
    
    Args:
        args = command line arguments.
    Returns:
        training_data: list, sentence_pb2.Sentence objects.
        dev_data: list, sentence_pb2.Sentence objects.
    """
    logging.info("Loading dataset")
    
    if not args.test_data:
        split = [float(args.split[0]), float(args.split[1])]
        assert split[0] + split[1] == 1, "Cannot split data!!"
        _TRAIN_DATA_DIR = os.path.join(_TREEBANK_DIR, args.language, "training")
        path = os.path.join(_TRAIN_DATA_DIR, "{}.pbtxt".format(args.train_data))
        treebank = reader.ReadTreebankTextProto(path)
        logging.info("Total sentences in treebank {}".format(len(treebank.sentence)))
        sentence_list = list(treebank.sentence)
        training_portion = int(split[0] * len(sentence_list))
        dev_portion = int(split[1] * len(sentence_list))
        training_data = sentence_list[:training_portion]
        dev_data = sentence_list[-dev_portion:]
    else:
        _TRAIN_DATA_DIR = os.path.join(_TREEBANK_DIR, args.language, "training")
        _TEST_DATA_DIR = os.path.join(_TREEBANK_DIR, args.language, "test")
        train_path = os.path.join(_TRAIN_DATA_DIR, "{}.pbtxt".format(args.train_data))
        train_treebank = reader.ReadTreebankTextProto(train_path)
        logging.info("Total sentences in train data {}".format(len(train_treebank.sentence)))
        training_data = list(train_treebank.sentence)
        test_path = os.path.join(_TEST_DATA_DIR, "{}.pbtxt".format(args.test_data))
        test_treebank = reader.ReadTreebankTextProto(test_path)
        logging.info("Total sentences in test data {}".format(len(test_treebank.sentence)))
        dev_data = list(test_treebank.sentence)
        
    return training_data, dev_data

def _get_size(object):
    """Dump a pickle of object to accurately get the size of the model.
    Args:
        object: the model. 
    Returns:
        gigabytes: the size of the model. 
    """
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
    """Trains a dependency parser.
    
    Args:
        args: the command line arguments.
    
    Saves a trained dependency parsing model. 
    """
    t,d = _get_data(args)
    training_data = map(common.ConnectSentenceNodes, t)
    training_data = map(common.ExtendSentence, training_data)
    logging.info("Training Data Size {}".format(len(training_data)))
    if len(d) > 0:
        dev_data = map(common.ConnectSentenceNodes, d)
        dev_data = map(common.ExtendSentence, dev_data)
        logging.info("Dev Data Size {}".format(len(d)))
    else:
        dev_data=None
    del t,d
   
    #feature_opts = get_feature_opts(args.features)
    
    # Make the model
    # TODO: add feature_opts to the model call.
    model = DependencyParser(decoding="mst")
    if args.load:
        logging.info("Loading model from {}".format(args.model))
        model.Load(args.model)
    else:
        logging.info("Creating featureset..")
        model.MakeFeatures(training_data)
    logging.info("Number of features: {}".format(model.arc_perceptron.feature_count))
    raw_input("Press any key to continue: ")
    #print("Memory used by model: {} GB.".format(_get_size(model)))
    
    
    # Train
    model.Train(args.epochs, training_data, dev_data=dev_data, approx=50)
    
    # Evaluate
    print("\n*******----------------------*******")
    logging.info("Start Evaluation on Dev Data..")
    logging.info("Weights not averaged..")
    if not dev_data:
        dev_data = training_data
        
    dev_acc_unavg = model._Evaluate(dev_data)
    logging.info("Accuracy before averaging weights on dev: {}".format(dev_acc_unavg))
    raw_input("Press any key to continue: ")
    
    # Average the weights and evaluate again
    logging.info("Averaging perpceptron weights and evaluating on dev..")
    unaveraged_weights = deepcopy(model.arc_perceptron.weights)
    accumulated_weights = deepcopy(model.arc_perceptron._accumulator)
    averaged_weights = model.arc_perceptron.AverageWeights(accumulated_weights)
    iters = model.arc_perceptron.iters
    
    # Sanity check to ensure averaging worked as intended.
    common.ValidateAveragedWeights(unaveraged_weights, 
                                   model.arc_perceptron._accumulator,
                                   averaged_weights,
                                   iters)
    model.arc_perceptron.weights = deepcopy(averaged_weights)
    logging.info("Evaluating after Averaged Weights..")
    dev_acc_avg = model._Evaluate(dev_data)
    raw_input("Press any key to continue: ")
    logging.info("Accuracy after averaging weights on dev: {}".format(dev_acc_avg))
    
    #Save the model.
    logging.info("Saving model to {}".format(args.model))
    raw_input("Press any key to continue: ")
    test_data_path = args.test_data if args.test_data else args.train_data
    model.Save(
        args.model, train_data_path=args.train_data, 
        test_data_path=test_data_path, nr_epochs=args.epochs, 
        accuracy=dict(
            test_unavg = round(dev_acc_unavg, 2),
            test_avg = round(dev_acc_avg, 2)
            )
        )
    

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
        
    # optionally save the parsed output to somewhere. 
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["train", "eval", "plot", "parse"], 
                        help="Choose action for the parser.")
    
    # Data args.
    parser.add_argument("--language", type=str, choices=["English", "Turkish"],
                        help="language")
    parser.add_argument("--train_data", type=str, help="The train data to read (.protobuf)")
    parser.add_argument("--test_data", type=str, help="The test data to read (.protobuf)")
    parser.add_argument("--split", '--list', action="append", help="Split training and test")
    
    # Model args.
    parser.add_argument("--decoder", type=str, choices=["mst", "eisner"],
                        default="mst", help="decoder to extract tree from scores.")
    parser.add_argument("--model", type=str, default="model.json",
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
    elif args.mode == "parse":
        parse(args)
    