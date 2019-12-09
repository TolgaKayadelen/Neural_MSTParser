
# -*- coding: utf-8 -*-

"""Module to train a dependency parser."""

import argparse
import os
import pickle
import random
import tempfile
from copy import deepcopy
from data.treebank import sentence_pb2
from data.treebank import treebank_pb2
from parser.dependency_parser import DependencyParser
from parser.dependency_labeler import DependencyLabeler
from parser_main.parse import parse
from util import common
from util import reader
from util import writer

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
        test_portion = int(split[1] * len(sentence_list))
        training_data = sentence_list[:training_portion]
        test_data = sentence_list[-test_portion:]
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
        test_data = list(test_treebank.sentence)

    return training_data, test_data

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


def train_labeler(args):
  """Trains a dependency labeler.
  Args: 
    args: the command line arguments. 
  """
  training_data, test_data = _get_data(args)
  logging.info("Training data size {}".format(len(training_data)))
  if len(test_data) > 0:
    logging.info("Test data size {}".format(len(test_data)))
  labeler = DependencyLabeler(args.labelfeatures)
  labeler.MakeFeatures(training_data)
  logging.info("Number of features for dependency labeler {}".format(
    labeler.label_perceptron.feature_count
  ))
  
  # Train
  logging.info("learning_rate is {}".format(args.learning_rate))
  labeler.Train(args.epochs, training_data, test_data, args.learning_rate)
  
  # Evaluate
  print("\n*******----------------------*******")
  logging.info("Start Evaluation on Eval Data")
  logging.info("Weights not averaged")
  if not test_data:
    test_data = training_data
    eval_type = "train"
  else:
    eval_type = "test"
  test_acc_unavg = labeler.Evaluate(test_data, eval_type=eval_type)
  logging.info("Accuracy before averaging weights on eval: {}".format(test_acc_unavg))
  
  # Average the weights and evaluate again.
  logging.info("Averaging label perceptron weights and evaluating on eval data..")
  unaveraged_weights = deepcopy(labeler.label_perceptron.label_weights)
  labeler.label_perceptron.FinalizeAccumulator()
  labeler.label_perceptron.AverageClassWeights()
  
  common.ValidateAveragingLabelPer(
    unaveraged_weights,
    labeler.label_perceptron._label_accumulator,
    labeler.label_perceptron.label_weights,
    labeler.label_perceptron.iters
  )
  
  test_acc_avg = labeler.Evaluate(test_data, eval_type=eval_type)
  logging.info("Accuracy after averaging weights on eval {}".format(test_acc_avg))
  
  model_output = {
    "train_data": args.train_data,
    "train_data_size": len(training_data),
    "test_data": args.test_data if args.test_data else "split_%10",
    "test_data_size": len(test_data),
    "train_acc": None, # TODO: find a way to log this.
    "test_acc_unavg": test_acc_unavg,
    "test_acc_avg": test_acc_avg,
    "epochs": args.epochs,
    "learning_rate": args.learning_rate,
    "features": args.labelfeatures,
    "feature_count": labeler.label_perceptron.feature_count
  }
  writer.write_model_output(model_output, labeler=True)

def train_parser(args):
    """Trains a dependency parser.

    Args:
        args: the command line arguments.

    Saves a trained dependency parsing model.
    """
    tr,te = _get_data(args)
    training_data = map(common.ConnectSentenceNodes, tr)
    training_data = map(common.ExtendSentence, training_data)
    logging.info("Training Data Size {}".format(len(training_data)))
    if len(te) > 0:
        test_data = map(common.ConnectSentenceNodes, te)
        test_data = map(common.ExtendSentence, test_data)
        logging.info("Test Data Size {}".format(len(te)))
    else:
        test_data=None
    del tr,te

    #feature_opts = get_feature_opts(args.features)

    # Make the model
    # TODO: add feature_opts to the model call.
    parser = DependencyParser(feature_file=args.arcfeatures, decoding="mst")
    if args.load:
        logging.info("Loading model from {}".format(args.model))
        parser.Load(args.model)
    else:
        logging.info("Creating featureset..")
        parser.MakeFeatures(training_data)
    logging.info("Number of features for parser: {}".format(
      parser.arc_perceptron.feature_count))
    raw_input("Press any key to continue: ")
    #print("Memory used by model: {} GB.".format(_get_size(model)))


    # Train
    parser.Train(args.epochs, training_data, test_data=test_data, approx=50)

    # Evaluate
    print("\n*******----------------------*******")
    logging.info("Start Evaluation on Eval Data..")
    logging.info("Weights not averaged..")
    if not test_data:
        test_data = training_data

    test_acc_unavg = parser._Evaluate(test_data)
    logging.info("Accuracy before averaging weights on eval: {}".format(test_acc_unavg))
    raw_input("Press any key to continue: ")

    # Average the weights and evaluate again
    logging.info("Averaging arc perceptron weights and evaluating on eval..")
    unaveraged_weights = deepcopy(parser.arc_perceptron.weights)
    accumulated_weights = deepcopy(parser.arc_perceptron._accumulator)
    averaged_weights = parser.arc_perceptron.AverageWeights(accumulated_weights)
    iters = parser.arc_perceptron.iters

    # Sanity check to ensure averaging worked as intended.
    common.ValidateAveragingArcPer(
      unaveraged_weights,
      parser.arc_perceptron._accumulator,
      averaged_weights,
      iters
    )
    parser.arc_perceptron.weights = deepcopy(averaged_weights)
    logging.info("Evaluating after Averaged Weights..")
    #TODO: make model._Evaluate public.
    test_acc_avg = parser._Evaluate(test_data)
    raw_input("Press any key to continue: ")
    logging.info("Accuracy after averaging weights on eval: {}".format(test_acc_avg))

    # Save the model.
    logging.info("Saving model to {}".format(args.model))
    raw_input("Press any key to continue: ")
    test_data_path = args.test_data if args.test_data else args.train_data
    parser.Save(
        args.model, train_data_path=args.train_data,
        test_data_path=test_data_path, nr_epochs=args.epochs,
        accuracy=dict(
            test_unavg = round(test_acc_unavg, 2),
            test_avg = round(test_acc_avg, 2)
            )
        )
