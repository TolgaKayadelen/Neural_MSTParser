
# -*- coding: utf-8 -*-

"""Module to train a dependency parser."""

import argparse
import os
import pickle
import random
import matplotlib.pyplot as plt
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


def plot(x, y, z, model_name, _type="parsing"):
  fig = plt.figure()
  ax = plt.axes()
  ax.plot(x,y, "-g", label="training", color="blue")
  ax.plot(x,z, "-g", label="test", color="red")
  #if not type(x) == list:
  #  plt.xlim(1, len(x))
  plt.ylim(min(z)-10,105)
  plt.title("Dependency {} performance on training and test data".format(_type))
  plt.xlabel("epochs")
  plt.ylabel("accuracy")
  plt.legend()
  plt.savefig(os.path.join("./model/pretrained/plot", "{}_{}.png".format(_type, model_name.strip(".json"))))
  logging.info("Saved plot to ./model/pretrained/plot/{}".format(model_name.strip(".json")))

def train_labeler(args):
  """Trains a dependency labeler.
  Args: 
    args: the command line arguments. 
  """
  training_data, test_data = _get_data(args)
  logging.info("Training data size {}".format(len(training_data)))
  if len(test_data) > 0:
    logging.info("Test data size {}".format(len(test_data)))
  
  labeler = DependencyLabeler(feature_file=args.labelfeatures)
  if not args.load:
    labeler.MakeFeatures(training_data)
  else:
    logging.info("Loading dependency labeler model from: {}..".format(args.labeler_model))
    labeler.Load(args.labeler_model)
  logging.info("Number of features for dependency labeler: {}".format(
        labeler.label_perceptron.feature_count
      ))
  
  # Train
  logging.info("learning_rate is {}".format(args.learning_rate))
  epochs, train_scores, test_scores = labeler.Train(args.epochs, training_data, test_data, args.learning_rate)
  plot(epochs, train_scores, test_scores, args.labeler_model, "labeling")
  
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
  
  model_dict = {
    "train_data": args.train_data,
    "train_data_size": len(training_data),
    "test_data": args.test_data if args.test_data else "split_%10",
    "test_data_size": len(test_data),
    "train_acc": labeler.label_accuracy_train, # TODO: find a way to log this.
    "test_acc_unavg": test_acc_unavg,
    "test_acc_avg": test_acc_avg,
    "epochs": args.epochs,
    "learning_rate": args.learning_rate,
    "features": args.labelfeatures,
    "feature_count": labeler.label_perceptron.feature_count
  }
  writer.write_model_output(model_dict, labeler=True)
  
  # Save the model.
  logging.info("Saving model to {}".format(args.labeler_model))
  test_data_path = args.test_data if args.test_data else args.train_data
  labeler.Save(
      args.labeler_model, train_data_path=args.train_data,
      test_data_path=test_data_path, labels=labeler.label_perceptron.labels,
      nr_epochs=args.epochs,
      test_accuracy=dict(
          test_unavg = round(test_acc_unavg, 2),
          test_avg = round(test_acc_avg, 2)
          )
      )

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

    # Make the model
    parser = DependencyParser(feature_file=args.arcfeatures, decoding="mst")
    if args.load:
        logging.info("Loading model from {}".format(args.parser_model))
        if args.arcfeatures:
          parser.Load(args.parser_model, feature_file=args.arcfeatures)
        else:
          parser.Load(args.parser_model)
    else:
        logging.info("Creating featureset..")
        parser.MakeFeatures(training_data)
    logging.info("Number of features for parser: {}".format(
      parser.arc_perceptron.feature_count))
    #print("Memory used by model: {} GB.".format(_get_size(model)))


    # Train
    epochs, train_scores, test_scores = parser.Train(args.epochs, training_data, test_data=test_data)
    plot(epochs, train_scores, test_scores, args.parser_model, "parsing")

    # Evaluate
    print("\n*******----------------------*******")
    logging.info("Start Evaluation on Eval Data..")
    logging.info("Weights not averaged..")
    if not test_data:
        test_data = training_data

    test_acc_unavg = parser._Evaluate(test_data)
    logging.info("Accuracy before averaging weights on eval: {}".format(test_acc_unavg))

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
    # TODO: add pruning here.
    logging.info("Evaluating after Averaged Weights..")
    #TODO: make model._Evaluate public.
    test_acc_avg = parser._Evaluate(test_data)
    logging.info("Accuracy after averaging weights on eval: {}".format(test_acc_avg))
    
    model_dict = {
      "train_data": args.train_data,
      "train_data_size": len(training_data),
      "test_data": args.test_data if args.test_data else "split_%10",
      "test_data_size": len(test_data),
      "train_acc": parser.train_acc,
      "test_acc_unavg": test_acc_unavg,
      "test_acc_avg": test_acc_avg,
      "epochs": args.epochs,
      "learning_rate": args.learning_rate,
      "features": args.arcfeatures,
      "feature_count": parser.arc_perceptron.feature_count
    }
    writer.write_model_output(model_dict, parser=True)

    # Save the model.
    logging.info("Saving model to {}".format(args.parser_model))
    test_data_path = args.test_data if args.test_data else args.train_data
    parser.Save(
        args.parser_model, train_data_path=args.train_data,
        test_data_path=test_data_path,
        nr_epochs=args.epochs,
        test_accuracy=dict(
            test_unavg = round(test_acc_unavg, 2),
            test_avg = round(test_acc_avg, 2)
            )
        )
