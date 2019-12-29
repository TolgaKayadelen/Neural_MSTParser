# -*- coding: utf-8 -*-

"""Module to label sentences with dependency labeler."""

import argparse
import os
import sys
from copy import deepcopy
from data.treebank import sentence_pb2
from data.treebank import treebank_pb2
from parser.dependency_labeler import DependencyLabeler
from util import common
from util import reader
from util.writer import write_proto_as_text

from google.protobuf import text_format

import logging
logging.basicConfig(format='%(levelname)s : %(message)s', level=logging.DEBUG)

_TREEBANK_DIR = "data/UDv23"

def label(path, **kwargs):
  """Label dependency arcs in a treebank.
  Args:
    path: str, path to treebank where dependency arcs are not labeled.
  Expected kwargs:
    model: the model to label the sentences with.
    treebank_name: the name of the treebank we are parsing.
    
  Returns:
    treebank where dependency arcs are labeled. 
  """
  languages = {"Turkish": 1, "English": 2}
  _TEST_DATA_DIR = os.path.join(_TREEBANK_DIR, kwargs["language"], "test")
  treebank_path = os.path.join(_TEST_DATA_DIR, "{}.pbtxt".format(path))
  treebank = reader.ReadTreebankTextProto(treebank_path)
  sentences = list(treebank.sentence)
  logging.info("Total number of sentences {}".format(len(sentences)))
  labeler = DependencyLabeler()
  
  if kwargs["model"]:
    labeler.Load(kwargs["model"])
    logging.info("Loading model from {}".format(kwargs["model"]))
    logging.info("Number of features: {}".format(labeler.label_perceptron.feature_count))
  else:
    sys.exit("Need to specify a model to load!!")
  
  
  output_treebank = treebank_pb2.Treebank()
  output_treebank.language = languages[kwargs["language"]]
  for sentence in sentences:
    predicted_labels = labeler.PredictLabels(sentence)
    labeled_sentence = labeler.InsertLabels(sentence, predicted_labels)
    output_treebank.sentence.extend([labeled_sentence])
  
  assert len(sentences) == len(output_treebank.sentence), "Mismatch in the number of input and output sentences!!"
  try:
    output_dir = os.path.join(_TREEBANK_DIR, kwargs["language"], "labeled")
  except:
    logging.warning("Couldn't find the output file")
    output_dir = os.path.join(_TREEBANK_DIR, "Turkish", "labeled")
  
  output_path = os.path.join(output_dir, "labeled_{}_{}.pbtxt".format(kwargs["model"], path))
  write_proto_as_text(output_treebank, output_path)
  logging.info("{} sentences written to {}".format(len(sentences), output_path))
  
  return output_treebank
    
  
  