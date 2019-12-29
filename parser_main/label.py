# -*- coding: utf-8 -*-

"""Module to label sentences with dependency labeler."""


import argparse
import os
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

def label(treebank):
  """Label dependency arcs in a treebank.
  Args:
    treebank: treebank_pb2.Treebank(), where dependency arcs are not labeled.
  Returns:
    treebank where dependency arcs are labeled. 
  """
  sentences = list(treebank.sentence)
  
  