"""Module to parse sentences with dependency parser."""

import os
import tensorflow as tf
from input import preprocessor
from input import embeddor
from data.treebank import sentence_pb2
from data.treebank import treebank_pb2
from util import common
from util import reader
from util.writer import write_proto_as_text

from google.protobuf import text_format

import logging
logging.basicConfig(format='%(levelname)s : %(message)s', level=logging.INFO)

_DATA_DIR = "data/UDv23/Turkish/test"

def parse(*, prep, treebank, parser) -> treebank_pb2.Treebank:
    """Parses a set of sentences with the a parsing model.

    Args:
        args: commandline arguments from argparse
    Returns:
        treebank of parsed sentences.
    """
    output_treebank = treebank_pb2.Treebank()
    
    # generate tf.examples from the sentences
    logging.info(f"Generating dataset from {treebank}")
    trb = reader.ReadTreebankTextProto(os.path.join(_DATA_DIR, treebank))
    tf_examples = prep.make_dataset(from_sentences=trb.sentence)
    
    # parse sentences with the loaded model.
    # for example in tf_examples:
      # edges, labels = parser.parse(example)
      # print("edges ", edges)
      # print("labels ", labels)
      # input("press to cont...")
    
    '''
    # save the parsed sentences to an output
    if save_treebank:
      _OUTPUT_DIR = os.path.join(_TREEBANK_DIR, args.language, "parsed")
      output_path = os.path.join(_OUTPUT_DIR, "parsed_{}.pbtxt".format(args.test_data))
      write_proto_as_text(output_treebank, output_path)
      logging.info("{} sentences written to {}".format(len(output_treebank.sentence), output_path))

    return output_treebank
    '''