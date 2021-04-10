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
    # TODO: also add sentence ids to the dataset generator.
    dataset = prep.make_dataset_from_generator(
      path=os.path.join(_DATA_DIR, treebank),
      batch_size=1)
    #  print([word.decode("utf-8") for word in a.numpy().tolist()])
    for example in dataset:
      tokens = example["tokens"].numpy().tolist()
      print([token.decode("utf-8") for token in tokens[0]])
      edge_scores, label_scores = parser.parse(example)
      print(f"""edge scores: {edge_scores}, {edge_scores.shape}, 
             label_scores: {label_scores}, {label_scores.shape})""")
      print(f"edge_preds: {tf.argmax(edge_scores, axis=2)}")
      print(f"label preds {tf.argmax(label_scores, axis=2)}")
      input("press to cont..")
    
    '''
    # save the parsed sentences to an output
    if save_treebank:
      _OUTPUT_DIR = os.path.join(_TREEBANK_DIR, args.language, "parsed")
      output_path = os.path.join(_OUTPUT_DIR, "parsed_{}.pbtxt".format(args.test_data))
      write_proto_as_text(output_treebank, output_path)
      logging.info("{} sentences written to {}".format(len(output_treebank.sentence), output_path))

    return output_treebank
    '''