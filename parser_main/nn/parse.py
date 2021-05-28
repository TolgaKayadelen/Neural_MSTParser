"""Module to parse sentences with dependency parser."""

import os
import itertools
import tensorflow as tf

from input import preprocessor
from input import embeddor

from data.treebank import sentence_pb2
from data.treebank import treebank_pb2
from tagset.reader import LabelReader

from util import common
from util import reader
from util.writer import write_proto_as_text

from eval import evaluate

from google.protobuf import text_format

import logging
logging.basicConfig(format='%(levelname)s : %(message)s', level=logging.INFO)

_DATA_DIR = "data/UDv23/Turkish/test"

def parse(*, prep, treebank, parser) -> treebank_pb2.Treebank:
    """Parses a set of sentences with the a parsing model.

    Args:
        prep: preprocessor used to make a dataset from the treebank.
        treebank: path to a treebank_pb2.Treebank
        parser: the parser object used to parse the sentences.
    Returns:
        dict: dict of sent_id:sentence_pb2.Sentence objects.
    """
    output_dict = {}
    dep_labels = LabelReader.get_labels("dep_labels", reverse=True).labels
    logging.info(f"Generating dataset from {treebank}")
    trb = reader.ReadTreebankTextProto(os.path.join(_DATA_DIR, treebank))
   
    dataset = prep.make_dataset_from_generator(
      path=os.path.join(_DATA_DIR, treebank),
      batch_size=1)
    
    # TODO: start debugging from here for the case where --parser=biaffine
    for example in dataset:
      tokens = example["tokens"].numpy().tolist()[0]
      # print([token.decode("utf-8") for token in tokens])
      edge_scores, label_scores = parser.parse(example)
      edge_preds = tf.argmax(edge_scores, axis=2).numpy().tolist()[0]
      label_preds = tf.argmax(label_scores, axis=2).numpy().tolist()[0]
      sent_id = example["sent_id"][0][0]
      
      # Fill in the sent_id and the length of the sentence.
      sentence = sentence_pb2.Sentence(
        sent_id=sent_id.numpy().decode("utf-8"),
        length=(len(tokens))
      )
      
      # Fill in the tokens.
      for i, values in enumerate(itertools.zip_longest(
                                          tokens, edge_preds, label_preds,
                                          fillvalue=None)):
        if None in values:
          raise ValueError("Length of feature sequences don't match!!")
     
        token = sentence.token.add(
          word=values[0].decode("utf-8"),
          selected_head=sentence_pb2.Head(address=values[1]),
          label=dep_labels[values[2]],
          index=i
        )
    
      output_dict[sent_id.numpy().decode("utf-8")] = sentence
    
    # TODO: but this is creating a problem, normally you don't have access 
    # to a gold treebank when you want to parse data.
    evaluator = evaluate.Evaluator(gold=trb, test=output_dict)
    evaluator.evaluate("all")
    
    # print(output_dict)                                  
    return output_dict
    
    '''
    # save the parsed sentences to an output
    if save_treebank:
      _OUTPUT_DIR = os.path.join(_TREEBANK_DIR, args.language, "parsed")
      output_path = os.path.join(_OUTPUT_DIR, "parsed_{}.pbtxt".format(args.test_data))
      write_proto_as_text(output_treebank, output_path)
      logging.info("{} sentences written to {}".format(len(output_treebank.sentence), output_path))

    return output_treebank
    '''