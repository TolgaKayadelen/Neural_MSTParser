# -*- coding: utf-8 -*-

"""Module to parse and label sentences with dependency parser and labeler."""
import os

from parser_main.parse import parse
from parser_main.label import label
from util import writer

from google.protobuf import text_format

import logging
logging.basicConfig(format='%(levelname)s : %(message)s', level=logging.DEBUG)

def parse_and_label(args):
  
  parsed_treebank = parse(args, save_treebank=False)
  parsed_and_labeled_treebank = label(save_treebank=False,
    model=args.labeler_model, treebank=parsed_treebank,
    treebank_name=args.test_data, language=args.language
  )
  parsed_and_labeled_treebank.is_gold = False
  parsed_and_labeled_treebank.parser_model = args.parser_model
  parsed_and_labeled_treebank.labeler_model = args.labeler_model
  
  text_format.MessageToString(parsed_and_labeled_treebank, as_utf8=True)
  
  output_dir = os.path.join("data/UDv23", args.language, "parsed_and_labeled")
  output_path = os.path.join(output_dir, args.test_data+".pbtxt")
  writer.write_proto_as_text(parsed_and_labeled_treebank, output_path)
  logging.info("{} sentences written to: {}".format(len(parsed_and_labeled_treebank.sentence), output_path))