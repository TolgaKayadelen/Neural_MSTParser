# -*- coding: utf-8 -*-

"""Module to parse and label sentences with dependency parser and labeler."""

from parser_main.parse import parse
from parser_main.label import label

from google.protobuf import text_format

def parse_and_label(args):
  
  parsed_treebank = parse(args, save_treebank=False)
  parsed_and_labeled_treebank = label(
    model=args.labeler_model, treebank=parsed_treebank,
    treebank_name=args.test_data, language=args.language
  )
  
  text_format.MessageToString(parsed_and_labeled_treebank, as_utf8=True)
  
  
  
  