# -*- coding: utf-8 -*-

"""Main module to train and parse sentences with dependency parser."""

import argparse
import sys
from parser_main.parse import parse
from parser_main.label import label
from parser_main.parse_and_label import parse_and_label
from parser_main.train import train_parser, train_labeler
from parser_main.evaluate import evaluate_parser

import logging
logging.basicConfig(format='%(levelname)s : %(message)s', level=logging.INFO)

def validate_args(args):
  if not args.mode:
    sys.exit("Need to define a mode!!")
  if args.mode == "train" and not args.language:
    sys.exit("Need to specify a language!!")
  if args.mode == "train" and not args.train_data:
    sys.exit("Need to speficy training data!!")
  if args.mode == "train" and not (args.parser or args.labeler):
    sys.exit("Need to train either a parser or a labeler!!")
  if (args.mode == "train" and args.parser and
   not args.load and not args.arcfeatures.startswith("arcfeatures")):
    sys.exit("Need to train a parser with arc features!!")
  if (args.mode == "train" and args.labeler and
   not args.load and not args.labelfeatures.startswith("labelfeatures")):
    sys.exit("Need to train a labeler with label features!!")
  if args.mode == "evaluate" and not args.gold_data:
    sys.exit("Need to specify gold data to evaluate a model!!")
  if args.mode == "evaluate" and not args.test_data:
    sys.exit("Need to specify test data to evaluate a model!!")
  if args.mode == "parse_and_label" and (not args.parser_model or not args.labeler_model):
    sys.exit("Need to have both parser and labeler models to parse and label!!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["train", "evaluate", "plot", "parse", "label", "parse_and_label"], 
                        help="Choose action for the parser.")
    parser.add_argument("--parser", type=bool, help="Whether to train a dependency parser.")
    parser.add_argument("--labeler", type=bool, help="Whether to train a dependency labeler.")
    
    # Data args.
    parser.add_argument("--language", type=str, choices=["English", "Turkish"],
                        help="language")
    parser.add_argument("--train_data", type=str,
    					help="The train data to read(.protobuf)")
    parser.add_argument("--test_data", type=str,
    					help="The test data to read (.protobuf)")
    parser.add_argument("--gold_data", type=str,
    					help="Gold data to use for evaluating a model.")
    parser.add_argument("--split", '--list', action="append",
    					help="Split training and test")
    # Model args.
    parser.add_argument("--decoder", type=str, choices=["mst", "eisner"],
                        default="mst",
                        help="decoder to extract tree from scores.")
    parser.add_argument("--parser_model", type=str, default="model.json",
                        help="path to save the parser model to, or load it from.")
    parser.add_argument("--labeler_model", type=str, default="model.json",
                        help="path to save the labeler model to, or to load it from.")
    parser.add_argument("--load", type=bool,
                        help="Load pretrained model, speficy which w/ --model.",
                        default=False)
    parser.add_argument('--labelfeatures', type=str,
                        help="name of the file that contain the label features to train with.",
                        default="labelfeatures_base")
    parser.add_argument("--arcfeatures", type=str, 
                        help="name of the file that contain the arc features to train with",
                        default="arcfeatures_exp2")
            
    
    # Training args
    parser.add_argument("--epochs", type=int, default=10,
                        help="Number of epochs to train on.")
    parser.add_argument("--out", type=str,
    					help="dir to put the parsed output files.")
    parser.add_argument("--learning_rate", type=float, default=1.0,
                        help="The learning rate for the parser/labeler training.")
    
    # Evaluate args
    parser.add_argument("--metrics", nargs='*',
    					default=["uas_total", "las_total"],
    					help="Space separated list of metrics to evaluate.",
    					choices=["uas_total", "las_total", "typed_uas",
    					"typed_las_prec", "typed_las_recall", "typed_las_f1",
    					"all"])
    parser.add_argument("--print_eval_results", type=bool, default=False,
                        help="whether to print eval results on screen.")
    
    args = parser.parse_args()
    
    validate_args(args)
    
    if args.mode == "train" and args.parser and not args.labeler:
      train_parser(args)
    elif args.mode == "train" and args.labeler and not args.parser:
      train_labeler(args)
    elif args.mode == "train" and args.parser and args.labeler:
      train_parser(args)
      train_labeler(args)
    elif args.mode == "parse":
      parse(args)
    elif args.mode == "label":
      label(path=args.test_data, model=args.labeler_model,
            language=args.language, treebank_name=args.test_data
            )
    elif args.mode == "evaluate":
      evaluate_parser(args, print_results=args.print_eval_results)
    elif args.mode == "parse_and_label": 
      parse_and_label(args)
    
