# -*- coding: utf-8 -*-

"""Main module to train and parse sentences with dependency parser."""

import argparse
from parser_main.parse import parse
from parser_main.train import train_parser, train_labeler
from parser_main.evaluate import evaluate_parser


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["train", "evaluate", "plot", "parse"], 
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
    parser.add_argument("--model", type=str, default="model.json",
                        help="path to save the model to, or load it from.")
    parser.add_argument("--load", type=bool,
                        help="Load pretrained model, speficy which w/ --model.",
                        default=False)
    parser.add_argument('--features', nargs='+', default=[],
                        help='Space separated list of additional features',
                        choices=['dist', 'surround', 'between'])
    
    # Training args
    parser.add_argument("--epochs", type=int, default=10,
                        help="Number of epochs to train on.")
    parser.add_argument("--out", type=str,
    					help="dir to put the parsed output files.")
    
    # Evaluate args
    parser.add_argument("--metrics", nargs='*',
    					default=["uas_total", "las_total"],
    					help="Space separated list of metrics to evaluate.",
    					choices=["uas_total", "las_total", "typed_uas",
    					"typed_las_prec", "typed_las_recall", "typed_las_f1",
    					"all"])
    
    args = parser.parse_args()
    
    if args.mode == "train" and args.parser and not args.labeler:
      train_parser(args)
    elif args.mode == "train" and args.labeler and not args.parser:
      train_labeler(args)
    elif args.mode == "train" and args.parser and args.labeler:
      train_parser(args)
      train_labeler(args)
    elif args.mode == "parse":
      parse(args)
    elif args.mode == "evaluate":
      evaluate_parser(args)
    
