"""Usage example
bazel-bin/parser_main/nn/main --train --parser_type=label_first \
--train_treebank=treebank_train_0_50.pbtxt \
--test_treebank=treebank_0_3_gold.pbtxt \
--epochs=5 --model_name=stupid --predict edges labels
"""

import argparse
import logging
import os
import sys

from input import preprocessor
from input import embeddor
from parser_main.nn import parse
import numpy as np
import tensorflow as tf


from parser.nn import label_first_parser_deprecated as lfp
from parser.nn import label_first_parser2 as lfp2
from parser.nn import label_first_parser_joint_loss as lfp_joint_loss
from parser.nn import biaffine_parser as bfp
from parser.nn import seq2seq_labeler as seq
from parser.nn import seq2lstm_labeler as seqlstm
from parser.nn import bilstm_labeler
from util.nn import nn_utils
from util import writer

# Set up basic configurations
logging.basicConfig(format='%(levelname)s : %(message)s', level=logging.INFO)
np.set_printoptions(threshold=np.inf)

# Set up type aliases
Dataset = tf.data.Dataset
Embeddings = embeddor.Embeddings

# Set up global constants
_DATA_DIR = "data/UDv23/Turkish/training"
_TEST_DATA_DIR = "data/UDv23/Turkish/test"


def main(args):
  embeddings = nn_utils.load_embeddings()
  word_embeddings = embeddor.Embeddings(name= "word2vec", matrix=embeddings)
  logging.info(f"labels {args.labels}")
  logging.info(f"features {args.features}")
  prep = preprocessor.Preprocessor(
      word_embeddings=word_embeddings,
      features=args.features,
      labels=args.labels
  )
  label_feature = next((f for f in prep.sequence_features_dict.values() if f.name == "dep_labels"),
                        None)

  if args.load:
    if args.parser_type == "label_first":
      parser = lfp.LabelFirstMSTParser(word_embeddings=prep.word_embeddings,
                                       n_output_classes=label_feature.n_values,
                                       predict=args.predict,
                                       model_name=args.model_name)
      parser.load(name=args.model_name)
      print(parser)
      parser.plot()
      
    elif args.parser_type == "biaffine":
      parser = bfp.BiaffineMSTParser(word_embeddings=prep.word_embeddings,
                                     n_output_classes=label_feature.n_values,
                                     predict=args.predict,
                                     model_name=args.model_name)
      parser.load(name=args.model_name)
      print(parser)
      parser.plot()
  
    if args.parse:
      parse.parse(prep=prep, treebank=args.test_treebank, parser=parser,
                  parser_type=args.parser_type)

  if args.train:
    if args.parser_type == "biaffine":
      parser = bfp.BiaffineMSTParser(word_embeddings=prep.word_embeddings,
                                     n_output_classes=label_feature.n_values,
                                     predict=args.predict,
                                     model_name=args.model_name)
      parser.plot()
      print(parser)
    elif args.parser_type == "label_first":
      parser = lfp.LabelFirstMSTParser(word_embeddings=prep.word_embeddings,
                                       n_output_classes=label_feature.n_values,
                                       predict=args.predict,
                                       model_name=args.model_name)
      parser.plot()
      print(parser)
    
    elif args.parser_type == "label_first2":
      parser = lfp2.LabelFirstMSTParser2(word_embeddings=prep.word_embeddings,
                                         n_output_classes=label_feature.n_values,
                                         predict=args.predict,
                                         model_name=args.model_name)
      parser.plot()
      print(parser)
      input("press to cont.")
    elif args.parser_type == "label_first_joint_loss":
      parser = lfp_joint_loss.LabelFirstMSTParser(
                                       word_embeddings=prep.word_embeddings,
                                       n_output_classes=label_feature.n_values,
                                       predict=args.predict,
                                       model_name=args.model_name)
      parser.plot()
      print(parser)
    elif args.parser_type == "seq2lstm_labeler":
      parser = seqlstm.Seq2LSTMLabeler(word_embeddings=prep.word_embeddings,
                                       n_output_classes=label_feature.n_values,
                                       encoder_dim=512,
                                       decoder_dim=512,
                                       batch_size=args.batchsize,
                                       model_name=args.model_name)
      print(parser)
    
    elif args.parser_type == "bilstm_labeler":
      parser = bilstm_labeler.BiLSTMLabeler(
                                        word_embeddings=prep.word_embeddings,
                                        n_output_classes=label_feature.n_values,
                                        n_units=256,
                                        model_name=args.model_name)
      print(parser)
    elif args.parser_type == "seq2seq_labeler":
      parser = seq.Seq2SeqLabeler(word_embeddings=prep.word_embeddings,
                                  n_output_classes=label_feature.n_values,
                                  encoder_dim=512,
                                  decoder_dim=512,
                                  batch_size=args.batchsize
                                  )
    else:
      raise ValueError("Unsupported value for the parser argument.")
    
    
    
    if args.dataset:
      logging.info(f"Reading from tfrecords {args.dataset}")
      dataset = prep.read_dataset_from_tfrecords(
                                 batch_size=args.batchsize,
                                 records="./input/treebank_train_0_50.tfrecords")
    else:
      logging.info(f"Generating dataset from {args.train_treebank}")
      dataset = prep.make_dataset_from_generator(
        path=os.path.join(_DATA_DIR, args.train_treebank),
        batch_size=args.batchsize)
    if args.test:
      if not args.train:
        sys.exit("Testing with a pretrained model is not supported yet.")
      test_dataset = prep.make_dataset_from_generator(
        path=os.path.join(_TEST_DATA_DIR, args.test_treebank),
        batch_size=1)
    
    # Start training
    metrics = parser.train(dataset, args.epochs, test_data=test_dataset)

    writer.write_proto_as_text(metrics,
                               f"./model/nn/plot/{args.model_name}_metrics.pbtxt")
    nn_utils.plot_metrics(name=args.model_name, metrics=metrics)
    logging.info(f"{args.model_name} results written to ./model/nn/plot directory")
    parser.save()
    
if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  # Determine which parser to use.
  parser.add_argument("--parser_type", 
                      type=str,
                      choices=["label_first",
                              "label_first2",
                              "biaffine",
                              "label_first_joint_loss",
                              "seq2seq_labeler",
                              "seq2lstm_labeler",
                              "bilstm_labeler"],
                      default="label_first",
                      help="Which parser to use.")

  # Determine if you want to load a pretrained parser.
  parser.add_argument("--load",
                      action='store_true',
                      help="Whether to load a pretrained model.")
  
  # Choose a model name to load from or save to. 
  parser.add_argument("--model_name",
                      type=str,
                      required=True,
                      help="Name of the model to save or load.")

  # Choose parser modes.
  parser.add_argument("--train",
                      action='store_true',
                      help="Trains a new model.")
  parser.add_argument("--test",
                      type=bool,
                      default=True,
                      help="Whether to test the trained model on test data.")
  parser.add_argument("--parse",
                      action="store_true",
                      help="Parses --test_treebank with the --model_name")

  # Set up parser config.
  parser.add_argument("--epochs",
                      type=int,
                      default=70,
                      help="Trains a new model.")
  parser.add_argument("--batchsize",
                      type=int, 
                      default=250,
                      help="Size of training and test data batches")
  parser.add_argument("--features",
                      nargs="+",
                      type=str,
                      default=["words", "pos", "morph", "dep_labels", "heads"],
                      help="features to use to train the model.")
  # TODO: make sure --labels and --predict follow consistent naming.
  # --labels is used to understand what are the label features.
  # you might leave it as is.
  parser.add_argument("--labels",
                      nargs="+",
                      type=str,
                      default=["heads", "dep_labels"],
                      help="labels to predict.")
  # predict determines which features the parser will train to predict.
  # you need to set this up. Pass this as --predict edges labels
  parser.add_argument("--predict",
                      nargs="+",
                      type=str,
                      # default=["edges", "labels"],
            					choices=["edges", "labels"],
                      help="which features to predict")
  
  # Determine which datasets to use for train, test or evaluate.
  parser.add_argument("--train_treebank",
                      type=str,
                      default="treebank_tr_imst_ud_train_dev.pbtxt")
  parser.add_argument("--test_treebank",
                      type=str,
                      default="treebank_tr_imst_ud_test_fixed.pbtxt")
  parser.add_argument("--gold_treebank",
                      type=str,
                      help="treebank to compare model parses against")
  parser.add_argument("--dataset",
                      help="path to a prepared tf.data.Dataset")

  args = parser.parse_args()
  main(args)