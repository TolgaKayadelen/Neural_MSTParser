import argparse
import logging
import os
import sys

from input import preprocessor
from input import embeddor
from parser_main.nn import parse
import numpy as np
import tensorflow as tf


from parser.nn import joint_biaffine_parser_copy as jbp
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
  prep = preprocessor.Preprocessor(
    word_embeddings=word_embeddings, features=args.features,
    labels=args.labels)
  label_feature = next((f for f in prep.sequence_features if f.name == "dep_labels"),
                        None)
  
  
  if args.load:
    parser = jbp.NeuralMSTParser(word_embeddings=prep.word_embeddings,
                                 n_output_classes=label_feature.n_values,
                                 predict=args.predict, model_name=args.model_name)
    parser.load(name=args.model_name)
    print(parser)
    parser.plot()
  
  if args.parse:
    parse.parse(prep=prep, treebank=args.test_treebank, parser=parser)

  if args.train:
    parser = jbp.NeuralMSTParser(word_embeddings=prep.word_embeddings,
                                 n_output_classes=label_feature.n_values,
                                 predict=args.predict, model_name=args.model_name)
    print(parser)
    parser.plot()
    
    if args.dataset:
      logging.info(f"Reading from tfrecords {args.dataset}")
      dataset = prep.read_dataset_from_tfrecords(
                                 batch_size=args.batchsize,
                                 records="./input/treebank_train_0_50.tfrecords")
    else:
      logging.info(f"Generating dataset from {args.treebank}")
      dataset = prep.make_dataset_from_generator(
        path=os.path.join(_DATA_DIR, args.treebank),
        batch_size=args.batchsize)
    if args.test:
      if not args.train:
        sys.exit("Testing with a pretrained model is not supported yet.")
      test_dataset = prep.make_dataset_from_generator(
        path=os.path.join(_TEST_DATA_DIR, args.test_treebank),
        batch_size=1)
    
    metrics = parser.train(dataset, args.epochs, test_data=test_dataset)
    writer.write_proto_as_text(metrics,
                               f"./model/nn/plot/{args.model_name}_metrics.pbtxt")
    nn_utils.plot_metrics(name=args.model_name, metrics=metrics)
    logging.info(f"{args.model_name} results written to ./model/nn/plot directory")
    parser.save()
    
if __name__ == "__main__":
  parser = argparse.ArgumentParser()
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
  parser.add_argument("--epochs",
                      type=int,
                      default=70,
                      help="Trains a new model.")
  parser.add_argument("--treebank",
                      type=str,
                      default="treebank_tr_imst_ud_train_dev.pbtxt")
  parser.add_argument("--test_treebank",
                      type=str,
                      default="treebank_tr_imst_ud_test_fixed.pbtxt")
  parser.add_argument("--dataset",
                      help="path to a prepared tf.data.Dataset")
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
  # you need to set this up
  parser.add_argument("--predict",
                      nargs="+",
                      type=str,
                      default=["edges", "labels"],
            					choices=["edges", "labels"],
                      help="which features to predict")
  parser.add_argument("--batchsize",
                      type=int, 
                      default=250,
                      help="Size of training and test data batches")
  parser.add_argument("--model_name",
                      type=str,
                      required=True,
                      help="Name of the model to save or load.")
  parser.add_argument("--load",
                      action='store_true',
                      help="Whether to load a pretrained model.")

  args = parser.parse_args()
  main(args)