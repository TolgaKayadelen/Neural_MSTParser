import argparse
import logging
import os
import sys

from input import preprocessor, embeddor
import tensorflow as tf
import numpy as np

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
  if args.train:
    embeddings = nn_utils.load_embeddings()
    word_embeddings = embeddor.Embeddings(name= "word2vec", matrix=embeddings)
    prep = preprocessor.Preprocessor(
      word_embeddings=word_embeddings, features=args.features,
      labels=args.labels)
    label_feature = next((f for f in prep.sequence_features if f.name == "dep_labels"),
                          None)
    # print("label feature ", label_feature)
    # print("n output classes", label_feature.n_values)
    # input("press to cont.")
    
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
  parser = jbp.NeuralMSTParser(word_embeddings=prep.word_embeddings,
                               n_output_classes=label_feature.n_values,
                               predict=args.predict)
  print(parser)
  parser.plot()
  metrics = parser.train(dataset, args.epochs, test_data=test_dataset)
  writer.write_proto_as_text(metrics,
                             f"./model/nn/{args.model_name}_metrics.pbtxt")
  nn_utils.plot_metrics(name=args.model_name, metrics=metrics)
  logging.info(f"{args.model_name} results written to ./model/nn directory")
  
    
if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--train",
                      type=bool,
                      default=True,
                      help="Trains a new model.")
  parser.add_argument("--test",
                      type=bool,
                      default=True,
                      help="Whether to test the trained model on test data.")
  parser.add_argument("--epochs",
                      type=int,
                      default=10,
                      help="Trains a new model.")
  parser.add_argument("--treebank",
                      type=str,
                      default="treebank_train_0_10.pbtxt")
  parser.add_argument("--test_treebank",
                      type=str,
                      default="treebank_0_3_gold.pbtxt")
  parser.add_argument("--dataset",
                      help="path to a prepared tf.data.Dataset")
  parser.add_argument("--features",
                      nargs="+",
                      type=str,
                      default=["words", "pos", "morph", "dep_labels", "heads"],
                      help="features to use to train the model.")
  # TODO: make sure --labels and --predict follow consistent naming.
  parser.add_argument("--labels",
                      nargs="+",
                      type=str,
                      default=["heads", "dep_labels"],
                      help="labels to predict.")
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
                      default="edges_biaffine",
                      help="Name of the model to save.")

  args = parser.parse_args()
  main(args)