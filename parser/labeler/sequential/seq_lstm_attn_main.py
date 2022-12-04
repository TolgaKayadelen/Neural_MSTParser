import os
import logging
import tensorflow as tf
import datetime

import argparse
from parser.nn.seq_lstm_attn import SeqLSTMAttnLabeler
from parser.nn import load_models
from util import converter, writer
from util.nn import nn_utils
from input import embeddor, preprocessor

def main(args):
  current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
  log_dir = "debug/seq_lstm_attn/" + current_time
  # tf.debugging.experimental.enable_dump_debug_info(
  #  log_dir,
  #  tensor_debug_mode="FULL_HEALTH",
  #  circular_buffer_size=-1)
  word_embeddings = load_models.load_word_embeddings()
  prep = load_models.load_preprocessor(word_embeddings=word_embeddings)
  label_feature = next(
    (f for f in prep.sequence_features_dict.values() if f.name == "dep_labels"), None)

  parser = SeqLSTMAttnLabeler(word_embeddings=prep.word_embeddings,
                              n_output_classes=label_feature.n_values,
                              predict=["labels"],
                              features=["words", "pos", "morph"],
                              model_name="seq_lstm_attn_labeler_test",
                              log_dir=log_dir,
                              test_every=args.test_every)

  train_treebank= "tr_boun-ud-train-random500.pbtxt"
  test_treebank = "tr_boun-ud-test-random50.pbtxt"
  train_dataset, dev_dataset, test_dataset = load_models.load_data(preprocessor=prep,
                                                                 train_treebank=train_treebank,
                                                                 batch_size=100,
                                                                 test_treebank=test_treebank,
                                                                 test_batch_size=1,
                                                                 type="pbtxt")
  metrics = parser.train(dataset=train_dataset, epochs=args.epochs,
                         test_data=test_dataset)
  print(metrics)
  writer.write_proto_as_text(metrics,
                             f"./model/nn/plot/final/{parser.model_name}_metrics.pbtxt")
  # nn_utils.plot_metrics(name=parser.model_name, metrics=metrics)
  parser.save_weights()
  logging.info(f"{parser.model_name} results written")

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--train_data",
                      type=str,
                      default="tr_boun-ud-train-random10.tfrecords",
                      help="Train dataset name")
  parser.add_argument("--train_batch_size",
                      type=int,
                      default=2,
                      help="Size of training data batches")
  parser.add_argument("--test_data",
                      type=str,
                      default="tr_boun-ud-test-random10.tfrecords",
                      help="Test/dev dataset name.")
  parser.add_argument("--test_batch_size",
                      type=int,
                      default=2,
                      help="Size of test data batches")
  parser.add_argument("--epochs",
                      type=int,
                      default=100,
                      help="Trains a new model.")
  parser.add_argument("--test_every",
                      type=int,
                      default=3,
                      help="Decides after how many iterations to test.")

  args = parser.parse_args()
  main(args)