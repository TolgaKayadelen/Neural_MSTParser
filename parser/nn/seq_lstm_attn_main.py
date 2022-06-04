import os
import logging
import tensorflow as tf
import datetime

import argparse
from parser.nn.seq_lstm_attn import SeqLSTMAttnLabeler
from util import converter, writer
from util.nn import nn_utils
from input import embeddor, preprocessor

def main(args):
  current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
  log_dir = "debug/seq_lstm_attn/" + current_time
  tf.debugging.experimental.enable_dump_debug_info(
    log_dir,
    tensor_debug_mode="FULL_HEALTH",
    circular_buffer_size=-1)
  embeddings = nn_utils.load_embeddings()
  word_embeddings = embeddor.Embeddings(name="word2vec", matrix=embeddings)
  prep = preprocessor.Preprocessor(
    word_embeddings=word_embeddings,
    features=["words", "pos", "morph", "heads", "dep_labels"],
    labels=["heads"]
  )
  label_feature = next(
    (f for f in prep.sequence_features_dict.values() if f.name == "dep_labels"), None)

  parser = SeqLSTMAttnLabeler(word_embeddings=prep.word_embeddings,
                              n_output_classes=label_feature.n_values,
                              predict=["labels"],
                              features=["words", "pos", "morph"],
                              model_name="seq_lstm_attn_labeler",
                              log_dir=log_dir,
                              test_every=args.test_every)

  _DATA_DIR="data/UDv29/train/tr/"
  _TEST_DATA_DIR="data/UDv29/test/tr/"
  logging.info(f"Reading training data from {args.train_data}")
  dataset = prep.read_dataset_from_tfrecords(
    records= _DATA_DIR + args.train_data,
    batch_size=args.train_batch_size)
  test_treebank = args.test_data # None

  """
  train_treebank = "tr_boun-ud-train-random500.pbtxt"
  test_treebank = None # "tr_boun-ud-test-random50.pbtxt"
  train_sentences = prep.prepare_sentence_protos(
    path=os.path.join(_DATA_DIR, train_treebank))
  dataset = prep.make_dataset_from_generator(
    sentences=train_sentences,
    batch_size=50
  )
  """
  if test_treebank is not None:
    # test_sentences = prep.prepare_sentence_protos(
    #  path=os.path.join(_TEST_DATA_DIR, test_treebank))
    test_dataset = prep.read_dataset_from_tfrecords(
      records= _TEST_DATA_DIR + test_treebank,
      batch_size=args.test_batch_size)
  else:
    test_dataset=None

  metrics = parser.train(dataset=dataset, epochs=args.epochs,
                         test_data=test_dataset)
  print(metrics)
  # writer.write_proto_as_text(metrics,
  #                           f"./model/nn/plot/{parser.model_name}_metrics.pbtxt")
  # nn_utils.plot_metrics(name=parser.model_name, metrics=metrics)
  # parser.save_weights()
  # logging.info(f"{parser.model_name} results written")

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--train_data",
                      type=str,
                      default="tr_boun-ud-train-random10.tfrecords",
                      help="Train dataset name")
  parser.add_argument("--train_batch_size",
                      type=int,
                      default=100,
                      help="Size of training data batches")
  parser.add_argument("--test_data",
                      type=str,
                      default="tr_boun-ud-test-random10.tfrecords",
                      help="Test/dev dataset name.")
  parser.add_argument("--test_batch_size",
                      type=int,
                      default=10,
                      help="Size of test data batches")
  parser.add_argument("--epochs",
                      type=int,
                      default=300,
                      help="Trains a new model.")
  parser.add_argument("--test_every",
                      type=int,
                      default=10,
                      help="Decides after how many iterations to test.")

  args = parser.parse_args()
  main(args)