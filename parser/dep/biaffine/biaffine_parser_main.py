
import sys
import argparse
import logging

import datetime

from util.nn import nn_utils
from input import embeddor, preprocessor
from parser.dep.biaffine.biaffine_parser import BiaffineParser
from parser.utils import load_models, pos_tag_to_id
from util import writer, reader

def main(args):
  current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
  parser_model_name = f"{args.language}_biaffine_{current_time}"
  logging.info(f"Parser model name is {parser_model_name}")
  #model_name_check = input("Are you happy with the model name: y/n?")
  #if model_name_check != "y":
  #  raise ValueError("Model name is not set properly!")
  log_dir = f"debug/label_first_parser/{args.language}/{parser_model_name}"
  logging.info(f"Logging to {log_dir}")

  if "pos" in args.features:
    if args.language != "tr":
      logging.info(f"Generating pos label to id dictionary for {args.language}..")
      pos_to_id = pos_tag_to_id.postags[args.language]
      pos_embedding_vocab_size=len(pos_to_id.keys())
      print(f"{args.language} pos to id {pos_to_id}")
      print(f"pos vocab size ", pos_embedding_vocab_size)
    else:
      pos_embedding_vocab_size = 37 # for turkish.
      pos_to_id=None
  else:
    pos_to_id=None
    pos_embedding_vocab_size=None

  word_embeddings = load_models.load_i18n_embeddings(language=args.language)
  logging.info(f"Loading word embeddings for {args.language}")

  prep = load_models.load_preprocessor(word_embedding_indexes=word_embeddings.token_to_index,
                                       pos_indexes=pos_to_id,
                                       language=args.language,
                                       features=args.features,
                                       embedding_type=args.embeddings)

  logging.info(f"features used to train are {args.features}")
  logging.info(f"parser will output predictions for {args.predict}")
  print(prep.word_embedding_indexes[b"test"])

  label_feature = next(
    (f for f in prep.sequence_features_dict.values() if f.name == "dep_labels"), None)

  parser = BiaffineParser(word_embeddings=word_embeddings,
                          language=args.language,
                          n_output_classes=label_feature.n_values,
                          predict=args.predict,
                          features=args.features,
                          log_dir=log_dir,
                          model_name=parser_model_name,
                          pos_embedding_vocab_size=pos_embedding_vocab_size,
                          one_hot_labels=False,
                          test_every=5)

  train_data_dir = f"./data/UDv29/train/{args.language}"
  test_data_dir = f"./data/UDv29/test/{args.language}"
  train_treebank_name = args.train_treebank
  test_treebank_name = args.test_treebank
  train_dataset, dev_dataset, test_dataset = load_models.load_data(preprocessor=prep,
                                                                   train_treebank=train_treebank_name,
                                                                   batch_size=250,
                                                                   test_treebank=test_treebank_name,
                                                                   type="pbtxt",
                                                                   language=args.language)
  #for batch in train_dataset:
  #  print(batch)
  #  input()
  metrics = parser.train(dataset=train_dataset, epochs=75,
                         test_data=test_dataset)
  print(metrics)
  writer.write_proto_as_text(metrics,
                             f"./model/nn/plot/final/{parser.model_name}_metrics.pbtxt")
  # nn_utils.plot_metrics(name=parser.model_name, metrics=metrics)
  parser.save_weights()
  logging.info(f"{parser.model_name} results written")

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--language",
                      type=str,
                      choices=["tr", "en", "de", "fi", "zh"],
                      required=True)
  parser.add_argument("--features",
                      nargs="+",
                      default=["words", "pos"])
  parser.add_argument("--predict",
                      nargs="+",
                      default=["heads"])
  parser.add_argument("--train_treebank",
                      type=str,
                      required=True)
  parser.add_argument("--test_treebank",
                      type=str,
                      required=True)
  parser.add_argument("--train_batch_size",
                      type=int,
                      default=2)
  parser.add_argument("--embeddings",
                      type=str,
                      default="conll")
  args = parser.parse_args()
  main(args)
