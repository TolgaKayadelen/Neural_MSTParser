# Use like:
# bazel-bin/parser/dep/lfp/label_first_parser_main \
# --language=tr --predict heads labels --train_treebank=train_dev_merged.pbtxt \
# --test_treebank=test_merged.pbtxt --features words pos morph srl


import tensorflow as tf
import argparse
import logging

import datetime
import os

from data.treebank import sentence_pb2
from data.treebank import treebank_pb2
from parser.utils import load_models, pos_tag_to_id, srl_to_id
from parser.dep.lfp.label_first_parser_srl import LabelFirstParser
from util import writer, reader


def main(args):
  current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
  parser_model_name = f"{args.language}_without_srl_on_dev_{current_time}"
  logging.info(f"Parser model name is {parser_model_name}")
  # model_name_check = input("Are you happy with the model name: y/n?")
  # if model_name_check != "y":
  #  raise ValueError("Model name is not set properly!")
  log_dir = f"debug/label_first_parser/{args.language}/{parser_model_name}"
  logging.info(f"Logging to {log_dir}")

  train_data_dir = f"./data/propbank/ud/srl/"
  test_data_dir = f"./data/propbank/ud/srl/"
  train_treebank_name = args.train_treebank
  test_treebank_name = args.test_treebank

  if "pos" in args.features:
    if args.language != "tr":
      logging.info(f"Generating pos label to id dictionary for {args.language}..")
      pos_to_id = pos_tag_to_id.postags[args.language]
      pos_embedding_vocab_size=len(pos_to_id.keys())
      print(f"{args.language} pos to id {pos_to_id}")
      print(f"pos vocab size ", pos_embedding_vocab_size)
    else:
      pos_embedding_vocab_size = 38 # for turkish.
      pos_to_id=None
  else:
    pos_to_id=None
    pos_embedding_vocab_size=None

  # loading pretrained word embeddings
  logging.info(f"Loading word embeddings for {args.language}")
  word_embeddings = load_models.load_word_embeddings()

  # initialize preprocessor
  prep = load_models.load_preprocessor_v2(word_embedding_indexes=word_embeddings.token_to_index,
                                          pos_indexes=pos_to_id,
                                          language=args.language,
                                          features=args.features,
                                          embedding_type=args.embeddings)
  logging.info(f"features used to train are {args.features}")
  logging.info(f"parser will output predictions for {args.predict}")
  print(prep.word_embedding_indexes["test"])


  label_feature = next(
    (f for f in prep.sequence_features_dict.values() if f.name == "dep_labels"),
    None)
  print(f"Label feature is: {label_feature}")

  # initialize the parser
  logging.info("Initializing parser")
  parser = LabelFirstParser(word_embeddings=word_embeddings,
                            language=args.language,
                            n_output_classes=label_feature.n_values,
                            predict=args.predict,
                            features=args.features,
                            log_dir=log_dir,
                            test_every=10,
                            model_name=parser_model_name,
                            pos_embedding_vocab_size=pos_embedding_vocab_size,
                            one_hot_labels=False)


  # load tf.datasets
  logging.info("Loading datasets")
  train_sentences = prep.prepare_sentence_protos(
    path=os.path.join(train_data_dir, train_treebank_name)
  )
  train_dataset = prep.make_dataset_from_generator(
    sentences=train_sentences, batch_size=100
  )
  test_sentences = prep.prepare_sentence_protos(
    path=os.path.join(test_data_dir, test_treebank_name)
  )
  test_dataset = prep.make_dataset_from_generator(
    sentences = test_sentences,
    batch_size=1
  )
  # for example in train_dataset:
  #  print(example)
  #  input()
  """
  for example in train_dataset:
    print(example)
    input()
    gold_sentence_pb2 = gold_treebank.sentence.add()
    # parsed_sentence_pb2 = parsed_treebank.sentence.add()
    sent_id, tokens, dep_labels, heads = (example["sent_id"], example["tokens"],
                                          example["dep_labels"], example["heads"])
    srl = example["srl"]
    # first populate gold treebank with the gold annotations
    index = 0
    # print("gold labels ", dep_labels)
    # input()
    # for token, dep_label, head in zip(tokens[0], dep_labels[0], heads[0]):
    for token, dep_label, head, srl in zip(tokens[0], dep_labels[0], heads[0], srl[0]):
      print("token ", token, "dep label ", dep_label , "head ", head, "srl", srl)
      input()
      gold_sentence_pb2.sent_id = sent_id[0][0].numpy()
      token = gold_sentence_pb2.token.add(
        word=tf.keras.backend.get_value(token),
        # label=self._label_index_to_name(tf.keras.backend.get_value(dep_label)),
        index=index)
      token.selected_head.address=tf.keras.backend.get_value(head)
      srl_indexes = tf.where(srl).numpy()
      print(srl_indexes)
      input()
      for srl_index in srl_indexes:
        id = int(tf.keras.backend.get_value(srl_index))
        print("id ", id)
        tag = srl_to_id.id[id]
        token.srl.append(tag)
      index += 1
      print(token)
      input()
  """
  # train the parser
  metrics = parser.train(dataset=train_dataset, epochs=90, test_data=test_dataset)
  print(metrics)

  # write metrics
  writer.write_proto_as_text(metrics, f"./model/nn/plot/final/{parser_model_name}_metrics.pbtxt")

  # save parser
  parser.save_weights()
  logging.info("weights saved!")

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
                      default="word2vec")
  args = parser.parse_args()
  main(args)
