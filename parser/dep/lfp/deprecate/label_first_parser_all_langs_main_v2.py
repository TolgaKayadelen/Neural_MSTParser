# Use like:
# bazel-bin/parser/dep/lfp/label_first_parser_main \
# --language=en --predict heads labels --train_treebank=en_ewt-ud-train.pbtxt \
# --test_treebank=en_ewt-ud-test.pbtxt --features words pos dep_labels


import sys
import logging

import datetime


from parser.utils import load_models, pos_tag_to_id
from parser.dep.lfp.label_first_parser_v2 import LabelFirstParser
from util import writer, reader


class Args:
  def __init__(self, language, features, predict, train_treebank, test_treebank, embeddings):
    self.language = language
    self.features = features
    self.predict = predict
    self.train_treebank = train_treebank
    self.test_treebank = test_treebank
    self.embeddings = embeddings

  def __str__(self):
    return f"""language: {self.language},
               features: {self.features},
               predict: {self.predict},
               train_treebank: {self.train_treebank},
               test_treebank: {self.test_treebank},
               embeddings: {self.embeddings}"""


def main(args):
  current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
  # parser_model_name = f"{args.language}_lfp_predicted_head_gold_labels_only_{current_time}"
  parser_model_name = f"{args.language}_lfp_nobert_predicted_pos_heads_labels_{current_time}"
  logging.info(f"Parser model name is {parser_model_name}")
  # model_name_check = input("Are you happy with the model name: y/n?")
  # if model_name_check != "y":
  #   raise ValueError("Model name is not set properly!")
  log_dir = f"debug/label_first_parser/{args.language}/{parser_model_name}"
  logging.info(f"Logging to {log_dir}")

  train_data_dir = f"./data/UDv29/train/{args.language}"
  test_data_dir = f"./data/UDv29/test/{args.language}"
  train_treebank_name = args.train_treebank
  test_treebank_name = args.test_treebank

  if "pos" in args.predict:
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

  # loading pretrained word embeddings
  logging.info(f"Loading word embeddings for {args.language}")
  word_embeddings = load_models.load_i18n_embeddings(language=args.language)

  # initialize preprocessor
  prep = load_models.load_preprocessor(word_embedding_indexes=word_embeddings.token_to_index,
                                       pos_indexes=pos_to_id,
                                       language=args.language,
                                       features=args.features,
                                       embedding_type=args.embeddings)
  logging.info(f"features used to train are {args.features}")
  logging.info(f"parser will output predictions for {args.predict}")
  print(prep.word_embedding_indexes[b"test"])


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
                            test_every=3,
                            model_name=parser_model_name,
                            pos_embedding_vocab_size=pos_embedding_vocab_size,
                            one_hot_labels=False)

  # load tf.datasets
  logging.info("Loading datasets")
  train_dataset, _, test_dataset = load_models.load_data(preprocessor=prep,
                                                        train_treebank=train_treebank_name,
                                                        batch_size=200,
                                                        test_treebank=test_treebank_name,
                                                        type="pbtxt",
                                                        language=args.language)
  # for batch in train_dataset:
  #   print(batch)
  #  input()

  # train the parser
  metrics = parser.train(dataset=train_dataset, epochs=100, test_data=test_dataset)
  print(metrics)

  # write metrics
  writer.write_proto_as_text(metrics, f"./model/nn/plot/final/{parser_model_name}_metrics.pbtxt")

  # save parser
  parser.save_weights()
  logging.info("weights saved!")

if __name__ == "__main__":

  languages = ["zh", "fi", "ko", "ru", "de", "en"]
  train_treebanks = ["zh_gsd-ud-train.pbtxt", "fi_tdt-ud-train.pbtxt", "ko_gsd-ud-train.pbtxt",
                     "ru_gsd-ud-train.pbtxt", "de_gsd-ud-train.pbtxt", "en_ewt-ud-train.pbtxt"]
  test_treebanks = ["zh_gsd-ud-test.pbtxt", "fi_tdt-ud-test.pbtxt", "ko_gsd-ud-test.pbtxt",
                    "ru_gsd-ud-test.pbtxt", "de_gsd-ud-test.pbtxt", "en_ewt-ud-test.pbtxt"]
  for language, train_treebank, test_treebank in zip(languages, train_treebanks, test_treebanks):
    if language in ["ko", "fi", "en", "de"]:
      continue
    parse_args = Args(
      language=language,
      features=["words", "pos"], # ["words", "dep_labels"],
      predict=["heads", "labels", "pos"], # ["heads"]
      train_treebank = train_treebank,
      test_treebank= test_treebank,
      embeddings = "conll"
    )
    print(parse_args)
    main(parse_args)

  """
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
  """
