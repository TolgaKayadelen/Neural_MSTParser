# Use like:
# bazel-bin/parser/dep/lfp/label_first_parser_main \
# --language=en --predict heads labels --train_treebank=en_ewt-ud-train.pbtxt \
# --test_treebank=en_ewt-ud-test.pbtxt --features words pos dep_labels


import sys
import logging

import datetime


from parser.utils import load_models, pos_tag_to_id
from parser.dep.lfp.label_first_parser import LabelFirstParser
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
  parser_model_name = f"{args.language}_lfp_predicted_head_gold_labels_only_{current_time}"
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

  # loading pretrained word embeddings
  logging.info(f"Loading word embeddings for {args.language}")
  if args.language == "tr":
    if args.embeddings == "conll":
      word_embeddings = load_models.load_i18n_embeddings(language="tr")
    else:
      word_embeddings = load_models.load_word_embeddings()
  else:
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
                            # predict=["heads",
                            #         # "labels"
                            #         ],
                            # features=["words",
                            #          # "pos",
                            #          "morph",
                            #          # "category",
                            #          # "heads",
                            #          "dep_labels"
                            #          ],
                            predict=args.predict,
                            features=args.features,
                            log_dir=log_dir,
                            test_every=10,
                            model_name=parser_model_name,
                            pos_embedding_vocab_size=pos_embedding_vocab_size,
                            one_hot_labels=False)

  """ Uncomment if you want to load pretrained weights
   labeler, label_feature = load_models.load_labeler("dependency_labeler", prep)
  # print(parser.model.pos_embeddings.weights)
  # print(labeler.model.pos_embeddings.weights)
  parser.model.pos_embeddings.set_weights(labeler.model.pos_embeddings.get_weights())
  # print("parser pos embeddings after transfer ")
  # print(parser.model.pos_embeddings.weights)

  for a, b in zip(parser.model.pos_embeddings.weights, labeler.model.pos_embeddings.weights):
    np.testing.assert_allclose(a.numpy(), b.numpy())

  parser.model.pos_embeddings.trainable = False
  parser.model.word_embeddings.trainable = False
  print("pos emb. trainable ", parser.model.pos_embeddings.trainable)
  print("word emb. trainble ", parser.model.word_embeddings.trainable)
  """

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
  # input()

  # train the parser
  metrics = parser.train(dataset=train_dataset, epochs=70, test_data=test_dataset)
  print(metrics)

  # write metrics
  writer.write_proto_as_text(metrics, f"./model/nn/plot/final/{parser_model_name}_metrics.pbtxt")

  # save parser
  parser.save_weights()
  logging.info("weights saved!")

if __name__ == "__main__":

  languages = ["zh", "fi", "ko", "ru", "de"]
  train_treebanks = ["zh_gsd-ud-train.pbtxt", "fi_tdt-ud-train.pbtxt", "ko_gsd-ud-train.pbtxt",
                     "ru_gsd-ud-train.pbtxt", "de_gsd-ud-train.pbtxt"]
  test_treebanks = ["zh_gsd-ud-test.pbtxt", "fi_tdt-ud-test.pbtxt", "ko_gsd-ud-test.pbtxt",
                    "ru_gsd-ud-test.pbtxt", "de_gsd-ud-test.pbtxt"]
  for language, train_treebank, test_treebank in zip(languages, train_treebanks, test_treebanks):
    if language in ["zh", "fi", "ko"]:
      continue
    parse_args = Args(
      language=language,
      features=["words", "dep_labels"],
      predict=["heads"],
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
