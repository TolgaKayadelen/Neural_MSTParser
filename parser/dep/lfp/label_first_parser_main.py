import logging
import numpy as np
import datetime

from parser.utils import load_models
from parser.dep.lfp.label_first_parser import LabelFirstParser
from util import writer
from util.nn import nn_utils

if __name__ == "__main__":
  # use_pretrained_weights_from_labeler = True
  current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
  parser_model_name = "label_first_gold_morph_and_labels"
  # parser_model_name = "label_first_only_gold_labels"
  logging.info(f"Parser model name is {parser_model_name}")
  model_name_check = input("Did you remember to set the model name properly: y/n?")
  if model_name_check != "y":
    raise ValueError("Model name is not set properly! Please set your model name and restart!")
  log_dir = f"debug/label_first_parser/{parser_model_name}/{current_time}"

  language = "tr"
  if language == "tr":
    word_embeddings = load_models.load_word_embeddings()
  else:
    word_embeddings=None

  prep = load_models.load_preprocessor(word_embeddings=word_embeddings, language=language,
  #                                     one_hot_features=["dep_labels"]
                                       )

  label_feature = next(
    (f for f in prep.sequence_features_dict.values() if f.name == "dep_labels"), None)

  parser = LabelFirstParser(word_embeddings=prep.word_embeddings,
                            language=language,
                            n_output_classes=label_feature.n_values,
                            predict=["heads",
                                     # "labels"
                                     ],
                            features=["words",
                                      # "pos",
                                      "morph",
                                      # "category",
                                      # "heads",
                                      "dep_labels"
                                      ],
                            log_dir=log_dir,
                            test_every=5,
                            model_name=parser_model_name,
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


  # get the data
  train_treebank= "tr_boun-ud-train.pbtxt"

  test_treebank = "tr_boun-ud-test.pbtxt"
  train_dataset, _, test_dataset = load_models.load_data(preprocessor=prep,
                                                      train_treebank=train_treebank,
                                                      batch_size=250,
                                                      test_treebank=test_treebank,
                                                      type="pbtxt",
                                                      language=language)
  # for batch in train_dataset:
  #   print(batch)
  # input()
  metrics = parser.train(dataset=train_dataset, epochs=75, test_data=test_dataset)
  print(metrics)
  writer.write_proto_as_text(metrics, f"./model/nn/plot/final/{parser_model_name}_metrics.pbtxt")
  # nn_utils.plot_metrics(name=parser_model_name, metrics=metrics)
  parser.save_weights()
  logging.info("weights saved!")
