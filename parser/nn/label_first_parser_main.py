import logging
import numpy as np
import datetime

from parser.nn import load_models
from parser.nn.label_first_parser import LabelFirstParser
from util import writer
from util.nn import nn_utils

if __name__ == "__main__":
  # use_pretrained_weights_from_labeler = True
  current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
  log_dir = "debug/label_first_parser/" + current_time
  word_embeddings = load_models.load_word_embeddings()
  prep = load_models.load_preprocessor(word_embeddings,
                                       # one_hot_features=["dep_labels"]
                                       )

  label_feature = next(
    (f for f in prep.sequence_features_dict.values() if f.name == "dep_labels"), None)

  parser_model_name = "lfp_gold_labels_pos_morph_boun_no_dense"
  parser = LabelFirstParser(word_embeddings=prep.word_embeddings,
                            n_output_classes=label_feature.n_values,
                            predict=["heads",
                                     "labels"
                                     ],
                            features=["words",
                                      # "pos",
                                      # "morph",
                                      # "category",
                                      "heads",
                                      # "dep_labels"
                                      ],
                            log_dir=log_dir,
                            test_every=1,
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
  train_treebank= "tr_boun-ud-train.tfrecords"
  test_treebank = "tr_boun-ud-test.tfrecords"
  train_dataset, test_dataset = load_models.load_data(preprocessor=prep,
                                                      train_treebank=train_treebank,
                                                      batch_size=2,
                                                      test_treebank=test_treebank,
                                                      type="tfrecords")


  # for batch in train_dataset:
  #   print(batch)
  # input()
  metrics = parser.train(dataset=train_dataset, epochs=50, test_data=test_dataset)
  print(metrics)
  writer.write_proto_as_text(metrics, f"./model/nn/plot/final/{parser_model_name}_metrics.pbtxt")
  nn_utils.plot_metrics(name=parser_model_name, metrics=metrics)
  parser.save_weights()
  logging.info("weights saved!")
