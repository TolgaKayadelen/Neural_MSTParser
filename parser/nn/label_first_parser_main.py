import logging
import numpy as np

from parser.nn import load_models
from parser.nn.label_first_parser import LabelFirstParser
from util import writer
from util.nn import nn_utils

if __name__ == "__main__":
  use_pretrained_weights_from_labeler = True
  word_embeddings = load_models.load_word_embeddings()
  prep = load_models.load_preprocessor(word_embeddings)

  labeler, label_feature = load_models.load_labeler("dependency_labeler", prep)

  parser_model_name = "dependency_parser_label_first"
  parser = LabelFirstParser(word_embeddings=prep.word_embeddings,
                            n_output_classes=label_feature.n_values,
                            predict=["heads"],
                            features=["words", "pos", "morph", "heads", "dep_labels"],
                            model_name=parser_model_name)

  # print(parser.model.pos_embeddings.weights)
  # input("press to cont.")
  # print(labeler.model.pos_embeddings.weights)
  # input("press to cont.")
  parser.model.pos_embeddings.set_weights(labeler.model.pos_embeddings.get_weights())
  # print("parser pos embeddings after transfer ")
  # print(parser.model.pos_embeddings.weights)
  # input("press to cont.")

  for a, b in zip(parser.model.pos_embeddings.weights, labeler.model.pos_embeddings.weights):
    np.testing.assert_allclose(a.numpy(), b.numpy())

  parser.model.pos_embeddings.trainable = False
  parser.model.word_embeddings.trainable = False
  print("pos emb. trainable ", parser.model.pos_embeddings.trainable)
  print("word emb. trainble ", parser.model.word_embeddings.trainable)


  # get the data
  train_treebank = "tr_imst_ud_train_dev.pbtxt" # "treebank_train_0_50.pbtxt"
  test_treebank = "tr_imst_ud_test_fixed.pbtxt" # "treebank_test_0_10.conllu"
  train_dataset, test_dataset = load_models.load_data(preprocessor=prep,
                                                      train_treebank=train_treebank,
                                                      batch_size=250,
                                                      test_treebank=test_treebank)

  metrics = parser.train(dataset=train_dataset, epochs=70, test_data=test_dataset)
  print(metrics)
  writer.write_proto_as_text(metrics, f"./model/nn/plot/{parser_model_name}_metrics.pbtxt")
  nn_utils.plot_metrics(name=parser_model_name, metrics=metrics)
  parser.save_weights()
  logging.info("weights saved!")
