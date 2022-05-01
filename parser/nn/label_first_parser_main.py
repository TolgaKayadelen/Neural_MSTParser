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

  label_feature = next(
    (f for f in prep.sequence_features_dict.values() if f.name == "dep_labels"), None)

  parser_model_name = "label_first_predict_heads_and_labels_boun"
  parser = LabelFirstParser(word_embeddings=prep.word_embeddings,
                            n_output_classes=label_feature.n_values,
                            predict=["heads", "labels"],
                            features=["words", "pos", "morph", "heads", "dep_labels"],
                            model_name=parser_model_name)

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
  test_treebank = "tr_boun-ud-dev.pbtxt"
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
