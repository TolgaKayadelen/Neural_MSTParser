import os
import logging

# from parser.nn.bilstm_labeler import BiLSTMLabeler
from parser.nn.bilstm_labeler_exp import BiLSTMLabeler
from util import converter, writer
from util.nn import nn_utils
from input import embeddor, preprocessor
from parser.nn import load_models


if __name__ == "__main__":
  word_embeddings = load_models.load_word_embeddings()
  prep = load_models.load_preprocessor(word_embeddings)
  parser = BiLSTMLabeler(word_embeddings=prep.word_embeddings,
                         n_output_classes=8,
                         predict=["labels"],
                         features=["words", "pos", "morph"],
                         model_name="bilstm_labeler_reduced_tags",
                         test_every=5)

  # LOADING A PRETRAINED PARSER AND PARSING WITH THAT.
  # parser.load_weights(name="dependency_labeler") # uncomment
  # for w in parser.model.weights:
  #   print(type(w))
  # print(parser.model.weights[-2])
  # weights = parser.model.get_weights() # Uncomment
  # print("weights are ", weights)
  # input("press to cont.")
  #for layer in parser.model.layers:
  #  print(layer.name)
  #  if layer.name == "word_embeddings":
  #    print("working the labels layer")
  #    input("press to cont.")
  #    trainable_weights = layer.trainable_weights
  #    print("trainable weights ", trainable_weights)
  # print(tf.math.reduce_sum(trainable_weights[0], axis=0))
  #print("parser ", parser)
  #input("press to cont.")

  train_treebank= "tr_boun-ud-train.tfrecords"
  test_treebank = "tr_boun-ud-test.tfrecords"
  train_dataset, test_dataset = load_models.load_data(preprocessor=prep,
                                                      train_treebank=train_treebank,
                                                      batch_size=250,
                                                      test_treebank=test_treebank,
                                                      type="tfrecords")

  _metrics = parser.train(dataset=train_dataset, epochs=200, test_data=test_dataset)
  print(_metrics)
  writer.write_proto_as_text(metrics,
                             f"./model/nn/plot/{parser.model_name}_metrics.pbtxt")
  # nn_utils.plot_metrics(name=parser.model_name, metrics=metrics)
  # parser.save_weights()
  # logging.info(f"{parser.model_name} results written")
