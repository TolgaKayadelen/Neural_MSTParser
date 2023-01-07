import logging
import datetime
import sys
import tensorflow as tf

from parser.labeler.bilstm.bilstm_labeler import BiLSTMLabeler
from util import converter, writer
from util.nn import nn_utils
from input import embeddor, preprocessor
from parser.utils import load_models


if __name__ == "__main__":
  current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
  labeler_model_name = "bilstm_labeler_topk_dev_treebank"
  log_dir = f"debug/bilstm_labeler/{labeler_model_name}/{current_time}"
  response = input(f"Are you happy with the model name {labeler_model_name}: y/n")
  if response != "y":
    sys.exit("Exiting! Please set your model name.")


  word_embeddings = load_models.load_word_embeddings()
  prep = load_models.load_preprocessor(word_embeddings=word_embeddings)

  label_feature = next(
    (f for f in prep.sequence_features_dict.values() if f.name == "dep_labels"), None)

  parser = BiLSTMLabeler(word_embeddings=prep.word_embeddings,
                         n_output_classes=label_feature.n_values,
                         predict=["labels"],
                         features=["words", "pos", "morph"],
                         model_name=labeler_model_name,
                         log_dir=log_dir,
                         top_k=False,
                         k=5,
                         test_every=5)

  """
  # UNCOMMENT TO LOAD A PRETRAINED PARSER AND PARSE WITH THAT.
  parser.load_weights(name="bilstm_labeler_topk") # uncomment
  for w in parser.model.weights:
    print(type(w))
  print(parser.model.weights[-2])
  weights = parser.model.get_weights() # Uncomment
  # print("weights are ", weights)
  for layer in parser.model.layers:
    # print(f"layer name {layer.name}, trainable {layer.trainable}")
    if layer.name == "word_embeddings":
      trainable_weights = layer.trainable_weights
      # print("word embedding trainable weights ", trainable_weights)
  # print(tf.math.reduce_sum(trainable_weights[0], axis=0))
  print("parser ", parser)
  """

  train_treebank= "tr_boun-ud-train-random500.pbtxt"
  dev_treebank = "tr_boun-ud-dev.pbtxt"
  test_treebank = "tr_boun-ud-test.pbtxt"
  train_dataset, dev_dataset, test_dataset = load_models.load_data(preprocessor=prep,
                                                                   train_treebank=train_treebank,
                                                                   batch_size=5,
                                                                   dev_treebank=dev_treebank,
                                                                   dev_batch_size=20,
                                                                   test_treebank=test_treebank,
                                                                   type="pbtxt")

  metrics = parser.train(dataset=dev_dataset, epochs=30, test_data=test_dataset)
  print(metrics)
  # writer.write_proto_as_text(metrics,
  #                            f"./model/nn/plot/{parser.model_name}_metrics.pbtxt")
  parser.save_weights()
  logging.info(f"{parser.model_name} results written")
