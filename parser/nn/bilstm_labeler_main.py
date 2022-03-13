import os
import logging

from parser.nn.bilstm_labeler import BiLSTMLabeler
from util import converter, writer
from util.nn import nn_utils
from input import embeddor, preprocessor

if __name__ == "__main__":
  embeddings = nn_utils.load_embeddings()
  word_embeddings = embeddor.Embeddings(name= "word2vec", matrix=embeddings)
  prep = preprocessor.Preprocessor(
    word_embeddings=word_embeddings,
    features=["words", "pos", "morph", "heads", "dep_labels"],
    labels=["heads"]
  )
  label_feature = next(
    (f for f in prep.sequence_features_dict.values() if f.name == "dep_labels"), None)

  parser = BiLSTMLabeler(word_embeddings=prep.word_embeddings,
                         n_output_classes=label_feature.n_values,
                         predict=["labels"],
                         features=["words", "pos", "morph"],
                         model_name="dependency_labeler_test")

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

  _DATA_DIR="data/UDv23/Turkish/training"
  _TEST_DATA_DIR="data/UDv23/Turkish/test"
  train_treebank="tr_imst_ud_train_dev.pbtxt"
  test_treebank = "tr_imst_ud_test_fixed.pbtxt"
  train_sentences = prep.prepare_sentence_protos(
    path=os.path.join(_DATA_DIR, train_treebank))
  dataset = prep.make_dataset_from_generator(
    sentences=train_sentences,
    batch_size=250
  )
  if test_treebank is not None:
    test_sentences = prep.prepare_sentence_protos(
      path=os.path.join(_TEST_DATA_DIR, test_treebank))
    test_dataset = prep.make_dataset_from_generator(
      sentences=test_sentences,
      batch_size=1)
  else:
    test_dataset=None
  # for batch in test_dataset:      # uncomment for testing loading
  #  scores = parser.parse(batch)   # uncomment for testing loading
  #  print(scores)                  # uncomment for testing loading

  metrics = parser.train(dataset=dataset, epochs=70,
                         test_data=test_dataset)
  # metrics = parser.test(dataset=test_dataset)
  print(metrics)
  writer.write_proto_as_text(metrics,
                             f"./model/nn/plot/{parser.model_name}_metrics.pbtxt")
  nn_utils.plot_metrics(name=parser.model_name, metrics=metrics)
  parser.save_weights()
  logging.info(f"{parser.model_name} results written")
