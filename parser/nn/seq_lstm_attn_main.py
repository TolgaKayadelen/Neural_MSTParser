import os
import logging

from parser.nn.seq_lstm_attn import SeqLSTMAttnLabeler
from util import converter, writer
from util.nn import nn_utils
from input import embeddor, preprocessor

if __name__ == "__main__":
  embeddings = nn_utils.load_embeddings()
  word_embeddings = embeddor.Embeddings(name="word2vec", matrix=embeddings)
  prep = preprocessor.Preprocessor(
    word_embeddings=word_embeddings,
    features=["words", "pos", "morph", "heads", "dep_labels"],
    labels=["heads"]
  )
  label_feature = next(
    (f for f in prep.sequence_features_dict.values() if f.name == "dep_labels"), None)

  parser = SeqLSTMAttnLabeler(word_embeddings=prep.word_embeddings,
                              n_output_classes=label_feature.n_values,
                              predict=["labels"],
                              features=["words", "pos", "morph"],
                              model_name="seq_lstm_attn_labeler",
                              test_every=5)

  _DATA_DIR="data/UDv29/train/tr"
  _TEST_DATA_DIR="data/UDv29/test/tr"
  train_treebank = "tr_boun-ud-train-random500.pbtxt"
  test_treebank = None # "tr_boun-ud-test-random50.pbtxt"
  train_sentences = prep.prepare_sentence_protos(
    path=os.path.join(_DATA_DIR, train_treebank))
  dataset = prep.make_dataset_from_generator(
    sentences=train_sentences,
    batch_size=50
  )
  if test_treebank is not None:
    test_sentences = prep.prepare_sentence_protos(
      path=os.path.join(_TEST_DATA_DIR, test_treebank))
    test_dataset = prep.make_dataset_from_generator(
      sentences=test_sentences,
      batch_size=3)
  else:
    test_dataset=None

  metrics = parser.train(dataset=dataset, epochs=400,
                         test_data=test_dataset)
  print(metrics)
  # writer.write_proto_as_text(metrics,
  #                           f"./model/nn/plot/{parser.model_name}_metrics.pbtxt")
  # nn_utils.plot_metrics(name=parser.model_name, metrics=metrics)
  # parser.save_weights()
  # logging.info(f"{parser.model_name} results written")