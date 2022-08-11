
import os

from util.nn import nn_utils
from input import embeddor, preprocessor
from parser.nn.biaffine_parser import BiaffineParser

if __name__ == "__main__":
  embeddings = nn_utils.load_embeddings()
  word_embeddings = embeddor.Embeddings(name="word2vec", matrix=embeddings)
  prep = preprocessor.Preprocessor(
    word_embeddings=word_embeddings,
    features=["words", "pos", "morph", "heads", "dep_labels"],
    labels=["heads", "dep_labels"]
  )
  label_feature = next(
    (f for f in prep.sequence_features_dict.values() if f.name == "dep_labels"), None)

  parser = BiaffineParser(word_embeddings=prep.word_embeddings,
                          n_output_classes=label_feature.n_values,
                          predict=["heads", "labels"],
                          features=["words", "pos", "morph"],
                          model_name="tests_biaffine_base_parser")

  _DATA_DIR="data/UDv23/Turkish/training"
  _TEST_DATA_DIR="data/UDv23/Turkish/test"
  train_treebank="treebank_train_0_50.pbtxt"
  test_treebank = "treebank_test_0_10.pbtxt"
  train_sentences = prep.prepare_sentence_protos(path=os.path.join(_DATA_DIR,
                                                                   train_treebank))
  dataset = prep.make_dataset_from_generator(
    sentences=train_sentences,
    batch_size=5
  )
  if test_treebank is not None:
    test_sentences = prep.prepare_sentence_protos(path=os.path.join(_TEST_DATA_DIR, test_treebank))
    test_dataset = prep.make_dataset_from_generator(
      sentences=test_sentences,
      batch_size=1
    )
  else:
    test_dataset=None
  metrics = parser.train(dataset=dataset, epochs=10, test_data=test_dataset)
  print(metrics)
