"""A BiLSTM based labeler to a sequence of tokens."""

import os
from util import reader
from data.treebank import sentence_pb2
from learner.nn import bilstm

import logging
logging.basicConfig(format='%(levelname)s : %(message)s', level=logging.INFO)

_DATA_DIR = "data/UDv23"

class Labeler:
  def __init__(self, train_data, test_data=None, epochs=10, learning_rate=0.2):
    self.epochs = epochs
    self.train_data = train_data
    self.test_data = test_data
    self.learning_rate = learning_rate
    
  def _get_sentences_and_labels(self):
    """Returns a list of sentences and the list of labels."""
    _TRAIN_DATA_DIR = os.path.join(_DATA_DIR, "Turkish", "training")
    _TEST_DATA_DIR = os.path.join(_DATA_DIR, "Turkish", "test")
    train_path = os.path.join(_TRAIN_DATA_DIR, "{}.pbtxt".format(self.train_data))
    train_treebank = reader.ReadTreebankTextProto(train_path)
    logging.info("Total sentences in train data {}".format(len(train_treebank.sentence)))
    training_data = list(train_treebank.sentence)
    test_path = os.path.join(_TEST_DATA_DIR, "{}.pbtxt".format(self.test_data))
    test_treebank = reader.ReadTreebankTextProto(test_path)
    logging.info("Total sentences in test data {}".format(len(test_treebank.sentence)))
    test_data = list(test_treebank.sentence)
    
    return train_data, test_data
    
  
  def train(self):
    labeler = bilstm.BiLSTM()
    sentences, labels = self._get_sentences_and_labels(self.train_data)
    labeler.train(train_data=sentences, train_labels=labels, label_dict=label_dict,
                  epochs=self.epochs, embeddings=True, batch_size=20)



if __name__ == "__main__":
  labeler = Labeler(train_data="treebank_train_0_50",
                    test_data="treebank_train_0_10")
                    
  train_data, test_data = labeler._get_sentences_and_labels()
    