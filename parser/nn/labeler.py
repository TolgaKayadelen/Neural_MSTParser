"""A BiLSTM based labeler to a sequence of tokens."""

import os
from util import reader
from data.treebank import sentence_pb2
from model import tags_and_labels_enum_pb2 as tags_and_labels
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
    
  def _get_train_and_test_data(self):
    """Returns a list of sentences and the list of labels."""
    train_path = os.path.join(_DATA_DIR, "Turkish", "training", "{}.pbtxt".format(self.train_data))
    train_treebank = reader.ReadTreebankTextProto(train_path)
    logging.info("Total sentences in train data {}".format(len(train_treebank.sentence)))
    training_data = list(train_treebank.sentence)
    test_path = os.path.join(_DATA_DIR, "Turkish", "training", "{}.pbtxt".format(self.test_data))
    test_treebank = reader.ReadTreebankTextProto(test_path)
    logging.info("Total sentences in test data {}".format(len(test_treebank.sentence)))
    test_data = list(test_treebank.sentence)
    
    return training_data, test_data
    
  
  def _get_sentences_and_labels(self, data):
    """Returns sentences and labels from training data.
    
    Args:
      data: list, a list of sentence_pb2.Sentence() objects. 
    Returns:
      sentences: a list of lists where each list is a list of words.
      labels: a list of lists where each list is a list of labels. 
    """
    sentences, labels = [], []
    counter = 0
    for sentence in data:
      words = [token.word for token in sentence.token]
      labels_ = [token.pos for token in sentence.token]
      sentences.append(words)
      labels.append(labels_)
      counter += 1
      if counter > 10:
        break
        
    return sentences, labels
    
    
  def train(self):
    learner = bilstm.BiLSTM()
    train_data, test_data = self._get_train_and_test_data()
    train_sentences, train_labels = self._get_sentences_and_labels(train_data)
    # print(train_sentences, train_labels)
    
    label_dict = {}
    for key in tags_and_labels.Tag.DESCRIPTOR.values_by_name.keys():
      if key == "UNKNOWN_TAG":
        continue
      label_dict[key] = tags_and_labels.Tag.Value(key)
    label_dict["-pad-"] = 0
    print(label_dict)
    #input("Press to start training ..")
    # learner.train(train_data=train_sentences, train_labels=train_labels, label_dict=label_dict,
    #              epochs=self.epochs, embeddings=True, batch_size=10)



if __name__ == "__main__":
  labeler = Labeler(train_data="treebank_train_0_50",
                    test_data="treebank_train_0_10")
                    
  labeler.train()