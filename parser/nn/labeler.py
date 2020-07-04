"""A BiLSTM based labeler to a sequence of tokens."""

import os
import argparse
from util import reader
from data.treebank import sentence_pb2
from tagset.fine_pos import fine_tag_enum_pb2 as fine_tags
from learner.nn import bilstm

import logging
logging.basicConfig(format='%(levelname)s : %(message)s', level=logging.INFO)

_DATA_DIR = "data/UDv23"

class Labeler:
  def __init__(self, train_data, vld_data=None, test_data=None):
    self.train_data = train_data
    self.vld_data = vld_data
    self.test_data = test_data
    
  def _read_data(self):
    """Returns a list of sentences and the list of labels."""
    train_path = os.path.join(_DATA_DIR, "Turkish", "training", "{}.pbtxt".format(self.train_data))
    train_treebank = reader.ReadTreebankTextProto(train_path)
    training_data = list(train_treebank.sentence)
    logging.info("Total sentences in train data {}".format(len(training_data)))
    
    validation_data, testing_data = None, None
    if self.vld_data:
      vld_path = os.path.join(_DATA_DIR, "Turkish", "training", "{}.pbtxt".format(self.vld_data))
      vld_treebank = reader.ReadTreebankTextProto(vld_path)
      validation_data = list(vld_treebank.sentence)
      logging.info("Total sentences in validation data {}".format(len(validation_data)))
    
    if self.test_data:
      test_path = os.path.join(_DATA_DIR, "Turkish", "training", "{}.pbtxt".format(self.test_data))
      test_treebank = reader.ReadTreebankTextProto(test_path)
      testing_data = list(test_treebank.sentence)
      logging.info("Total sentences in test data {}".format(len(testing_data)))
    
    
    return training_data, validation_data, testing_data
    
  
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
    return sentences, labels
    
    
  def train(self, epochs=20, learning_rate=0.2, batch_size=50):
    
    # Initialize the learner.
    learner = bilstm.BiLSTM()
  
    # Prepare the dictionary of labels.
    label_dict = {}
    for key in fine_tags.Tag.DESCRIPTOR.values_by_name.keys():
      if key == "UNKNOWN_TAG":
        continue
      label_dict[key] = fine_tags.Tag.Value(key)
    label_dict["-pad-"] = 0
    
    logging.info(f"number of labels: {len(label_dict)}")
    
    # Get the training, validation and test data.
    train_data, vld_data, test_data = self._read_data()
    train_sentences, train_labels = self._get_sentences_and_labels(train_data)
    vld_sentences, vld_labels = self._get_sentences_and_labels(vld_data)
    
    if test_data:
      test_sentences, test_labels = self._get_sentences_and_labels(test_data)
    else:
      test_sentences = vld_sentences
    
    # Start training
    input("Press to start training ..")
    learner.train(train_data=train_sentences, train_data_labels=train_labels, label_dict=label_dict,
                  epochs=epochs, embeddings=True, batch_size=batch_size, 
                  vld_data=vld_sentences, vld_data_labels=vld_labels, test_data=test_sentences)


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  
  parser.add_argument("--epochs", type=int, default=20, help="number of epochs.")
  parser.add_argument("--learning_rate", type=float, default=0.2, help="learning rate")
  parser.add_argument("--train_data", type=str, help="train data path")
  parser.add_argument("--vld_data", type=str, help="validation data path.")
  parser.add_argument("--test_data", type=str, help="test data path.")
  parser.add_argument("--batch_size", type=int, default=50, help="batch size.")
  
  args = parser.parse_args()
  labeler = Labeler(train_data=args.train_data, vld_data=args.vld_data,
                    test_data=args.test_data)
  labeler.train(args.epochs, args.learning_rate, args.batch_size)