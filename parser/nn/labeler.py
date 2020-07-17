"""A BiLSTM based labeler to a sequence of tokens."""

import os
import sys
import argparse

from util import reader
from data.treebank import sentence_pb2
from tagset.fine_pos import fine_tag_enum_pb2 as fine_tags
from tagset.coarse_pos import coarse_tag_enum_pb2 as coarse_tags
from tagset.dep_labels import dep_label_enum_pb2 as dep_labels
from tagset.arg_str import semantic_role_enum_pb2 as srl
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
      vld_path = os.path.join(_DATA_DIR, "Turkish", "test", "{}.pbtxt".format(self.vld_data))
      vld_treebank = reader.ReadTreebankTextProto(vld_path)
      validation_data = list(vld_treebank.sentence)
      logging.info("Total sentences in validation data {}".format(len(validation_data)))
    
    if self.test_data:
      test_path = os.path.join(_DATA_DIR, "Turkish", "test", "{}.pbtxt".format(self.test_data))
      test_treebank = reader.ReadTreebankTextProto(test_path)
      testing_data = list(test_treebank.sentence)
      logging.info("Total sentences in test data {}".format(len(testing_data)))
    
    
    return training_data, validation_data, testing_data
  
  def _read_labels(self, tagset):
    
    # read the tagset
    assert(tagset), "Tagset cannot be initialized!"
    if tagset == "fine_pos":
      tags = fine_tags
    elif tagset == "coarse_pos":
      tags = coarse_tags
    elif tagset == "dep_labels":
      tags = dep_labels
    elif tagset == "semantic_roles":
      tags = srl
    else:
      sys.exit("Couldn't determine tagset!")
    
    def _get_bio_tags_from_srl():
      labels_list = ["-pad-"]
      for key in tags.Tag.DESCRIPTOR.values_by_name.keys():
        if key == "UNKNOWN_SRL":
          continue
        if key.startswith(("A_", "AM_", "C_", "R_")):
          key = key.replace("_", "-")
        labels_list.extend(["B-"+key, "I-"+key])
      labels_list.append("O")
      return {v: k for k, v in enumerate(labels_list)}
  
    
    if tags == srl:
      label_dict = _get_bio_tags_from_srl()
    else:
      for key in tags.Tag.DESCRIPTOR.values_by_name.keys():
        if key in ["UNKNOWN_TAG", "UNKNOWN_CATEGORY", "UNKNOWN_LABEL"]:
          continue
        if key in {"advmod_emph", "aux_q", "compound_lvc", "compound_redup", "nmod_poss"}:
          label_dict[key.replace("_", ":")] = tags.Tag.Value(key)
        else:
          label_dict[key] = tags.Tag.Value(key)
    label_dict["-pad-"] = 0
    
    logging.info(f"number of labels: {len(label_dict)}")
    return label_dict
  
  def _get_sentences_and_labels(self, data, tagset):
    """Returns sentences and labels from training data.
    
    Args:
      data: list, a list of sentence_pb2.Sentence() objects. 
    Returns:
      sentences: a list of lists where each list is a list of words.
      labels: a list of lists where each list is a list of labels. 
    """
    assert(tagset in ["fine_pos", "coarse_pos", "dep_labels"]), "Invalid Tagset!"
    sentences, labels = [], []
    for sentence in data:
      words = [token.word for token in sentence.token]
      if tagset == "fine_pos":
        labels_ = [token.pos for token in sentence.token]
      elif tagset == "coarse_pos":
        labels_ = [token.category for token in sentence.token]
      else:
        sentence.token[0].label = "TOP"
        labels_ = [token.label for token in sentence.token]
      sentences.append(words)
      labels.append(labels_)
    return sentences, labels
    
    
  def train(self, epochs=20, learning_rate=0.2, batch_size=50, tagset=None):
    
    # Initialize the learner.
    learner = bilstm.BiLSTM()
    
    # Get the labels
    label_dict = self._read_labels(tagset)
    print(label_dict)    
    logging.info(f"number of labels: {len(label_dict)}")
    input("Press to continue..")
    
    # Get the training, validation and test data.
    train_data, vld_data, test_data = self._read_data()
    train_sentences, train_labels = self._get_sentences_and_labels(train_data, tagset)
    vld_sentences, vld_labels = self._get_sentences_and_labels(vld_data, tagset)
    
    if test_data:
      test_sentences, test_labels = self._get_sentences_and_labels(test_data, tagset)
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
  parser.add_argument("--tagset", type=str,
                      choices=["fine_pos", "coarse_pos", "dep_labels", "semantic_roles"],
                      help="path to tagset proto file.")
  
  args = parser.parse_args()
  labeler = Labeler(train_data=args.train_data, vld_data=args.vld_data,
                    test_data=args.test_data)
  labeler.train(args.epochs, args.learning_rate, args.batch_size, args.tagset)