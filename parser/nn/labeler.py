"""A BiLSTM based labeler to a sequence of tokens.

Example Use:
Train a postagger.
bazel build //parser/nn:labeler && bazel-bin/parser/nn/labeler \
--train_data=treebank_train_1000_1500 \
--vld_data=treebank_train_0_50 \
--test_data=treebank_0_3_test \
--batch_size=50 \
--tagset=pos

Train a semantic role labeler.
bazel build //parser/nn:labeler && bazel-bin/parser/nn/labeler \
--train_data=propbank_ud_fixed.pbtxt \
--test_data=propbank_ud_test_0_10.pbtxt \
--batch_size=50 \
--tagset=pos
"""



import os
import sys
import argparse
import numpy as np
import random

from util import reader
from util import common
from data.treebank import sentence_pb2
from tagset.fine_pos import fine_tag_enum_pb2 as fine_tags
from tagset.coarse_pos import coarse_tag_enum_pb2 as coarse_tags
from tagset.dep_labels import dep_label_enum_pb2 as dep_labels
from tagset.arg_str import semantic_role_enum_pb2 as srl
from tagset.reader import LabelReader
from learner.nn import bilstm

import logging
logging.basicConfig(format='%(levelname)s : %(message)s', level=logging.INFO)

_DATA_DIR = "data/UDv23"

class Labeler:
  """A sequence labeler class."""
  
  def __init__(self, train_data, vld_data=None, test_data=None, data_dir=_DATA_DIR):
    self.train_data = train_data
    self.vld_data = vld_data
    self.test_data = test_data
    self.data_dir = data_dir
    
    
  def _read_data(self):
    """Returns a list of sentences for training, validation, and test.
    
    Returns:
      list, a list of sentence_pb2.Sentence objects.
    """
    train_path = os.path.join(self.data_dir, "Turkish", "training", "{}.pbtxt".format(self.train_data))
    train_treebank = reader.ReadTreebankTextProto(train_path)
    training_data = list(train_treebank.sentence)
    logging.info("Total sentences in train data {}".format(len(training_data)))
    
    validation_data, testing_data = None, None
    if self.vld_data:
      vld_path = os.path.join(self.data_dir, "Turkish", "test", "{}.pbtxt".format(self.vld_data))
      vld_treebank = reader.ReadTreebankTextProto(vld_path)
      validation_data = list(vld_treebank.sentence)
      logging.info("Total sentences in validation data {}".format(len(validation_data)))
    
    if self.test_data:
      test_path = os.path.join(self.data_dir, "Turkish", "test", "{}.pbtxt".format(self.test_data))
      test_treebank = reader.ReadTreebankTextProto(test_path)
      testing_data = list(test_treebank.sentence)
      logging.info("Total sentences in test data {}".format(len(testing_data)))
    
    return training_data, validation_data, testing_data
  
  def _get_sentences_and_labels(self, data, tagset, maxlen=None):
    """Returns sentences and labels from training data.
    
    Args:
      data: list, a list of sentence_pb2.Sentence() objects. 
      tagset: str, one of [dep_labels, fine_pos, coarse_pos, semantic_roles]
      maxlen: maximum sequence length, this is used for padding data. If not given
            computed on the fly.
    Returns:
      sentences: a list of lists where each list is a list of words.
      labels: a list of lists where each list is a list of labels. 
    """
    assert(tagset in ["pos", "category", "dep_labels", "srl"]), "Invalid Tagset!"
    sentences, labels = [], []
    
    # TODO: need a consistent handling of maxlen and padding of datasets.
    if not maxlen:
      maxlen = common.GetMaxlenSentence(data)
    # print(f"maxlen: {maxlen}")
    # Reading semantic role tags from data require special treatment. 
    # We need to generate a new [sentence, labels] pair for each predicate
    # in the sentence.
    if tagset == "srl":
      predicate_info = []
      for sentence in data:
        if not sentence.argument_structure:
          continue
        words = [token.word for token in sentence.token]
        # Create a new training instance for each predicate-argument structure
        # that exists in the sentence.
        # arg_str = random.choice(sentence.argument_structure)
        for arg_str in sentence.argument_structure:
          sentences.append(words)
          
          labels_ = ["O"] * len(words)
          # Index the word position of the predicate with V.
          labels_[arg_str.predicate_index] = "V"
          # print(f"labels_: {labels_}")
          
          # we also give data about whether a token is predicate or not as
          # additional input to the learner for semantic role labeling.
          predicate_info_ = [0] * len(words)
          predicate_info_[arg_str.predicate_index] = 1
          # pad the rest of the predicate_info_ array with 0s.
          predicate_info_.extend([0] * (maxlen-len(predicate_info_)))
          assert(len(predicate_info_) == maxlen), "Padding error!!"
          # print(f"predicate_info_: {predicate_info_}")
          
          for argument in arg_str.argument:
            labels_[argument.token_index[0]] = "B-"+argument.srl
            for idx in argument.token_index[1:]:
              labels_[idx] = "I-"+argument.srl
          labels.append(labels_)
          predicate_info.append(predicate_info_)
          
      return sentences, labels, predicate_info, maxlen
    
    for sentence in data:
      words = [token.word for token in sentence.token]
      if tagset == "pos":
        labels_ = [token.pos for token in sentence.token]
      elif tagset == "category":
        labels_ = [token.category for token in sentence.token]
      else:
        sentence.token[0].label = "TOP"
        labels_ = [token.label for token in sentence.token]
      sentences.append(words)
      labels.append(labels_)
    return sentences, labels
    
    
  def train(self, epochs=20, learning_rate=0.2, batch_size=50, tagset=None, save_as=None):
    
    # Initialize the learner.
    learner = bilstm.BiLSTM()
    
    # Read data and labels.
    logging.info(f"Getting data and labels..")
    
    # Get the labels
    label_dict = LabelReader.get_labels(tagset).labels
    print(label_dict)
    # label_dict = self._read_labels(tagset)
    
    # Read the training, validation and test data.
    train_data, vld_data, test_data = self._read_data()
    
    # Extract the input sentences, labels, and any other additional input from data.
    # TODO: these data set ups should be handled by the train_test_split() method.
    if tagset == "srl":
      train_sentences, train_labels, predicate_info, maxlen = self._get_sentences_and_labels(train_data, tagset)
      if vld_data:
        vld_sentences, vld_labels, predicate_info_vld, _ = self._get_sentences_and_labels(vld_data, tagset, maxlen)
      else:
        vld_sentences, vld_labels, predicate_info_vld = None, None, None
      if test_data:
        test_sentences, test_labels, predicate_info_test, maxlen = self._get_sentences_and_labels(test_data, tagset, maxlen)
        # print(test_sentences, test_labels, predicate_info_test)
      else:
        test_sentences, test_labels, predicate_info_test = None, None, None
        
    else:
      train_sentences, train_labels = self._get_sentences_and_labels(train_data, tagset)
      if vld_data:
        vld_sentences, vld_labels = self._get_sentences_and_labels(vld_data, tagset)
      else:
        vld_sentences, vld_labels = None, None
      if test_data:
        test_sentences, test_labels = self._get_sentences_and_labels(test_data, tagset)
      else:
        test_sentences = None
    
    # for i,sentence in enumerate(train_sentences):
    # print(sentence, train_labels[i])
    #print(train_labels)
    #print(predicate_info)
    #print(np.array(train_labels).reshape(2, 16, 1))
    
    # Start training
    # input("Press to start training ..")
    additional_input={"name": "predicate_info", "data": predicate_info, "shape": (maxlen, 1)} 
    additional_input_test={"name": "predicate_info", "data": predicate_info_test, "shape": (maxlen, 1)}
    learner.train(train_data=train_sentences,
                  train_data_labels=train_labels,
                  label_dict=label_dict,
                  epochs=epochs, 
                  embeddings=True, 
                  batch_size=len(train_sentences), # TODO: need to make this stochastic.
                  vld_data=vld_sentences, 
                  vld_data_labels=vld_labels, 
                  test_data=test_sentences,
                  additional_input=additional_input,
                  additional_input_test=additional_input_test
    )
    if save_as:
      print(f"Saving model to {save_as}")
      learner.save(save_as)


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  
  parser.add_argument("--epochs", type=int, default=20, help="number of epochs.")
  parser.add_argument("--learning_rate", type=float, default=0.01, help="learning rate")
  parser.add_argument("--train_data", type=str, help="train data path")
  parser.add_argument("--vld_data", type=str, help="validation data path.")
  parser.add_argument("--test_data", type=str, help="test data path.")
  parser.add_argument("--batch_size", type=int, default=50, help="batch size.")
  parser.add_argument("--tagset", type=str,
                      choices=["pos", "category", "dep_labels", "srl"],
                      help="path to tagset proto file.")
  parser.add_argument("--save_model_as", type=str, default=None,
                      help="location to save the model to, just give the filename.")
  
  args = parser.parse_args()
  labeler = Labeler(train_data=args.train_data, vld_data=args.vld_data,
                    test_data=args.test_data)
  labeler.train(args.epochs, args.learning_rate, args.batch_size, args.tagset, args.save_model_as)